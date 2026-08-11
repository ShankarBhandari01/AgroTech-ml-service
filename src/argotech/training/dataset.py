"""Build a real, non-circular training set for the agronomic risk model.

What changed and why
--------------------
The previous builder synthesised 20 of its 24 features with `np.random` and then derived the label
from an `if/else` over the other four. The model could only ever recover the rule it was handed.

This builder uses only measurements:

* **Features** — ERA5 daily reanalysis over the 90 days *before* the prediction date, passed through
  `argotech.domain` (thermal time, season onset, FAO-56 water balance, dry spells, heat stress), plus
  the Sentinel-2 canopy state on the prediction date.
* **Label** — the Sentinel-2 NDVI anomaly *30 days after* the prediction date, standardised against
  the concurrent cohort of sites in the same farming cluster.

The label is a future satellite observation of a different field-state than the features describe,
so there is no path by which a feature determines its own target. The task is genuinely hard, and
the honest consequence is that the reported scores are far below the previous pipeline's ~0.99.

Leakage control
---------------
* Features come strictly from `t - 90 … t`; the label strictly from `t + 30`.
* VCI uses only this site's observations *before* `t`.
* The peer z-score is cross-sectional — computed against other sites' observations in the *same*
  interval — so it needs no historical baseline that could carry future information.
* The rainfall climatology used for `rain_anomaly_30` excludes the sample's own year.

Run: `python -m argotech.training.dataset --sites 30 --years 4`
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from argotech.data import meteo
from argotech.data.sentinel import sentinel_client
from argotech.features.agronomic import FEATURE_COLUMNS, WINDOW_DAYS, build, satellite_block

# Real Sub-Saharan farming zones. Each spans >100 km so that rainfall genuinely varies within a
# cluster — a cohort whose members all share one weather cell would make the peer label unlearnable.
CLUSTERS = [
    {"name": "Kaduna_Grain_Belt",   "lat": (10.2, 11.5), "lon": (7.3, 8.5)},
    {"name": "Kano_Sudan_Savannah", "lat": (11.5, 12.6), "lon": (8.0, 9.2)},
    {"name": "Benue_River_Basin",   "lat": (7.0, 8.1),   "lon": (8.2, 9.4)},
    {"name": "Kenya_Rift_Valley",   "lat": (-0.2, 1.2),  "lon": (34.8, 36.1)},
    {"name": "Ethiopian_Highlands", "lat": (8.3, 9.9),   "lon": (38.0, 39.6)},
    {"name": "Tanzania_Morogoro",   "lat": (-8.0, -6.5), "lon": (36.4, 37.8)},
]

LABEL_HORIZON_INTERVALS = 1     # one P30D bucket ahead
SEVERE_Z, ELEVATED_Z = -1.0, -0.35

CACHE_DIR = Path(".cache/sentinel")


def sample_sites(per_cluster: int, seed: int = 7) -> list[dict]:
    """Deterministic pseudo-random sites. A fixed lattice jitter rather than `np.random.uniform`
    so that adding sites later extends the set instead of reshuffling it."""
    sites = []
    for c_idx, cluster in enumerate(CLUSTERS):
        lat0, lat1 = cluster["lat"]
        lon0, lon1 = cluster["lon"]
        for i in range(per_cluster):
            # Halton-ish low-discrepancy fill: better spatial coverage than uniform draws at small n.
            u = _radical_inverse(i + seed, 2)
            v = _radical_inverse(i + seed, 3)
            sites.append({
                "site_id": f"{cluster['name']}-{i:03d}",
                "cluster": cluster["name"],
                "cluster_idx": c_idx,
                "latitude": round(lat0 + u * (lat1 - lat0), 4),
                "longitude": round(lon0 + v * (lon1 - lon0), 4),
            })
    return sites


def _radical_inverse(n: int, base: int) -> float:
    out, denom = 0.0, 1.0
    while n > 0:
        denom *= base
        out += (n % base) / denom
        n //= base
    return out


def _sentinel_history(site: dict, days: int) -> list[dict]:
    """Sentinel history with an on-disk cache. Training touches every site repeatedly across
    experiments; the CDSE free tier should be spent once."""
    path = CACHE_DIR / f"{site['site_id']}-{days}.json"
    if path.exists():
        return json.loads(path.read_text())
    obs = sentinel_client.fetch_history(site["latitude"], site["longitude"], days=days, res_m=60)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obs))
    return obs


def collect_site(site: dict, years: int) -> dict | None:
    """Fetch both upstreams for one site. Returns None when either is unusable."""
    days = years * 365
    start = (date.today() - timedelta(days=days + WINDOW_DAYS + 10)).isoformat()
    end = (date.today() - timedelta(days=6)).isoformat()   # ERA5 lags ~5 days

    payload = meteo.fetch_archive(site["latitude"], site["longitude"], start, end)
    history = _sentinel_history(site, days)
    if not payload or len(history) < 6:
        return None

    return {
        **site,
        "elevation": payload.get("elevation", 0.0),
        "daily": meteo.daily_frame(payload),
        "history": history,
    }


def _climatological_rain_30(daily: dict, end_idx: int, exclude_year: str) -> float:
    """Mean 30-day rainfall over the same calendar window in *other* years at this site."""
    times = daily["time"]
    rain = daily["precipitation_sum"]
    target = times[end_idx][5:]                          # MM-DD
    totals = []
    for i, ts in enumerate(times):
        if ts[5:] != target or ts[:4] == exclude_year or i < 30:
            continue
        totals.append(sum(rain[i - 30:i]))
    return statistics.fmean(totals) if totals else sum(rain[max(0, end_idx - 30):end_idx])


def build_samples(sites: list[dict]) -> pd.DataFrame:
    """Cross-join sites with their satellite observation dates, and label each from t+30."""
    # Cohort NDVI per (cluster, sensing_date) for the cross-sectional peer z-score.
    cohort: dict[tuple[str, str], list[float]] = {}
    for s in sites:
        for obs in s["history"]:
            cohort.setdefault((s["cluster"], obs["sensing_date"]), []).append(obs["ndvi"])

    rows = []
    for s in sites:
        times = s["daily"]["time"]
        time_index = {t: i for i, t in enumerate(times)}
        history = s["history"]

        for k, obs in enumerate(history):
            future = history[k + LABEL_HORIZON_INTERVALS:k + LABEL_HORIZON_INTERVALS + 1]
            if not future:
                continue
            label_obs = future[0]

            # Align the satellite date onto the weather series; require a full look-back window.
            end_idx = time_index.get(obs["sensing_date"])
            if end_idx is None or end_idx < WINDOW_DAYS:
                continue

            window = {k2: v[end_idx - WINDOW_DAYS:end_idx] for k2, v in s["daily"].items()}
            peers_now = [v for v in cohort.get((s["cluster"], obs["sensing_date"]), []) if v != obs["ndvi"]]
            past_ndvi = [o["ndvi"] for o in history[:k]]

            sat = satellite_block(obs, past_ndvi, peers_now)
            site_ctx = {
                "latitude": s["latitude"], "longitude": s["longitude"], "elevation": s["elevation"],
                "clim_rain_30": _climatological_rain_30(s["daily"], end_idx, obs["sensing_date"][:4]),
            }
            row = build(window, sat, site_ctx)

            # --- label: peer-standardised NDVI one interval ahead ---
            peers_future = [v for v in cohort.get((s["cluster"], label_obs["sensing_date"]), [])
                            if v != label_obs["ndvi"]]
            if len(peers_future) < 5:
                continue
            mean = statistics.fmean(peers_future)
            sd = statistics.pstdev(peers_future)
            if sd < 1e-6:
                continue
            z = (label_obs["ndvi"] - mean) / sd
            row["label"] = 2 if z <= SEVERE_Z else (1 if z <= ELEVATED_Z else 0)

            row["forward_z"] = round(z, 4)
            row["site_id"] = s["site_id"]
            row["cluster"] = s["cluster"]
            row["obs_date"] = obs["sensing_date"]
            rows.append(row)

    return pd.DataFrame(rows)


def build_dataset(per_cluster: int = 30, years: int = 4, workers: int = 3) -> pd.DataFrame:
    sites = sample_sites(per_cluster)
    print(f"Collecting {len(sites)} sites x {years}y from Open-Meteo ERA5 + Sentinel-2 ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        collected = [s for s in pool.map(lambda s: collect_site(s, years), sites) if s]
    print(f"  {len(collected)}/{len(sites)} sites usable")

    df = build_samples(collected)
    print(f"  {len(df)} labelled samples")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=30, help="sites per cluster")
    ap.add_argument("--years", type=int, default=4)
    ap.add_argument("--out", default="data/training_set.parquet")
    args = ap.parse_args()

    df = build_dataset(args.sites, args.years)
    if df.empty:
        raise SystemExit("no samples built — check Sentinel credentials and network")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"\nSaved {args.out}")
    print(f"Features: {len(FEATURE_COLUMNS)}   Samples: {len(df)}   Sites: {df.site_id.nunique()}")
    print(f"Date range: {df.obs_date.min()} .. {df.obs_date.max()}")
    print("Class distribution:")
    print(df.label.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
