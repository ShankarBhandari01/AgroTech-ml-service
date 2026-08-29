"""E09 Sentinel arm — what displacement costs the optical canopy features.

Counterpart to `weather_arm.py`. The economics are the opposite of weather, and that is the point:

  * ERA5-Land serves a 7.83 x 8.75 km cell = 68.5 km^2, so many displaced draws land in the SAME
    cell and return byte-identical data. Cell-level dedup made M draws nearly free.
  * Sentinel uses a +/-0.005 deg AOI (`data/sentinel.py:204`) = 1.11 x 1.10 km = 1.23 km^2, which is
    56x smaller. There is no dedup: every draw has its own footprint. Measured geometrically over
    the 122 sites, a 5 km displacement leaves an expected **10.5%** AOI overlap and **75.1%** of
    draws share NO pixels at all with the true field.

So geometry already says the displaced AOI is usually a different piece of ground. What it cannot
say is whether the *values* still track, since NDVI at 5 km may correlate through landscape-scale
autocorrelation -- same agro-ecological zone, rainfall and crop calendar. That is the open question
this arm measures.

CACHE HAZARD, handled explicitly: `panel._sentinel_history` keys its cache on `{site_id}-{days}`,
NOT on coordinates. Passing a displaced site that keeps its `site_id` would silently return the TRUE
site's series and produce a perfect, entirely fictional correlation. This module therefore never
calls `panel._sentinel_history`; it calls the client directly and caches under a coordinate-derived
key of its own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from argotech.data.sentinel import SentinelClient
from argotech.lab.panel import panel as panel_mod

from weather_arm import displace

CACHE = Path(".cache/e09_sentinel")
DAYS = 1400          # covers the panel window; matches the res_m=60 convention used by the panel
RES_M = 60


def _key(lat: float, lon: float, days: int, kind: str) -> Path:
    h = hashlib.sha1(f"{kind},{lat:.5f},{lon:.5f},{days},{RES_M}".encode()).hexdigest()
    return CACHE / f"{h}.json"


def fetch_history(client, lat, lon, days, kind="opt"):
    """Coordinate-keyed read-through cache. Never memoises an empty result: `_stats` returns [] for
    both 'no imagery' and 'request failed', and caching that is what previously turned a burst of
    CDSE 429s into 69 permanently band-less sites (`panel._cached_fetch`'s own docstring)."""
    p = _key(lat, lon, days, kind)
    if p.exists():
        return json.loads(p.read_text()), True
    obs = (client.fetch_history(lat, lon, days=days, res_m=RES_M) if kind == "opt"
           else client.fetch_sar_history(lat, lon, days=days, res_m=RES_M))
    if not obs:
        return [], False
    CACHE.mkdir(parents=True, exist_ok=True)
    # temp+rename: the optical and SAR arms can run concurrently against this directory, and a
    # bare write_text lets one process read a half-written file. Same fix as the SoilGrids fetcher.
    tmp = p.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obs))
    tmp.rename(p)
    return obs, False


def match(true_obs: list[dict], disp_obs: list[dict], max_gap_days: int = 10) -> list[tuple]:
    """Pair observations by nearest sensing date. The displaced AOI has its own cloud mask, so the
    two series do not share dates; requiring exact matches would discard nearly everything.
    `max_gap_days` is well under the 30-day aggregation interval."""
    out = []
    dd = [(pd.Timestamp(o["sensing_date"]), o) for o in disp_obs if o.get("sensing_date")]
    for o in true_obs:
        if not o.get("sensing_date"):
            continue
        t = pd.Timestamp(o["sensing_date"])
        best, gap = None, max_gap_days + 1
        for td, cand in dd:
            g = abs((td - t).days)
            if g < gap:
                best, gap = cand, g
        if best is not None:
            out.append((o, best, gap))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=24)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--radius-km", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--pace", type=float, default=2.0)
    ap.add_argument("--sar", action="store_true", help="also run the Sentinel-1 radar arm")
    ap.add_argument("--out", default="experiments/E09-displacement/sentinel_arm.json")
    a = ap.parse_args()

    client = SentinelClient()
    if not client.enabled:
        raise SystemExit("SENTINEL_CLIENT_ID / _SECRET not configured")

    rng = np.random.default_rng(a.seed)
    df = pd.read_parquet("data/training_set.parquet")
    sites = df[["site_id", "latitude", "longitude", "cluster"]].drop_duplicates("site_id")
    per = max(1, a.sites // sites.cluster.nunique())
    sites = (sites.groupby("cluster", group_keys=False)
             .apply(lambda g: g.sample(min(len(g), per), random_state=a.seed)))
    print(f"[E09-S] {len(sites)} sites x {a.draws} draws, R={a.radius_km} km", flush=True)

    kinds = ["opt"] + (["sar"] if a.sar else [])
    rows, stats = [], {"requests": 0, "cache_hits": 0, "empty_true": 0, "empty_disp": 0}

    for s in sites.itertuples():
        for kind in kinds:
            t_obs, hit = fetch_history(client, s.latitude, s.longitude, DAYS, kind)
            stats["cache_hits" if hit else "requests"] += 1
            if not hit:
                time.sleep(a.pace)
            if not t_obs:
                stats["empty_true"] += 1
                print(f"[E09-S] {s.site_id} {kind}: TRUE history empty, skipped", flush=True)
                continue

            for k in range(a.draws):
                dlat, dlon = displace(s.latitude, s.longitude, a.radius_km, rng)
                d_obs, hit = fetch_history(client, dlat, dlon, DAYS, kind)
                stats["cache_hits" if hit else "requests"] += 1
                if not hit:
                    time.sleep(a.pace)
                if not d_obs:
                    stats["empty_disp"] += 1
                    print(f"[E09-S] {s.site_id} {kind} draw {k}: displaced history EMPTY "
                          f"(no imagery or request failed) -- recorded, not silently dropped",
                          flush=True)
                    continue
                for to, do, gap in match(t_obs, d_obs):
                    rows.append({"site_id": s.site_id, "cluster": s.cluster, "kind": kind,
                                 "draw": k, "sensing_date": to["sensing_date"], "gap_days": gap,
                                 **{f"t_{m}": to.get(m) for m in to if m != "sensing_date"},
                                 **{f"d_{m}": do.get(m) for m in do if m != "sensing_date"}})
        print(f"[E09-S] {s.site_id}: {len(rows)} paired obs, {stats['requests']} requests",
              flush=True)

    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out.replace(".json", ".parquet"))

    metrics = {}
    for col in [c[2:] for c in out.columns if c.startswith("t_")]:
        t = pd.to_numeric(out[f"t_{col}"], errors="coerce").to_numpy(float)
        d = pd.to_numeric(out[f"d_{col}"], errors="coerce").to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(d)
        if ok.sum() < 10 or np.std(t[ok]) == 0:
            metrics[col] = {"n": int(ok.sum()), "note": "degenerate"}
            continue
        sub = out.loc[ok, ["site_id"]].copy(); sub["t"], sub["d"] = t[ok], d[ok]
        wt = (sub["t"] - sub.groupby("site_id")["t"].transform("mean")).to_numpy()
        wd = (sub["d"] - sub.groupby("site_id")["d"].transform("mean")).to_numpy()
        metrics[col] = {
            "n": int(ok.sum()),
            "raw_r": round(float(np.corrcoef(t[ok], d[ok])[0, 1]), 4),
            "within_site_r": (round(float(np.corrcoef(wt, wd)[0, 1]), 4)
                              if np.std(wt) > 0 and np.std(wd) > 0 else None),
            "nrmse": round(float(np.sqrt(np.mean((t[ok] - d[ok]) ** 2)) / np.std(t[ok])), 4),
        }

    summary = {"seed": a.seed, "radius_km": a.radius_km, "draws": a.draws,
               "sites": int(out.site_id.nunique()) if len(out) else 0,
               "paired_obs": len(out), **stats,
               "aoi_km": [1.11, 1.10], "expected_aoi_overlap": 0.1054,
               "zero_overlap_share": 0.7508, "metrics": metrics}
    Path(a.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2), flush=True)
    for k, v in metrics.items():
        print(f"  {k:14s} {v}", flush=True)


if __name__ == "__main__":
    main()
