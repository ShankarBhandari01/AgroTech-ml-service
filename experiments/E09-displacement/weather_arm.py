"""E09 weather arm — what a survey-style coordinate displacement costs the weather features.

Design (see docs/superpowers/specs/2026-08-29-dataset-joining-design.md 4a):

  Hold the satellite block fixed at the true site and vary ONLY the weather coordinate. The 18
  weather features in `features.agronomic.build` depend solely on `daily` and `clim_rain_30`, so
  this isolates the weather channel exactly and costs no Sentinel API calls.

  Displacement follows the DHS/LSMS rule the public releases use: uniform random direction,
  distance uniform in [0, R]. R = 5 km (rural). Labels stay bound to the TRUE site -- that is the
  point: the outcome is real, the coordinate is not.

Reuses the production fetch and feature code unchanged; nothing here reimplements a feature.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from argotech.data import meteo
from argotech.features import agronomic
from argotech.lab.panel import panel

# The archive window must start >= WINDOW_DAYS before the panel's first obs_date (2022-09-10) so the
# trailing 90-day slice is complete for every row -- 2022-05-14 leaves 119 days, comfortably clear.
#
# The exact dates are chosen to MATCH AN ALREADY-CACHED WINDOW, and that is the whole point:
# `panel.collect_site` derives its window from `date.today()`, so the meteo cache holds a different
# window per build day. An unaligned window turns all 122 true-site fetches into fresh API calls,
# which is what exhausted Open-Meteo's rate limit on the first full run and dropped 88 of 122 sites
# -- with the loss cluster-structured (Kano 6%, Kenya 10%) because sites are processed in name order.
# This window is cached for 122/122 sites, so the true arm costs nothing and the whole budget goes
# to the displaced cells, which are the only ones that actually need fetching.
START, END = "2022-05-14", "2026-08-15"

# Open-Meteo's served archive grid, measured by probing (not assumed): successive cell centres are
# 0.070299 deg apart in latitude and 0.080143 deg in longitude -- a ~7.8 x 8.8 km cell, i.e.
# ERA5-Land at ~9 km, NOT the 0.25 deg / 28 km grid the spec originally assumed.
DLAT, DLON = 0.070299, 0.080143
LAT0, LON0 = 11.353251, 7.934105


def cell_of(lat: float, lon: float) -> tuple[int, int]:
    return (int(round((lat - LAT0) / DLAT)), int(round((lon - LON0) / DLON)))


def displace(lat: float, lon: float, r_km: float, rng) -> tuple[float, float]:
    """One DHS-style offset: uniform direction, distance uniform in [0, r_km]."""
    d = rng.uniform(0.0, r_km)
    th = rng.uniform(0.0, 2 * math.pi)
    dlat = (d * math.sin(th)) / 111.32
    dlon = (d * math.cos(th)) / (111.32 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def trim(daily: dict, end_date: str) -> dict | None:
    """`daily` truncated to end on `end_date`, matching what agronomic._slice expects."""
    times = daily["time"]
    try:
        i = times.index(end_date)
    except ValueError:
        return None
    if i < agronomic.WINDOW_DAYS:
        return None
    return {k: v[: i + 1] for k, v in daily.items()}


def features_at(daily: dict, obs_date: str) -> dict | None:
    """The 18 weather features at one prediction date. `sat` is a constant stub: it feeds only the
    satellite block, which this arm does not compare, and is identical across both arms."""
    d = trim(daily, obs_date)
    if d is None:
        return None
    clim = panel._climatological_rain_30(d, len(d["time"]) - 1, obs_date[:4])
    stub = {"ndvi": 0.0, "ndmi": 0.0, "evi": 0.0, "vci": 0.0, "ndvi_z_peer": 0.0}
    row = agronomic.build(d, stub, {"latitude": 0.0, "longitude": 0.0,
                                    "elevation": 0.0, "clim_rain_30": clim})
    return {k: row[k] for k in WEATHER_FEATURES}


WEATHER_FEATURES = [
    "gdd_90", "gdd_since_onset", "tmax_mean_30", "tmin_mean_30", "diurnal_range_30",
    "heat_stress_days", "rain_30", "rain_90", "et0_90", "water_satisfaction_30",
    "water_deficit_30", "dry_spell_30", "dry_spell_90", "rain_anomaly_30",
    "rh_mean_30", "radiation_90", "days_since_onset", "stage_kc",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=30, help="sites sampled per run (0 = all 122)")
    ap.add_argument("--draws", type=int, default=4, help="displacement realisations per site")
    ap.add_argument("--radius-km", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--dates", type=int, default=8, help="obs_dates sampled per site")
    ap.add_argument("--out", default="experiments/E09-displacement/weather_arm.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    df = pd.read_parquet("data/training_set.parquet")
    sites = df[["site_id", "latitude", "longitude", "cluster"]].drop_duplicates("site_id")
    if a.sites:                       # stratify across clusters so one region cannot dominate
        sites = (sites.groupby("cluster", group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), max(1, a.sites // sites.cluster.nunique())),
                                           random_state=a.seed)))

    rows, n_cell_changed, n_pairs = [], 0, 0
    for s in sites.itertuples():
        true_payload = meteo.fetch_archive(s.latitude, s.longitude, START, END)
        if not true_payload:
            print(f"[E09] {s.site_id}: true fetch failed, skipped")
            continue
        true_daily = meteo.daily_frame(true_payload)
        if not meteo.has_all_variables(true_daily):
            print(f"[E09] {s.site_id}: incomplete true series, skipped")
            continue

        obs = sorted(df.loc[df.site_id == s.site_id, "obs_date"].astype(str).unique())
        if len(obs) > a.dates:
            obs = list(pd.Series(obs).sample(a.dates, random_state=a.seed).sort_values())

        for k in range(a.draws):
            dlat, dlon = displace(s.latitude, s.longitude, a.radius_km, rng)
            moved = cell_of(s.latitude, s.longitude) != cell_of(dlat, dlon)
            n_pairs += 1
            n_cell_changed += int(moved)

            disp_payload = meteo.fetch_archive(dlat, dlon, START, END)
            if not disp_payload:
                print(f"[E09] {s.site_id} draw {k}: displaced fetch failed, skipped")
                continue
            disp_daily = meteo.daily_frame(disp_payload)
            if not meteo.has_all_variables(disp_daily):
                continue

            for od in obs:
                ft, fd = features_at(true_daily, od), features_at(disp_daily, od)
                if ft is None or fd is None:
                    continue
                rows.append({"site_id": s.site_id, "cluster": s.cluster, "draw": k,
                             "obs_date": od, "cell_changed": moved,
                             **{f"t_{k2}": v for k2, v in ft.items()},
                             **{f"d_{k2}": v for k2, v in fd.items()}})
        print(f"[E09] {s.site_id}: {len(rows)} paired rows so far")

    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out.replace(".json", ".parquet"))

    summary = {"seed": a.seed, "radius_km": a.radius_km, "draws": a.draws,
               "sites": int(out.site_id.nunique()) if len(out) else 0,
               "paired_rows": len(out),
               "cell_change_rate": round(n_cell_changed / max(1, n_pairs), 4),
               "grid_cell_km": [round(DLAT * 111.32, 2), round(DLON * 111.32 * 0.98, 2)],
               "features": {}}
    for f in WEATHER_FEATURES:
        t, d = out[f"t_{f}"].to_numpy(float), out[f"d_{f}"].to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(d)
        t, d = t[ok], d[ok]
        if len(t) < 3 or np.std(t) == 0:
            summary["features"][f] = {"n": int(len(t)), "note": "degenerate"}
            continue
        # Raw correlation across sites and dates is dominated by season and geography, which a few
        # kilometres of displacement trivially preserves -- it would read ~0.99 even if every
        # decision-relevant deviation were destroyed. The model consumes within-site deviations
        # (peer z-scores, the within_* estimands), so the honest figure demeans by site first.
        sub = out.loc[ok, ["site_id"]].copy()
        sub["t"], sub["d"] = t, d
        wt = (sub["t"] - sub.groupby("site_id")["t"].transform("mean")).to_numpy()
        wd = (sub["d"] - sub.groupby("site_id")["d"].transform("mean")).to_numpy()
        within = (round(float(np.corrcoef(wt, wd)[0, 1]), 4)
                  if np.std(wt) > 0 and np.std(wd) > 0 else None)
        summary["features"][f] = {
            "n": int(len(t)),
            "pearson_r_raw": round(float(np.corrcoef(t, d)[0, 1]), 4),
            "pearson_r_within_site": within,
            "spearman_raw": round(float(pd.Series(t).corr(pd.Series(d), method="spearman")), 4),
            "rmse": round(float(np.sqrt(np.mean((t - d) ** 2))), 4),
            "sd_true": round(float(np.std(t)), 4),
            "sd_within_true": round(float(np.std(wt)), 4),
            "nrmse": round(float(np.sqrt(np.mean((t - d) ** 2)) / np.std(t)), 4),
        }
    Path(a.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:1500])


if __name__ == "__main__":
    main()
