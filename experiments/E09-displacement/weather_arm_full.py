"""E09 weather arm at full scale — all 122 sites, M draws, via cell-level deduplication.

The naive form (one archive fetch per displacement draw) is ~3,800 heavy calls and rate-limits
immediately. It is also wasteful: Open-Meteo's archive serves a ~7.83 x 8.74 km grid, and every
point inside one cell returns byte-identical data. Verified by probe: +0.02 deg and +0.05 deg of
longitude both return cell (11.353251, 8.014248) with the same tmax.

So: draw M displacements per site (free), map each to its grid cell, and fetch each DISTINCT cell
once. M becomes nearly free and the fetch count collapses to the number of reachable cells --
typically 4-6 per site rather than M.

The grid origin is NOT assumed. Each site's own response reports the cell centre it was served,
and neighbours are enumerated relative to that local anchor, so this stays correct in every cluster
regardless of whether the global grid origin matches the one measured at Kaduna.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from argotech.data import meteo
from argotech.features import agronomic

from weather_arm import START, END, WEATHER_FEATURES, features_at, displace

DLAT, DLON = 0.070299, 0.080143     # measured spacing, used only for RELATIVE neighbour offsets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--radius-km", type=float, default=5.0)
    ap.add_argument("--dates", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--pace", type=float, default=2.5, help="seconds between uncached fetches")
    ap.add_argument("--out", default="experiments/E09-displacement/weather_arm_full.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    df = pd.read_parquet("data/training_set.parquet")
    sites = df[["site_id", "latitude", "longitude", "cluster"]].drop_duplicates("site_id")
    print(f"[E09] {len(sites)} sites, M={a.draws} draws, R={a.radius_km} km")

    def fetch(lat, lon):
        """Archive fetch with pacing applied only when the disk cache misses."""
        key = f"{lat:.4f},{lon:.4f},{START},{END}"
        cached = meteo._cache_path(key).exists()
        p = meteo.fetch_archive(lat, lon, START, END)
        if not cached:
            time.sleep(a.pace)
        return p

    rows, stats = [], {"cells_fetched": 0, "cells_failed": 0, "draws_lost": 0,
                       "draws_total": 0, "draws_moved": 0,
                       "sites_ok": 0, "sites_failed": []}

    for s in sites.itertuples():
        true_payload = fetch(s.latitude, s.longitude)
        if not true_payload:
            stats["sites_failed"].append(f"{s.site_id}:true")
            continue
        anchor_lat = true_payload.get("latitude")
        anchor_lon = true_payload.get("longitude")
        true_daily = meteo.daily_frame(true_payload)
        if anchor_lat is None or not meteo.has_all_variables(true_daily):
            stats["sites_failed"].append(f"{s.site_id}:incomplete")
            continue

        # Draw M displacements (free) and bucket them by cell offset from this site's own anchor.
        buckets: dict[tuple[int, int], int] = {}
        for _ in range(a.draws):
            dlat, dlon = displace(s.latitude, s.longitude, a.radius_km, rng)
            di = int(round((dlat - anchor_lat) / DLAT))
            dj = int(round((dlon - anchor_lon) / DLON))
            buckets[(di, dj)] = buckets.get((di, dj), 0) + 1
        stats["draws_total"] += a.draws
        stats["draws_moved"] += sum(n for (di, dj), n in buckets.items() if (di, dj) != (0, 0))

        obs = sorted(df.loc[df.site_id == s.site_id, "obs_date"].astype(str).unique())
        if len(obs) > a.dates:
            obs = list(pd.Series(obs).sample(a.dates, random_state=a.seed).sort_values())
        true_feats = {od: features_at(true_daily, od) for od in obs}

        for (di, dj), weight in sorted(buckets.items()):
            if (di, dj) == (0, 0):
                disp_daily = true_daily          # same cell => identical data, no fetch
            else:
                p = fetch(anchor_lat + di * DLAT, anchor_lon + dj * DLON)
                # Count the OUTCOME, not the attempt, and say so when a cell is lost. Incrementing
                # before the check reported 315 "fetched" when only 151 reached the output, and the
                # silent `continue` dropped 52% of the moved draws without a single log line --
                # the same silent-degradation-under-rate-limit that cost E06 40 sites.
                if not p:
                    stats["cells_failed"] += 1
                    stats["draws_lost"] += weight
                    print(f"[E09] {s.site_id}: cell ({di},{dj}) fetch FAILED, {weight} draws lost",
                          flush=True)
                    continue
                disp_daily = meteo.daily_frame(p)
                if not meteo.has_all_variables(disp_daily):
                    stats["cells_failed"] += 1
                    stats["draws_lost"] += weight
                    print(f"[E09] {s.site_id}: cell ({di},{dj}) INCOMPLETE, {weight} draws lost",
                          flush=True)
                    continue
                stats["cells_fetched"] += 1
            for od in obs:
                ft = true_feats.get(od)
                fd = features_at(disp_daily, od)
                if ft is None or fd is None:
                    continue
                rows.append({"site_id": s.site_id, "cluster": s.cluster, "obs_date": od,
                             "di": di, "dj": dj, "weight": weight,
                             "cell_changed": (di, dj) != (0, 0),
                             **{f"t_{k}": v for k, v in ft.items()},
                             **{f"d_{k}": v for k, v in fd.items()}})
        stats["sites_ok"] += 1
        print(f"[E09] {s.site_id}: {len(buckets)} distinct cells, {len(rows)} rows, "
              f"{stats['cells_fetched']} fetched so far", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(a.out.replace(".json", ".parquet"))

    # Each row carries `weight` = how many of the M draws landed in that cell, so the summary is
    # weighted by the displacement law rather than by how many distinct cells happened to exist.
    summary = {"seed": a.seed, "draws_per_site": a.draws, "radius_km": a.radius_km,
               "sites_ok": stats["sites_ok"], "sites_failed": stats["sites_failed"],
               "cells_fetched": stats["cells_fetched"],
               "cells_failed": stats["cells_failed"],
               "draws_lost": stats["draws_lost"],
               "draws_retained_frac": round(1 - stats["draws_lost"] / max(1, stats["draws_total"]), 4),
               "paired_rows": len(out),
               "cell_change_rate": round(stats["draws_moved"] / max(1, stats["draws_total"]), 4),
               "features": {}}
    for f in WEATHER_FEATURES:
        t, d, w = out[f"t_{f}"].to_numpy(float), out[f"d_{f}"].to_numpy(float), out["weight"].to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(d)
        if ok.sum() < 3 or np.std(t[ok]) == 0:
            summary["features"][f] = {"n": int(ok.sum()), "note": "degenerate"}
            continue
        sub = out.loc[ok, ["site_id"]].copy()
        sub["t"], sub["d"] = t[ok], d[ok]
        wt = (sub["t"] - sub.groupby("site_id")["t"].transform("mean")).to_numpy()
        wd = (sub["d"] - sub.groupby("site_id")["d"].transform("mean")).to_numpy()
        ww = w[ok]
        # weighted within-site correlation
        mt, md = np.average(wt, weights=ww), np.average(wd, weights=ww)
        cov = np.average((wt - mt) * (wd - md), weights=ww)
        sw = math.sqrt(np.average((wt - mt) ** 2, weights=ww) * np.average((wd - md) ** 2, weights=ww))
        summary["features"][f] = {
            "n": int(ok.sum()),
            "within_site_r": round(float(cov / sw), 4) if sw > 0 else None,
            "within_site_r_moved_only": None,
            "nrmse": round(float(np.sqrt(np.average((t[ok] - d[ok]) ** 2, weights=ww)) / np.std(t[ok])), 4),
        }
        m = ok & out["cell_changed"].to_numpy()
        if m.sum() > 10:
            s2 = out.loc[m, ["site_id"]].copy()
            s2["t"], s2["d"] = t[m], d[m]
            a2 = (s2["t"] - s2.groupby("site_id")["t"].transform("mean")).to_numpy()
            b2 = (s2["d"] - s2.groupby("site_id")["d"].transform("mean")).to_numpy()
            if np.std(a2) > 0 and np.std(b2) > 0:
                summary["features"][f]["within_site_r_moved_only"] = round(float(np.corrcoef(a2, b2)[0, 1]), 4)

    Path(a.out).write_text(json.dumps(summary, indent=2))
    print(f"[E09] done: {stats['sites_ok']} sites, {stats['cells_fetched']} cells fetched, "
          f"{len(out)} rows -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
