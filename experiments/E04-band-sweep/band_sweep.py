"""Task F driver: the transfer-vs-specificity band sweep.

Sweeps lat_band x elev_band over {5,10,20,90} x {500,1000,2000} = 12 cells, on level_z (the only
key-comparable target) and within_xy (where the E02 positive result lives), spatial folds only
(leave-one-cluster-out). For every cell writes a config yaml + .result.json into
experiments/E04-band-sweep/ (same provenance-carrying run_experiment() this whole investigation
uses), and separately computes "donor_rows" per held-out cluster the same way
experiments/E02/README.md's donor-row anchors were validated (peer_key bucket membership, no
model fit) — this is the coverage-mechanism number, `peer_coverage` (the fraction of a held-out
cluster's rows that found ANY reference at all) is the run's own per-fold field.

Writes a summary CSV (band_sweep_summary.csv, alongside this file) for building the README table.

Run from the repo root: `python experiments/E04-band-sweep/band_sweep.py`
"""
from __future__ import annotations

import os

# See E03-replication/bootstrap_sites.py: single-threaded BLAS/OpenMP is faster than default
# multi-threaded here for these small per-fold fits, and avoids thread-pool thrashing when jobs
# run concurrently.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, "src")
from argotech.lab.peers import peer_key  # noqa: E402
from argotech.lab.run import load_config, run_experiment  # noqa: E402
from argotech.lab.splits import leave_one_cluster_out  # noqa: E402

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT_DIR = HERE
BASE_CFG = REPO / "experiments/E02-within-vs-level.yaml"
SUMMARY_CSV = HERE / "band_sweep_summary.csv"

LAT_BANDS = [5.0, 10.0, 20.0, 90.0]
ELEV_BANDS = [500.0, 1000.0, 2000.0]
TARGETS = ["level_z", "within_xy"]
CLUSTER_ORDER = ["Benue_River_Basin", "Kaduna_Grain_Belt", "Kano_Sudan_Savannah", "Kenya_Rift_Valley"]


def donor_rows(df: pd.DataFrame, lat_band: float, elev_band: float) -> dict:
    out = {}
    for cluster, train, test in leave_one_cluster_out(df):
        train_keys = peer_key(train, "geo_month", lat_band=lat_band, elev_band=elev_band)
        test_keys = peer_key(test, "geo_month", lat_band=lat_band, elev_band=elev_band)
        out[cluster] = int(train_keys.isin(set(test_keys.unique())).sum())
    return out


def main():
    df = pd.read_parquet(REPO / "data/training_set.parquet")
    real_stderr = sys.stderr
    devnull = open("/dev/null", "w")

    rows = []
    t0 = time.time()
    n = len(LAT_BANDS) * len(ELEV_BANDS) * len(TARGETS)
    i = 0
    for lat_band in LAT_BANDS:
        for elev_band in ELEV_BANDS:
            donors = donor_rows(df, lat_band, elev_band)
            for target in TARGETS:
                i += 1
                name = f"E04-{target}-lat{int(lat_band)}-elev{int(elev_band)}"
                cfg = load_config(str(BASE_CFG))
                cfg["name"] = name
                cfg["target"] = target
                cfg["peer_key"] = "geo_month"
                cfg["lat_band"] = lat_band
                cfg["elev_band"] = elev_band
                cfg["splits"] = ["spatial"]

                cfg_path = OUT_DIR / f"{name}.yaml"
                cfg_out = dict(cfg)
                cfg_path.write_text(yaml.safe_dump(cfg_out, sort_keys=False))

                sys.stderr = devnull
                try:
                    result = run_experiment(cfg, df)
                finally:
                    sys.stderr = real_stderr

                result_path = OUT_DIR / f"{name}.result.json"
                result_path.write_text(json.dumps(result, indent=2, default=float))

                s = result["summary"].get("spatial", {})
                per_cluster_cov = {f["fold"]: f["peer_coverage"] for f in result["folds"]
                                    if f["split"] == "spatial"}
                for arm in ("zero", "persistence", "climatology", "linear", "boosted"):
                    a = s.get(arm)
                    rows.append({
                        "lat_band": lat_band, "elev_band": elev_band, "target": target, "arm": arm,
                        "net_benefit_mean": a["net_benefit_mean"] if a else None,
                        "ci_lo": a["net_benefit_ci"][0] if a else None,
                        "ci_hi": a["net_benefit_ci"][1] if a else None,
                        "folds_scored": a["folds_scored"] if a else 0,
                        "folds_attempted": 4,
                        **{f"donor_{c}": donors.get(c, 0) for c in CLUSTER_ORDER},
                        **{f"cov_{c}": per_cluster_cov.get(c) for c in CLUSTER_ORDER},
                    })
                print(f"[{i}/{n}] {name} done ({time.time()-t0:.0f}s elapsed)",
                      file=real_stderr, flush=True)

    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("DONE", SUMMARY_CSV, file=real_stderr)


if __name__ == "__main__":
    main()
