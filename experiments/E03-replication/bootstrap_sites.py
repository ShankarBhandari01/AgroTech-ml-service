"""Site-level bootstrap for E03 (Task E): replicate spatial/geo_month/within_xy.

Follows the same resampling approach as argotech.lab.variance.decompose: resample SITES with
replacement (never rows), relabel each draw "<site_id>#<i>" so alpha_hat/leave_one_cluster_out
treat every draw as an independent pseudo-site, then rerun the full lab pipeline
(run_experiment) on the resampled panel. Each replicate yields one net_benefit_mean per arm,
over the (still 4, since resampling is a flat draw across all 122 sites and no cluster is ever
plausibly emptied) leave-one-cluster-out folds. The distribution of those replicate-level means,
over n_boot replicates, gives a percentile CI on the headline number the same way
variance.decompose gives a percentile CI on the site ICC.

Writes incremental progress + final JSON to this scratchpad file's directory.
"""
from __future__ import annotations

import os

# Thread-cap BEFORE numpy/sklearn import: unthrottled BLAS/OpenMP thread pools thrash on this
# machine's core count and make each ~1-second model fit take 5-10x longer under contention
# (measured: 14.4s/replicate single-job -> 66-73s/replicate with default threading + a concurrent
# job). Single-threaded is faster here for these small (n~3-4k) per-fold fits.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from argotech.lab.run import load_config, run_experiment  # noqa: E402

REPO = Path("/Users/shankarbhandari/Desktop/LaliShank/Projects/FastApi-AI/argotech-ai")
OUT = Path(__file__).parent / "e03_bootstrap_result.json"
N_BOOT = 200
BOOT_SEED = 20260826  # distinct from the model's own seed=42 (cfg seed), used only for resampling


def bootstrap_panel(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sites = df["site_id"].unique()
    pick = rng.choice(sites, len(sites), replace=True)
    parts = [df[df["site_id"] == s].assign(site_id=f"{s}#{i}") for i, s in enumerate(pick)]
    return pd.concat(parts, ignore_index=True)


def main():
    cfg = load_config(str(REPO / "experiments/E02/E02-within_xy-geo_month.yaml"))
    cfg["splits"] = ["spatial"]
    cfg["arms"] = ["linear", "boosted"]
    df = pd.read_parquet(REPO / "data/training_set.parquet")

    rng = np.random.default_rng(BOOT_SEED)
    linear_vals, boosted_vals = [], []
    skipped_replicates = 0

    devnull = open("/dev/null", "w")
    real_stderr = sys.stderr

    t0 = time.time()
    for i in range(N_BOOT):
        boot_df = bootstrap_panel(df, rng)
        sys.stderr = devnull  # silence the per-fold [run]/[targets] chatter, 200x over
        try:
            r = run_experiment(cfg, boot_df)
        finally:
            sys.stderr = real_stderr
        s = r["summary"].get("spatial", {})
        if "linear" in s and "boosted" in s:
            linear_vals.append(s["linear"]["net_benefit_mean"])
            boosted_vals.append(s["boosted"]["net_benefit_mean"])
        else:
            skipped_replicates += 1
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[{i+1}/{N_BOOT}] elapsed={elapsed:.0f}s skipped={skipped_replicates}",
                  file=real_stderr, flush=True)

    def pct_ci(v):
        v = np.asarray(v, dtype=float)
        lo, hi = np.percentile(v, [2.5, 97.5])
        return float(lo), float(hi)

    result = {
        "n_boot_requested": N_BOOT,
        "n_boot_used": len(linear_vals),
        "skipped_replicates": skipped_replicates,
        "boot_seed": BOOT_SEED,
        "linear": {
            "values": linear_vals,
            "mean": float(np.mean(linear_vals)),
            "ci_2.5_97.5": pct_ci(linear_vals),
        },
        "boosted": {
            "values": boosted_vals,
            "mean": float(np.mean(boosted_vals)),
            "ci_2.5_97.5": pct_ci(boosted_vals),
        },
        "elapsed_seconds": time.time() - t0,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print("DONE", OUT, file=real_stderr)


if __name__ == "__main__":
    main()
