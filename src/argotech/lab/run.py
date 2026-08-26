"""One entrypoint. An experiment is a config file, not a flag.

    python -m argotech.lab.run experiments/E02-within-vs-level.yaml

The incumbent's experiments were `--no-radar`, `--no-bands`, `--seed` and `--folds-only` on a
684-line script, and its results were compared across builds whose data differed because the builder
keys its window off date.today(). Every run here records the panel's content hash, the git SHA and
the seed, so a number in a document can be traced to the run that produced it — or cannot be cited.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from argotech.lab.arms import ARMS, resid_sd
from argotech.lab.evaluate import (
    DEFAULT_TAU,
    DEFAULT_THRESHOLDS,
    bootstrap_ci,
    decision_curve,
    net_benefit,
    precision_at_k,
    prob_event,
    spearman,
)
from argotech.lab.panel import manifest
from argotech.lab.splits import forward_chaining, leave_one_cluster_out

SPLITS = {"spatial": leave_one_cluster_out, "temporal": forward_chaining}


def load_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("seed", 42)
    cfg.setdefault("min_history", 3)
    cfg.setdefault("shrink", 0.0)
    cfg.setdefault("tau", DEFAULT_TAU)
    cfg.setdefault("splits", ["spatial", "temporal"])
    return cfg


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def _score_arm(arm, train, test, features, tau):
    """One arm on one fold: net benefit at the decision threshold, plus ranking diagnostics."""
    fitted = arm.fit(train, features)
    pred = fitted.predict(test, features)
    prob = prob_event(pred, resid_sd(fitted, train, features), tau)
    y = (test["ztilde"].to_numpy() <= tau).astype(int)
    curve = decision_curve(y, prob, DEFAULT_THRESHOLDS)
    return {
        # The headline: net benefit at the midpoint threshold. The whole curve travels with it.
        "net_benefit": net_benefit(y, prob, 0.20),
        "decision_curve": curve,
        # -pred is the risk score: a more negative predicted ztilde means a field falling
        # further behind its own norm, so it ranks higher for a visit.
        "precision_at_25": precision_at_k(y, -pred, 25),
        "spearman": spearman(pred, test["ztilde"].to_numpy()),
        "event_rate": float(np.mean(y)),
        "n": len(test),
    }


def run_experiment(cfg: dict, df: pd.DataFrame) -> dict:
    """Every arm, every fold, every declared split. Returns folds and a summary with intervals."""
    from argotech.lab.targets import build_target  # local: keeps the import graph acyclic

    unknown = [a for a in cfg["arms"] if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; expected from {sorted(ARMS)}")

    frame, features = build_target(df, cfg["target"], features=cfg["features"],
                                    min_history=cfg["min_history"], shrink=cfg["shrink"])

    folds = []
    for split_name in cfg["splits"]:
        for fold_name, train, test in SPLITS[split_name](frame):
            scored = {name: _score_arm(ARMS[name](cfg["seed"]), train, test,
                                        features, cfg["tau"])
                      for name in cfg["arms"]}
            folds.append({"split": split_name, "fold": fold_name, "arms": scored})

    summary = {}
    for name in cfg["arms"]:
        per_fold = {m: [f["arms"][name][m] for f in folds]
                    for m in ("net_benefit", "precision_at_25", "spearman")}
        summary[name] = {f"{m}_mean": float(np.nanmean(v)) for m, v in per_fold.items()}
        summary[name].update({f"{m}_ci": bootstrap_ci(v, seed=cfg["seed"])
                               for m, v in per_fold.items()})

    return {"config": cfg,
            "provenance": {"git_sha": _git_sha(), "seed": cfg["seed"],
                            **{k: v for k, v in manifest(df).items()}},
            "folds": folds, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one lab experiment from its config file.")
    ap.add_argument("config", help="path to an experiments/*.yaml")
    ap.add_argument("--data", default="data/training_set.parquet")
    ap.add_argument("--out", default=None, help="where to write results (default: alongside config)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    result = run_experiment(cfg, pd.read_parquet(args.data))

    out = Path(args.out or Path(args.config).with_suffix(".result.json"))
    out.write_text(json.dumps(result, indent=2, default=float))

    try:
        import mlflow
        mlflow.set_experiment(cfg.get("name", Path(args.config).stem))
        with mlflow.start_run():
            mlflow.log_params({**{k: str(v) for k, v in cfg.items()}, **result["provenance"]})
            for arm, s in result["summary"].items():
                mlflow.log_metrics({f"{arm}.{k}": v for k, v in s.items()
                                     if not k.endswith("_ci")})
            mlflow.log_artifact(str(out))
    except ImportError:  # pragma: no cover — mlflow lives in the `train` extra
        print("mlflow not installed; results written to disk only", file=sys.stderr)

    print(f"{out}\n")
    for arm, s in sorted(result["summary"].items(), key=lambda kv: -kv[1]["net_benefit_mean"]):
        lo, hi = s["net_benefit_ci"]
        print(f"  {arm:<14} NB {s['net_benefit_mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
              f"   P@25 {s['precision_at_25_mean']:.3f}"
              f"   rho {s['spearman_mean']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
