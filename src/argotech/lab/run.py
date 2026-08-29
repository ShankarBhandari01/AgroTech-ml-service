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

from argotech.lab.arms.arms import ARMS, resid_sd
from argotech.lab.estimand.peers import apply_peer_z, fit_peer_stats
from argotech.lab.eval.evaluate import (
    DEFAULT_TAU,
    DEFAULT_THRESHOLDS,
    bootstrap_ci,
    decision_curve,
    net_benefit,
    precision_at_k,
    prob_event,
    spearman,
)
from argotech.lab.eval.splits import forward_chaining, leave_one_cluster_out
from argotech.lab.panel.panel import manifest

SPLITS = {"spatial": leave_one_cluster_out, "temporal": forward_chaining}


def load_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("seed", 42)
    cfg.setdefault("min_history", 3)
    cfg.setdefault("shrink", 0.0)
    cfg.setdefault("tau", DEFAULT_TAU)
    cfg.setdefault("splits", ["spatial", "temporal"])
    # "leaky" reproduces the pre-fix behaviour (panel's whole-frame baked column) as the control
    # arm; "geo_month" is the default because it is the key that lets a held-out region borrow a
    # reference at all (lab/estimand/peers.py) — cluster_month gives a held-out cluster none.
    cfg.setdefault("peer_key", "geo_month")
    cfg.setdefault("lat_band", 20.0)
    cfg.setdefault("elev_band", 1000.0)
    return cfg


def git_sha() -> str:
    """Shared with `lab.arms.export`, which writes provenance the same way — no cycle to avoid here,
    unlike `lab.panel.panel`'s own copy (it is imported *by* this module, so importing back would cycle)."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def git_dirty() -> bool:
    """Whether the working tree differs from the recorded SHA.

    Without this, `git_sha` names a commit that may not be the code that ran, and a number becomes
    *falsely* traceable — worse than untraceable, because it can be cited with confidence. Flagged
    rather than blocked: a research harness must be runnable on uncommitted work, it just must not
    claim that work was committed.
    """
    try:
        return subprocess.run(["git", "diff", "--quiet", "HEAD"],
                               capture_output=True).returncode != 0
    except (OSError, subprocess.SubprocessError):   # pragma: no cover
        return True


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


def _score_fold(cfg: dict, panel: pd.DataFrame, train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    """Peer-transform a fold, then build its target on the WHOLE PANEL — not just train+test.

    `forward_chaining` (splits.py) deliberately embargoes rows with `obs_date < boundary <=
    label_date`: their label isn't knowable at prediction time, so they belong in neither train nor
    test. But `alpha_hat` is a strictly-prior expanding mean of `ndvi_z_peer`, a FEATURE — an
    embargoed row's feature exists at prediction time even though its label doesn't, so it is a
    legitimate prior observation for a later row of the same site and must still feed `alpha_hat`.
    Restricting to the train+test union (as opposed to `panel`) would silently drop it again.

    `panel` is peer-transformed as a whole using stats fit on `train_raw` alone (fitting on rows
    outside `train_raw`, even embargoed ones, would be the leak this module exists to remove); then
    `build_target` runs once over all of `panel`, and the fold's train/test rows are picked back out
    by index. `alpha_hat` never looks at a row's own `ndvi_z_peer`, only strictly earlier ones,
    whichever side of the boundary — or the embargo gap — they sit on.

    `within_xy`'s degenerate-column decision (`std < 1e-12`) is still made from the train rows only
    (`train_mask`), never the whole panel — deciding it from non-training data, even embargoed rows
    folded in for `alpha_hat`'s sake, would be the same transduction `fit_peer_stats` refuses by only
    ever seeing training rows.

    Returns (train_t, test_t, features, peer_coverage).
    """
    from argotech.lab.estimand.targets import build_target  # local: keeps the import graph acyclic

    peer_key_kind = cfg["peer_key"]
    if peer_key_kind == "leaky":
        # The control arm: today's whole-frame column, untouched. Coverage is trivially 1.0 —
        # every row already carries the (leaky) reference panel.py baked in.
        applied, peer_coverage = panel, 1.0
    else:
        bands = {"lat_band": cfg["lat_band"], "elev_band": cfg["elev_band"]}
        stats = fit_peer_stats(train_raw, peer_key_kind, **bands)
        applied, _ = apply_peer_z(panel, stats, peer_key_kind, **bands)
        # peer_coverage: the share of TEST rows for which `ndvi_z_peer` specifically got a
        # reference. Defined off this column rather than apply_peer_z's own returned fraction,
        # which counts a row covered if EITHER ndvi or rvi found a bucket entry — ndvi_z_peer is
        # the load-bearing feature (+0.0642 permutation importance vs. +0.0051 for the next one)
        # and radar is additive-never-a-gate in this codebase, so a combined figure would
        # misreport the metric that actually matters.
        test_z = applied.loc[applied.index.isin(test_raw.index), "ndvi_z_peer"]
        peer_coverage = float(test_z.notna().mean()) if len(test_z) else 0.0

    is_train, is_test = "_fold_is_train", "_fold_is_test"
    marked = applied.assign(**{is_train: applied.index.isin(train_raw.index),
                                is_test: applied.index.isin(test_raw.index)})
    train_mask = marked[is_train].to_numpy()
    combined_t, features = build_target(marked, cfg["target"], features=cfg["features"],
                                         min_history=cfg["min_history"], shrink=cfg["shrink"],
                                         train_mask=train_mask)
    train_t = (combined_t[combined_t[is_train]].drop(columns=[is_train, is_test])
               .reset_index(drop=True))
    test_t = (combined_t[combined_t[is_test]].drop(columns=[is_train, is_test])
              .reset_index(drop=True))
    return train_t, test_t, features, peer_coverage


def run_experiment(cfg: dict, df: pd.DataFrame) -> dict:
    """Every arm, every fold, every declared split. Returns folds and a summary with intervals."""
    from argotech.lab.estimand.targets import build_target  # local: keeps the import graph acyclic

    unknown = [a for a in cfg["arms"] if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; expected from {sorted(ARMS)}")

    peer_key_kind = cfg["peer_key"]

    # Changing peer_key changes ndvi_z_peer, and within_y/delta_z define their target IN TERMS OF
    # ndvi_z_peer (alpha_hat is derived from it; delta_z subtracts it directly) — so for those two
    # kinds, a different peer_key is a different target, not just different features. Only
    # level_z's target (forward_z, from the label's own cluster/date cohort) is peer-key-independent
    # and safe to compare across keys. See spec §7a.
    if cfg["target"] != "level_z" and peer_key_kind != "leaky":
        print(f"[run] warning: target={cfg['target']!r} is defined in terms of ndvi_z_peer, so "
              f"peer_key={peer_key_kind!r} changes the TARGET, not just the features — its numbers "
              "are not comparable across peer_key values except under target=level_z "
              "(docs/superpowers/specs/2026-08-26-two-way-demeaned-estimand-design.md §7a)",
              file=sys.stderr)

    # `ndvi_z_peer` (and therefore `alpha_hat`, which is derived from it) must be fit on training
    # rows alone, so both the peer transform AND build_target move inside the fold loop: splitting
    # now happens on the raw panel, before either has been computed. `build_target` runs over the
    # WHOLE panel `df`, not just this fold's train/test union, so a forward_chaining embargo row
    # (obs_date < boundary <= label_date: no knowable label, but its feature exists) still feeds a
    # later row's `alpha_hat` — see `_score_fold`.
    folds = []
    for split_name in cfg["splits"]:
        for fold_name, train_raw, test_raw in SPLITS[split_name](df):
            train_t, test_t, features, peer_coverage = _score_fold(cfg, df, train_raw, test_raw)

            if train_t.empty or test_t.empty:
                # A real finding under an honest cluster-keyed reference (delta_z / within_y are
                # defined in terms of ndvi_z_peer, so no reference means no target), not a defect
                # to paper over — skip the fold rather than handing sklearn zero rows.
                which = "train" if train_t.empty else "test"
                msg = (f"[run] skipping fold {fold_name!r} ({split_name}): target={cfg['target']!r} "
                       f"peer_key={peer_key_kind!r} produced an empty {which} frame "
                       f"(peer_coverage={peer_coverage:.3f})")
                print(msg, file=sys.stderr)
                folds.append({"split": split_name, "fold": fold_name,
                              "peer_coverage": peer_coverage, "skipped": msg, "arms": {}})
                continue

            scored = {name: _score_arm(ARMS[name](cfg["seed"]), train_t, test_t,
                                        features, cfg["tau"])
                      for name in cfg["arms"]}
            folds.append({"split": split_name, "fold": fold_name,
                          "peer_coverage": peer_coverage, "arms": scored})

    # Keyed by split protocol first: a blocked (spatial) fold and a forward-chaining (temporal)
    # fold are evidence of different things (splits.py), so pooling their means into one number —
    # as a flat fold list would — silently averages two validation protocols together.
    summary = {}
    for split_name in cfg["splits"]:
        split_folds = [f for f in folds if f["split"] == split_name]
        scored_folds = [f for f in split_folds if f["arms"]]
        skipped_folds = [f for f in split_folds if not f["arms"]]
        summary[split_name] = {}
        if not scored_folds:
            # Every fold in this split was skipped — an empty summary would look like a real (if
            # unlucky) result rather than "nothing to report"; better to report nothing at all.
            continue
        for name in cfg["arms"]:
            per_fold = {m: [f["arms"][name][m] for f in scored_folds]
                        for m in ("net_benefit", "precision_at_25", "spearman")}
            s = {f"{m}_mean": float(np.nanmean(v)) for m, v in per_fold.items()}
            s.update({f"{m}_ci": bootstrap_ci(v, seed=cfg["seed"]) for m, v in per_fold.items()})
            # Same value for every arm in a split — coverage is a property of the fold's peer
            # transform, not of the arm — but recorded per-arm so `summary`'s {split: {arm: {...}}}
            # shape (and every consumer of it) stays exactly as it was. Averaged over every fold
            # in the split, skipped ones included: their (typically 0.0) coverage is real signal.
            s["peer_coverage_mean"] = float(np.nanmean([f["peer_coverage"] for f in split_folds]))
            # A partially-skipped split must not average its survivors and print as if nothing were
            # lost: `folds_scored`/`folds_attempted` travel with every arm's numbers (same
            # per-arm-duplication as peer_coverage_mean above), and `skipped_folds` names which ones
            # were dropped and why, so a 2-of-4-cluster result can never be read as a 4-cluster one.
            s["folds_scored"] = len(scored_folds)
            s["folds_attempted"] = len(split_folds)
            s["skipped_folds"] = [{"fold": f["fold"], "reason": f["skipped"]} for f in skipped_folds]
            summary[split_name][name] = s

    # Descriptive only, not used for scoring: how many rows in the raw panel have enough field
    # history to ever carry a target at all. The fold loop above builds its own train/test target
    # per split (ndvi_z_peer, and therefore alpha_hat, must be fold-fit), so there is no longer one
    # whole-panel `frame` to report a single N from; this reproduces that count for provenance,
    # the way `manifest` describes the panel rather than what any one fold scored.
    scored_rows = len(build_target(df, cfg["target"], features=cfg["features"],
                                    min_history=cfg["min_history"], shrink=cfg["shrink"])[0])

    return {"config": cfg,
            "provenance": {"git_sha": git_sha(), "dirty": git_dirty(), "seed": cfg["seed"],
                            # `manifest` describes the INPUT panel — the lineage anchor.
                            "scored_rows": scored_rows,
                            **manifest(df)},
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

    # Every fold produced an empty train or test frame (e.g. delta_z under cluster_month, where a
    # held-out cluster has no reference at all) — `summary` is empty per split, not a real result.
    # Say so plainly and fail the run rather than reporting nothing as if it were something.
    if result["folds"] and all(f.get("skipped") for f in result["folds"]):
        print(f"{out}\n", file=sys.stderr)
        print(f"error: every fold was skipped (0/{len(result['folds'])} scored) — "
              "no result to report", file=sys.stderr)
        return 1

    try:
        import mlflow
        mlflow.set_experiment(cfg.get("name", Path(args.config).stem))
        with mlflow.start_run():
            mlflow.log_params({**{k: str(v) for k, v in cfg.items()}, **result["provenance"]})
            for split_name, split_summary in result["summary"].items():
                for arm, s in split_summary.items():
                    mlflow.log_metrics({f"{split_name}.{arm}.{k}": v for k, v in s.items()
                                         if not k.endswith("_ci") and k != "skipped_folds"})
            mlflow.log_artifact(str(out))
    except ImportError:  # pragma: no cover — mlflow lives in the `train` extra
        print("mlflow not installed; results written to disk only", file=sys.stderr)

    print(f"{out}\n")
    for split_name, split_summary in result["summary"].items():
        split_folds = [f for f in result["folds"] if f["split"] == split_name]
        skipped = [f for f in split_folds if f.get("skipped")]
        if not split_summary:
            # Every fold in this split was skipped: say so instead of printing a bare, empty header
            # that a reader could mistake for "this split just wasn't run".
            print(f"[{split_name}]  0/{len(split_folds)} folds scored — every fold was skipped:")
            for f in skipped:
                print(f"    {f['fold']}: {f['skipped']}")
            print()
            continue
        print(f"[{split_name}]")
        for arm, s in sorted(split_summary.items(), key=lambda kv: -kv[1]["net_benefit_mean"]):
            lo, hi = s["net_benefit_ci"]
            print(f"  {arm:<14} NB {s['net_benefit_mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  f"   P@25 {s['precision_at_25_mean']:.3f}"
                  f"   rho {s['spearman_mean']:+.3f}"
                  f"   peer_coverage {s['peer_coverage_mean']:.3f}"
                  f"   folds {s['folds_scored']}/{s['folds_attempted']}")
        if skipped:
            # A PARTIALLY skipped split: the averaged numbers above are real, but they are an
            # average over the survivors, not over every fold the protocol attempted — say which
            # ones were dropped and why, right next to the numbers a reader would otherwise cite
            # as e.g. "4-cluster leave-one-cluster-out" when only 2 clusters actually scored.
            print(f"  ({len(skipped)}/{len(split_folds)} folds skipped:")
            for f in skipped:
                print(f"    {f['fold']}: {f['skipped']}")
            print("  )")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
