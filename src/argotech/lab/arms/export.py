"""Write the one artifact `models/registry.py` loads and `serving/pipeline.py` runs.

Every other module under `lab/` exists to remove a leak: `peers.py` fits the peer reference on
training rows only, `targets.py` derives `alpha_hat` from strictly-prior observations, `run.py`
evaluates on folds a model never saw. None of that applies here, and that is deliberate rather than
an oversight carried over from the retired `training/train.py`.

A shipped artifact is scored one row at a time, at inference, against a field with no future to
leak from. `peer_stats` in `features/agronomic.py` says so in its own docstring: fitting the peer
reference "over the full frame" is correct "for the shipped artifact", because "a live request is
one row with no future to leak from" — the same is true of `cluster_stats`, and of the arm fit
below. This module is the one place in `lab/` where a whole-frame fit is the honest choice, not the
leak the rest of the package exists to close.

Run: `python -m argotech.lab.arms.export experiments/export-production.yaml`
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml

from argotech.config import settings
from argotech.features.agronomic import (
    CLUSTER_RELATIVE,
    CZ_SUFFIX,
    FEATURE_COLUMNS,
    MODEL_FEATURES,
    add_cluster_relative,
    cluster_stats,
    peer_stats,
)
from argotech.lab.arms.arms import ARMS, Sklearn
from argotech.lab.panel.panel import CLUSTERS, manifest
from argotech.lab.run import git_dirty, git_sha

PRODUCTION_ARTIFACT = Path(settings.AGRONOMIC_MODEL_PATH)

# What `serving/pipeline.py` can actually feed the model: the raw row from `agronomic.build`, plus
# the `_cz` twins it derives from the artifact's own `cluster_stats` snapshot. A feature outside this
# set loads fine and then raises a `KeyError` on the first live prediction — the failure this repo
# has hit before, so it is a guard in `export_artifact`, not a comment.
SERVABLE_FEATURES = frozenset(FEATURE_COLUMNS) | {c + CZ_SUFFIX for c in CLUSTER_RELATIVE}


def load_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("arm", "boosted")
    cfg.setdefault("seed", 42)
    # MODEL_FEATURES is the production feature set's one source of truth (features/agronomic.py);
    # naming it again in every export config would let the two drift.
    cfg.setdefault("features", MODEL_FEATURES)
    return cfg


def export_artifact(panel: pd.DataFrame, cfg: dict, path) -> Path:
    """Fit `cfg["arm"]` on every row of `panel` and write the bundle to `path`.

    `panel` is the raw parquet frame (`lab.panel`'s output): one row per (site, date), carrying
    `forward_z` and the leaky whole-frame `ndvi_z_peer`/`rvi_z_peer` `lab.panel.build_samples`
    already baked in. Fitting on that baked-in reference is fine here for the same reason fitting
    peer_stats over the full frame is: this is the artifact, not an evaluation of it.

    Only an arm backed by a real scikit-learn estimator (`linear`, `boosted`) can be shipped — `zero`,
    `persistence` and `climatology` are evaluation baselines with nothing for `serving/pipeline.py`
    to call `.predict(row)` on.

    Returns `path`. Raises `ValueError` if `cfg["arm"]` is unknown, not exportable, or asks for a
    feature `serving/pipeline.py` cannot build.
    """
    arm_name = cfg.get("arm", "boosted")
    if arm_name not in ARMS:
        raise ValueError(f"unknown arm {arm_name!r}; expected one of {sorted(ARMS)}")
    features = list(cfg.get("features", MODEL_FEATURES))
    unbuildable = sorted(set(features) - SERVABLE_FEATURES)
    if unbuildable:
        raise ValueError(
            f"{len(unbuildable)} feature(s) not buildable by serving/pipeline.py: {unbuildable}. "
            "export.py only ever writes the production artifact — there is no ablation branch here "
            "the way the retired training/train.py had; run an ablation as a `lab.run` experiment "
            "instead of exporting one.")

    df = add_cluster_relative(panel, group="cluster")
    ztilde = df["forward_z"]
    ok = ztilde.notna()
    if not ok.all():
        print(f"[export] {(~ok).sum()} of {len(df)} rows have no forward_z; fitting on the rest")
    train = df[ok].assign(ztilde=ztilde[ok])

    arm = ARMS[arm_name](cfg.get("seed", 42))
    fitted = arm.fit(train, features)
    if not isinstance(fitted, Sklearn):
        raise ValueError(
            f"arm {arm_name!r} is not Sklearn-backed and has no fitted estimator to ship; "
            "expected one of ['linear', 'boosted']")
    model = fitted.estimator

    bundle = {
        "model": model,
        "version": f"agro-{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
        "feature_columns": features,
        # The reference distributions serving replays one row at a time — snapshotted over the WHOLE
        # panel `df`, not just the rows with a target, matching what `add_cluster_relative` itself
        # already computed the twins in `train` against.
        "cluster_stats": cluster_stats(df, group="cluster"),
        "peer_stats": peer_stats(df, group="cluster"),
        "cluster_bounds": {c["name"]: {"lat": list(c["lat"]), "lon": list(c["lon"])} for c in CLUSTERS},
        "target": "forward_z",
        "n_samples": len(train),
        "trained_on": f"{df['obs_date'].min()}..{df['obs_date'].max()}",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit the production arm on the whole panel and write "
                                             "the artifact registry.py loads.")
    ap.add_argument("config", help="path to an experiments/*.yaml (arm, features, seed)")
    ap.add_argument("--data", default="data/training_set.parquet")
    ap.add_argument("--out", default=str(PRODUCTION_ARTIFACT))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    panel = pd.read_parquet(args.data)
    out = export_artifact(panel, cfg, args.out)

    # Beside the artifact, not inside it: the bundle's shape is `registry.py`'s contract, and a
    # provenance key is not part of it. Same fields `run.py` records for a `.result.json` — the
    # manifest hash names the data, `git_sha`/`dirty` name the code, `seed` names the arm's own draw.
    provenance = {"arm": cfg["arm"], "seed": cfg["seed"], "artifact": str(out),
                  "git_sha": git_sha(), "dirty": git_dirty(), **manifest(panel)}
    prov_path = out.with_name(out.stem + ".provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2, default=str))
    print(f"wrote {out} and {prov_path}\n{json.dumps(provenance, indent=2, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
