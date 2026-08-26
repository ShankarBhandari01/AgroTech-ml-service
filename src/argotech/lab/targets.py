"""The estimand family.

`forward_z` standardises each observation against its cluster peers at the same date, which removes
the cohort-date effect. It never standardises against the field's own history, so the field effect
survives in the target — and `add_site_climatology`, which estimates exactly that effect, beats the
fitted model by supplying the half the target left in (docs/model-design.md, artifacts/metrics.json).

E01 measured the field effect at 34.5% of `forward_z` variance, CI [0.236, 0.435]. This module
removes it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_KINDS = ("level_z", "within_y", "within_xy", "delta_z")

WITHIN_SUFFIX = "_w"


def alpha_hat(df: pd.DataFrame, column: str = "ndvi_z_peer", unit: str = "site_id",
              min_history: int = 0, shrink: float = 0.0) -> pd.Series:
    """The field effect, estimated from a field's strictly earlier observations.

        n_j       = number of observations of this field before the j-th
        raw_j     = (1 / n_j) * SUM_{m < j} column_m
        alpha_j   = (n_j / (n_j + shrink)) * raw_j

    Shrinkage pulls toward **zero**, not toward a computed cluster mean. `forward_z` is already
    standardised within cluster and date, so the cluster mean is ~0 by construction, and E01
    measured the cluster ICC at exactly 0.0000. A cluster mean computed over the whole frame would
    also leak future observations into a quantity defined as prior-only.

    E01 also gives shrinkage a number to beat: the field effect is 34.5% of variance, but an
    unshrunk `alpha_hat` removes only 19.9% of it. The ~15-point gap is this estimator's own noise.

    Rows with fewer than `min_history` prior observations get NaN — they have no reference, and a
    default of 0.0 would assert "exactly average" about a field nothing is known of. `min_history=0`
    and `min_history=1` behave identically: a row with zero prior observations has weight 0 regardless,
    so without an implicit floor of 1 it would silently compute 0.0 * nan = 0.0 rather than NaN.
    """
    order = df.sort_values([unit, "obs_date"]).index
    d = df.loc[order]
    grouped = d.groupby(unit, sort=False)[column]

    prior_mean = grouped.transform(lambda s: s.expanding().mean().shift(1))
    prior_n = grouped.cumcount().astype(float)

    weight = np.where(prior_n > 0, prior_n / (prior_n + shrink), 0.0)
    out = pd.Series(weight * prior_mean.fillna(0.0).to_numpy(), index=d.index)
    out[prior_n < max(min_history, 1)] = np.nan

    return out.reindex(df.index)


def _within(df: pd.DataFrame, columns: list[str], unit: str, shrink: float = 0.0,
            train_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Each column minus its own strictly-prior field mean. The Frisch-Waugh-Lovell other half.

    A time-invariant regressor (e.g. `elevation`) demeans to exactly zero for every row with any
    history — correct FWL behaviour, since a field's own value has no variation to explain around
    its own prior mean. Handed to Ridge or the boosted arm as a live feature, an all-zero column is
    dead weight at best; excluded here rather than passed through, with a one-line note naming what
    was dropped so the exclusion is visible instead of silently changing the feature count.

    `df` may be a concatenated train+test frame (it must be, for a fold: see `alpha_hat` and
    `run.py`). `train_mask`, when given, is a boolean array the same length as `df` marking its
    train rows: degeneracy (`std < 1e-12`) is then decided from THOSE rows only, never from the
    union. Deciding it from the union would let a column that is constant in train but happens to
    vary in test be kept on the strength of test data alone — the same transduction this module
    exists to remove elsewhere. Without `train_mask`, degeneracy is decided from all of `df`.
    """
    out = {}
    dropped = []
    for c in columns:
        col = df[c] - alpha_hat(df, column=c, unit=unit, min_history=1, shrink=shrink)
        std_source = col.iloc[train_mask] if train_mask is not None else col
        if std_source.std(skipna=True) < 1e-12:
            dropped.append(c)
            continue
        out[c + WITHIN_SUFFIX] = col
    if dropped:
        print(f"[targets] within_xy: dropped time-invariant column(s) (demean to zero): "
              f"{', '.join(dropped)}")
    return pd.DataFrame(out, index=df.index)


def build_target(df: pd.DataFrame, kind: str, features: list[str], unit: str = "site_id",
                 min_history: int = 0, shrink: float = 0.0,
                 train_mask: np.ndarray | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Attach `alpha_hat`, `ztilde` and `persistence_pred`, drop rows with no reference, and name the
    arm's features.

    Returns (frame, feature_names). `within_xy` hands back demeaned feature names; every other kind
    hands back `features` unchanged. Residualising the target alone is *not* the within estimator —
    FWL requires demeaning both sides — so the two are separate kinds and E02 runs them head to head.

    `persistence_pred` is "carry the current reading forward", but what that means depends on what
    `ztilde` is: the field's raw level under `level_z`, its within-deviation under `within_y` /
    `within_xy`, and "no change" — 0.0 — under `delta_z`, whose target is already a difference. It
    lives here rather than in the arm because it is a property of the estimand, not of the model.

    For a fold's train/test pair, `df` must be the CONCATENATED train+test frame, called once —
    not train and test separately. `alpha_hat` is a strictly-prior expanding mean per site; called
    on the test slice alone, a site's pre-boundary rows become invisible to its own post-boundary
    `alpha_hat`, truncating real, legitimately-available history rather than fitting anything.
    `run.py` splits the result back apart after this call, by which side each row came from.

    `train_mask` (only meaningful under `within_xy`) is a boolean array the same length as `df`
    marking its train rows, so the demeaned-column degeneracy decision is made from train alone —
    see `_within`. It has no effect on `alpha_hat`, which correctly uses every row regardless.
    """
    if kind not in TARGET_KINDS:
        raise ValueError(f"unknown target kind {kind!r}; expected one of {TARGET_KINDS}")

    out = df.copy()
    out["alpha_hat"] = alpha_hat(out, unit=unit, min_history=min_history, shrink=shrink)

    if kind == "level_z":
        out["ztilde"] = out["forward_z"]
        out["persistence_pred"] = out["ndvi_z_peer"]
        names = list(features)
    elif kind == "within_y":
        out["ztilde"] = out["forward_z"] - out["alpha_hat"]
        out["persistence_pred"] = out["ndvi_z_peer"] - out["alpha_hat"]
        names = list(features)
    elif kind == "within_xy":
        within_df = _within(out, features, unit, shrink=shrink, train_mask=train_mask)
        out = pd.concat([out, within_df], axis=1)
        out["ztilde"] = out["forward_z"] - out["alpha_hat"]
        out["persistence_pred"] = out["ndvi_z_peer"] - out["alpha_hat"]
        names = list(within_df.columns)
    else:  # delta_z — the first difference; persistence predicts no change
        out["ztilde"] = out["forward_z"] - out["ndvi_z_peer"]
        out["persistence_pred"] = 0.0
        names = list(features)

    # alpha_hat stays in the required subset even for level_z, whose ztilde never touches it: this
    # keeps row counts identical across level_z / within_y / within_xy so E02 compares arms on the
    # same sample, not on samples of different sizes.
    required = ["ztilde"] + ([] if kind == "delta_z" else ["alpha_hat"])
    return out.dropna(subset=required).reset_index(drop=True), names
