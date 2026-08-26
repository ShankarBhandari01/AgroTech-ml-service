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


def _within(df: pd.DataFrame, columns: list[str], unit: str) -> pd.DataFrame:
    """Each column minus its own strictly-prior field mean. The Frisch-Waugh-Lovell other half."""
    out = {}
    for c in columns:
        out[c + WITHIN_SUFFIX] = df[c] - alpha_hat(df, column=c, unit=unit, min_history=1)
    return pd.DataFrame(out, index=df.index)


def build_target(df: pd.DataFrame, kind: str, features: list[str], unit: str = "site_id",
                 min_history: int = 0, shrink: float = 0.0) -> tuple[pd.DataFrame, list[str]]:
    """Attach `alpha_hat` and `ztilde`, drop rows with no reference, and name the arm's features.

    Returns (frame, feature_names). `within_xy` hands back demeaned feature names; every other kind
    hands back `features` unchanged. Residualising the target alone is *not* the within estimator —
    FWL requires demeaning both sides — so the two are separate kinds and E02 runs them head to head.
    """
    if kind not in TARGET_KINDS:
        raise ValueError(f"unknown target kind {kind!r}; expected one of {TARGET_KINDS}")

    out = df.copy()
    out["alpha_hat"] = alpha_hat(out, unit=unit, min_history=min_history, shrink=shrink)

    if kind == "level_z":
        out["ztilde"] = out["forward_z"]
        names = list(features)
    elif kind == "within_y":
        out["ztilde"] = out["forward_z"] - out["alpha_hat"]
        names = list(features)
    elif kind == "within_xy":
        out = pd.concat([out, _within(out, features, unit)], axis=1)
        out["ztilde"] = out["forward_z"] - out["alpha_hat"]
        names = [c + WITHIN_SUFFIX for c in features]
    else:  # delta_z — the first difference; persistence collapses to the zero predictor
        out["ztilde"] = out["forward_z"] - out["ndvi_z_peer"]
        names = list(features)

    # alpha_hat stays in the required subset even for level_z, whose ztilde never touches it: this
    # keeps row counts identical across level_z / within_y / within_xy so E02 compares arms on the
    # same sample, not on samples of different sizes.
    required = ["ztilde"] + ([] if kind == "delta_z" else ["alpha_hat"])
    return out.dropna(subset=required).reset_index(drop=True), names
