"""The peer reference, fitted per fold instead of over the whole frame.

`features.agronomic.peer_stats` computes one reference and its own docstring says that reference must
be fit per fold during evaluation — but no caller in the lab ever did that; `lab/panel.py` bakes a
whole-frame `peer_stats` into a static column instead, which is a held-out row standardising itself.

This module is the fold-fitted caller: `fit_peer_stats` sees only training rows, `apply_peer_z` scores
any frame (train or test) against that fixed reference. The two peer_stats arithmetic primitives,
`peer_reference` and `peer_z`, are reused unchanged from `features.agronomic` — this module only
changes *what rows* build the reference and *which key* buckets them.

Two cohort keys are offered. `cluster_month` is `peer_bucket`'s key, vectorised — tight, but a
held-out cluster's own key never appears in a training fit, so it gets no reference at all.
`geo_month` drops cluster identity for a fixed geographic band, so an unseen region can still land in
a bucket other regions also populate.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from argotech.features.agronomic import PEER_RELATIVE, peer_reference, peer_z

PEER_COLUMNS: tuple = (("ndvi", "ndvi_z_peer"), ("rvi", "rvi_z_peer"))


def peer_key(df: pd.DataFrame, kind: str, lat_band: float = 20.0, elev_band: float = 1000.0) -> pd.Series:
    """The reference-bucket key for every row.

    `"cluster_month"` -> `"<cluster>|<MM>"`. `"geo_month"` -> `"<lat_band>_<elev_band>|<MM>"`, where
    the bands are `floor(value / band) * band` — fixed cut points, never a quantile fitted over the
    frame, which would put the leak back in through the binning itself.
    """
    month = df["obs_date"].astype(str).str.slice(5, 7)
    if kind == "cluster_month":
        return df["cluster"].astype(str) + "|" + month
    if kind == "geo_month":
        lat_bucket = np.floor(df["latitude"].astype(float) / lat_band) * lat_band
        elev_bucket = np.floor(df["elevation"].astype(float) / elev_band) * elev_band
        return lat_bucket.astype(str) + "_" + elev_bucket.astype(str) + "|" + month
    raise ValueError(f"unknown peer key kind: {kind!r}")


def fit_peer_stats(train: pd.DataFrame, kind: str, lat_band: float = 20.0,
                    elev_band: float = 1000.0) -> dict:
    """`{bucket: {column: [mu, sigma]}}`, computed from `train` rows only.

    Mirrors `peer_stats`'s output shape, keyed by `peer_key` instead of the hardcoded cluster|month.
    A column whose sigma is NaN or zero (fewer than two observations, or a constant bucket) is left
    out of its bucket's entry entirely, rather than stored and later zeroed by `peer_z`: `peer_reference`
    only refuses a fabricated reference when it finds *no* entry, so a degenerate one has to be kept
    from ever being written, not caught downstream.
    """
    keys = peer_key(train, kind, lat_band=lat_band, elev_band=elev_band)
    grouped = train.assign(_bucket=keys).groupby("_bucket")
    mean, sd = grouped[list(PEER_RELATIVE)].mean(), grouped[list(PEER_RELATIVE)].std()
    stats: dict = {}
    for bucket in mean.index:
        entry = {}
        for col in PEER_RELATIVE:
            mu, sigma = float(mean.at[bucket, col]), float(sd.at[bucket, col])
            if math.isfinite(mu) and math.isfinite(sigma) and sigma > 0:
                entry[col] = [mu, sigma]
        if entry:
            stats[str(bucket)] = entry
    return stats


def apply_peer_z(df: pd.DataFrame, stats: dict, kind: str, lat_band: float = 20.0,
                  elev_band: float = 1000.0) -> tuple[pd.DataFrame, float]:
    """Rewrite `ndvi_z_peer`/`rvi_z_peer` against `stats`, and report coverage across both.

    `stats` was fit elsewhere, on training rows only; this function looks a row's bucket up in it and
    never folds `df`'s own values back into the reference.

    The returned coverage is OR semantics across `ndvi` and `rvi`: a row counts as covered if either
    column found a bucket entry, not only if both did. `run.py` does not use this figure for the
    reported `peer_coverage` metric — it computes its own `ndvi_z_peer`-specific coverage instead,
    because `ndvi_z_peer` is the load-bearing feature and radar is additive-never-a-gate in this
    codebase, so a fold where only `rvi` found a reference should not read as covered.
    """
    out = df.copy()
    keys = peer_key(df, kind, lat_band=lat_band, elev_band=elev_band)
    for raw_col, z_col in PEER_COLUMNS:
        out[z_col] = [peer_z(value, peer_reference(stats, bucket, raw_col))
                      for value, bucket in zip(out[raw_col], keys, strict=True)]
    coverage = float(keys.isin(stats.keys()).mean()) if len(out) else 0.0
    return out, coverage
