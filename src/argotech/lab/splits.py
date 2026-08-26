"""The two validation protocols, separated from anything that fits a model.

Leave-one-cluster-out answers "does this work in a district we have never seen"; forward chaining
answers "does it work next month". Both are retained from docs/model-design.md section 6.

One measured caveat belongs with the first, from E01: the cluster ICC of `forward_z` is exactly
0.0000, because `z` is standardised within cluster and date. Blocking by cluster therefore controls
feature distribution shift and nothing about the target's structure. That narrows what a blocked
score is evidence of; it does not make the protocol wrong.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

Fold = tuple[str, pd.DataFrame, pd.DataFrame]


def leave_one_cluster_out(df: pd.DataFrame) -> Iterator[Fold]:
    """One fold per cluster, that cluster held out entirely."""
    for cluster in sorted(df["cluster"].dropna().unique()):
        test = df[df["cluster"] == cluster]
        train = df[df["cluster"] != cluster]
        if len(train) and len(test):
            yield str(cluster), train, test


def forward_chaining(df: pd.DataFrame, n_folds: int = 3) -> Iterator[Fold]:
    """Expanding-window temporal folds, cut on the observation date.

    Cut on `obs_date`, the date a prediction would have been made, rather than on `label_date`.
    Cutting on the label date would let the last month of training outcomes become known after the
    first test prediction was made — the defect docs/RESEARCH_SUMMARY.md section 5 records as
    uncorrected in the incumbent.
    """
    dates = np.sort(df["obs_date"].unique())
    if len(dates) <= n_folds:
        return
    for cut in np.array_split(dates, n_folds + 1)[1:]:
        boundary = cut[0]
        train, test = df[df["obs_date"] < boundary], df[df["obs_date"] >= boundary]
        if len(train) and len(test):
            yield f"train < {boundary}", train, test
