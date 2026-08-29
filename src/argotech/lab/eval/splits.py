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
    """Expanding-window temporal folds, purged and embargoed at the boundary.

    Test rows are cut on `obs_date`, the date a prediction would have been made: `obs_date >=
    boundary`. Training rows are cut on `label_date`, the date the outcome became known:
    `label_date < boundary`. Cutting *training* on `obs_date` instead — the incumbent's
    construction — is the leak docs/RESEARCH_SUMMARY.md:64 records as uncorrected: it lets the
    last month of training outcomes become known only after the first test prediction was made.

    A row with `obs_date < boundary <= label_date` satisfies neither cut and falls into an
    embargo gap between the two sides. That is deliberate, standard purged/embargoed
    forward-chaining, not a bug.
    """
    dates = np.sort(df["obs_date"].unique())
    if len(dates) <= n_folds:
        return
    for cut in np.array_split(dates, n_folds + 1)[1:]:
        boundary = cut[0]
        train, test = df[df["label_date"] < boundary], df[df["obs_date"] >= boundary]
        if len(train) and len(test):
            yield f"train < {boundary}", train, test
