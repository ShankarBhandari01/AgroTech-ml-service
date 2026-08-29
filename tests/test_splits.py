"""What the two protocols do, and the one thing E01 says a blocked score is not evidence of.

Cluster ICC of forward_z is exactly 0.0000, so leave-one-cluster-out blocks against feature
distribution shift and never against target structure. These pin the mechanics; the interpretation
lives in the spec's section 7.
"""

from __future__ import annotations

import pandas as pd

from argotech.lab.eval.splits import forward_chaining, leave_one_cluster_out


def _df() -> pd.DataFrame:
    rows = []
    for c in ("C0", "C1", "C2"):
        for s in range(4):
            for m in range(1, 13):
                obs = pd.Timestamp(f"2025-{m:02d}-01")
                rows.append({"cluster": c, "site_id": f"{c}-{s}",
                             "obs_date": obs, "label_date": obs + pd.Timedelta(days=30),
                             "forward_z": 0.1 * m})
    return pd.DataFrame(rows)


def test_every_cluster_is_held_out_exactly_once():
    folds = list(leave_one_cluster_out(_df()))
    assert [name for name, _, _ in folds] == ["C0", "C1", "C2"]


def test_no_site_appears_on_both_sides_of_a_spatial_fold():
    for name, train, test in leave_one_cluster_out(_df()):
        assert set(train.site_id) & set(test.site_id) == set(), f"{name} leaks a site"
        assert set(test.cluster) == {name}


def test_forward_chaining_never_trains_on_the_future():
    for name, train, test in forward_chaining(_df(), n_folds=3):
        assert train.obs_date.max() < test.obs_date.min(), f"{name} trains on the future"


def test_forward_chaining_training_labels_are_known_by_the_boundary():
    """The leak RESEARCH_SUMMARY:64 records as uncorrected: a training row whose outcome
    lands after the boundary was not knowable when the first test prediction was made.

    Cutting the training side on obs_date (the incumbent's construction, and what this
    module shipped in its first commit) puts such rows in train. Cutting on label_date
    does not. This test fails against the former.
    """
    df = _df()   # must now carry label_date = obs_date + 30 days
    for name, train, test in forward_chaining(df, n_folds=3):
        boundary = test.obs_date.min()
        assert (train.label_date < boundary).all(), (
            f"{name}: {(train.label_date >= boundary).sum()} training rows have outcomes "
            "that postdate the first test prediction")


def test_forward_chaining_training_sets_grow():
    sizes = [len(train) for _, train, _ in forward_chaining(_df(), n_folds=3)]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3


def test_a_fold_with_an_empty_side_is_not_yielded():
    single = _df()[lambda d: d.cluster == "C0"]
    assert list(leave_one_cluster_out(single)) == [], \
        "one cluster cannot be blocked against itself"
