"""Serving must reproduce training's `_cz` twins bit for bit.

Run: python tests/test_cluster_relative.py (or pytest)

`add_cluster_relative` (vectorised, training) and `cluster_relative_row` (scalar, serving) are two
implementations of one formula, which is exactly the shape of the train/serve skew this repo keeps
getting bitten by. They are kept separate on purpose — training standardises 6700 rows at once and
serving standardises one — so the agreement has to be asserted rather than assumed.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argotech.features.agronomic import (  # noqa: E402
    CLUSTER_RELATIVE,
    CZ_SUFFIX,
    add_cluster_relative,
    assign_cluster,
    cluster_relative_row,
    cluster_stats,
)

BOUNDS = {
    "Kaduna_Grain_Belt": {"lat": [10.2, 11.5], "lon": [7.3, 8.5]},
    "Kenya_Rift_Valley": {"lat": [-0.2, 1.2], "lon": [34.8, 36.1]},
}


def _frame(seed: int = 0) -> pd.DataFrame:
    """Two clusters x 40 rows, with the three cases the formula splits on planted in it."""
    rng = np.random.default_rng(seed)
    rows = []
    for cluster in ("Kaduna_Grain_Belt", "Kenya_Rift_Valley"):
        for i in range(40):
            row = {c: float(rng.normal(10, 3)) for c in CLUSTER_RELATIVE}
            row["cluster"] = cluster
            # A cloud-obscured field: the raw value is genuinely missing.
            if i == 0:
                row["ndvi"] = float("nan")
            rows.append(row)
    df = pd.DataFrame(rows)
    # A column that is constant within every cluster: sigma == 0, the 0.0 branch.
    df["rh_mean_30"] = 55.0
    return df


def test_scalar_matches_vectorised():
    df = _frame()
    trained = add_cluster_relative(df)
    stats = cluster_stats(df)

    checked = 0
    for _, ref in trained.iterrows():
        served = cluster_relative_row(ref.to_dict(), stats[ref["cluster"]])
        for col in CLUSTER_RELATIVE:
            want, got = ref[col + CZ_SUFFIX], served[col + CZ_SUFFIX]
            if math.isnan(want):
                assert math.isnan(got), f"{col}: training NaN, serving {got}"
            else:
                assert abs(want - got) < 1e-9, f"{col}: training {want}, serving {got}"
            checked += 1
    assert checked == len(df) * len(CLUSTER_RELATIVE)
    print(f"ok: {checked} twins agree between the training and serving implementations")


def test_the_three_way_split_is_actually_exercised():
    """Guards the test above from passing vacuously on an all-ordinary frame."""
    df = _frame()
    stats = cluster_stats(df)
    twins = [cluster_relative_row(r, stats[r["cluster"]]) for r in df.to_dict("records")]

    assert any(math.isnan(t["ndvi" + CZ_SUFFIX]) for t in twins), "no missing-value case"
    assert all(t["rh_mean_30" + CZ_SUFFIX] == 0.0 for t in twins), "constant column should be 0.0"
    assert any(abs(t["rain_30" + CZ_SUFFIX]) > 0.5 for t in twins), "no ordinary z-score case"
    print("ok: missing, constant and ordinary branches all covered")


def test_unknown_cluster_yields_nan_not_zero():
    """A field outside every sampled zone is unmeasured, not average.

    0.0 would tell the model this farm sits exactly at its region's mean for all 17 columns — a
    confident claim about a region the training set never sampled.
    """
    df = _frame()
    stats = cluster_stats(df)
    row = df.to_dict("records")[0]

    assert assign_cluster(10.85, 7.66, BOUNDS) == "Kaduna_Grain_Belt"
    assert assign_cluster(0.5, 35.0, BOUNDS) == "Kenya_Rift_Valley"
    assert assign_cluster(48.85, 2.35, BOUNDS) is None          # Paris, no cluster
    assert assign_cluster(10.85, 7.66, None) is None            # artifact predates the snapshot

    orphan = cluster_relative_row(row, stats.get(None))
    assert all(math.isnan(v) for v in orphan.values()), "unknown cluster must not fabricate 0.0"
    assert set(orphan) == {c + CZ_SUFFIX for c in CLUSTER_RELATIVE}, "shape must stay constant"
    print("ok: unknown cluster yields NaN twins in the right shape")


def test_bounds_are_inclusive_at_the_edges():
    assert assign_cluster(10.2, 7.3, BOUNDS) == "Kaduna_Grain_Belt"
    assert assign_cluster(11.5, 8.5, BOUNDS) == "Kaduna_Grain_Belt"
    assert assign_cluster(11.51, 8.5, BOUNDS) is None
    print("ok: cluster bounds are closed intervals")


if __name__ == "__main__":
    test_scalar_matches_vectorised()
    test_the_three_way_split_is_actually_exercised()
    test_unknown_cluster_yields_nan_not_zero()
    test_bounds_are_inclusive_at_the_edges()
    print("\nall cluster-relative checks passed")
