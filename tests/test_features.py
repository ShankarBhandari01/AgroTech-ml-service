"""Checks on the two label-free transforms added for cross-region robustness.

Both are cheap to get subtly wrong in ways that silently inflate blocked-CV scores, which is
exactly the failure the blocked protocol exists to catch. One test each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argotech.features.agronomic import (
    CLUSTER_RELATIVE,
    CZ_SUFFIX,
    MODEL_FEATURES,
    UNINFORMATIVE,
    add_cluster_relative,
)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for cluster, offset in (("A", 0.0), ("B", 100.0)):
        for site in range(4):
            for day in range(5):
                row = {c: offset + rng.normal() for c in CLUSTER_RELATIVE}
                row |= {
                    "cluster": cluster,
                    "site_id": f"{cluster}{site}",
                    "obs_date": f"2024-01-0{day + 1}",
                    "ndvi_z_peer": float(day),
                    "label": day % 3,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def test_cluster_relative_removes_the_offset_between_clusters():
    """The whole point: a constant per-cluster shift must not survive the transform."""
    out = add_cluster_relative(_frame())
    for col in CLUSTER_RELATIVE:
        means = out.groupby("cluster")[col + CZ_SUFFIX].mean()
        assert np.allclose(means, 0.0, atol=1e-9), f"{col} keeps a cluster offset"
        # The raw column still carries it — we append, never replace.
        assert abs(out.groupby("cluster")[col].mean().diff().iloc[-1]) > 50


def test_cluster_relative_is_computed_within_fold_only():
    """A held-out cluster's statistics must not shift when other clusters change.

    This is what makes leave-one-cluster-out valid: transforming the full frame once must give the
    same answer for cluster A as transforming cluster A alone.
    """
    df = _frame()
    full = add_cluster_relative(df)
    alone = add_cluster_relative(df[df.cluster == "A"])
    col = CLUSTER_RELATIVE[0] + CZ_SUFFIX
    assert np.allclose(full[full.cluster == "A"][col].to_numpy(), alone[col].to_numpy())


def test_model_features_drop_the_uninformative_and_add_the_twins():
    assert not (UNINFORMATIVE & set(MODEL_FEATURES))
    assert all(c + CZ_SUFFIX in MODEL_FEATURES for c in CLUSTER_RELATIVE)
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES)), "duplicate feature name"

# `add_site_climatology`/`CLIM_COLUMN` (the site-climatology baseline) and `artifact_path` (the
# unbuildable-feature guard) lived in the retired `training/train.py` and have no standalone
# equivalent to repoint at: `lab.arms.Climatology` replaces the former, tested in
# `tests/test_arms.py::test_climatology_collapses_onto_zero_under_the_reformulated_target` and
# `test_climatology_is_a_real_predictor_under_the_level_target`; `lab.export.export_artifact`'s own
# unbuildable-feature guard replaces the latter, tested in
# `tests/test_export.py::test_export_rejects_a_feature_serving_cannot_build`.


# ---------------------------------------------------------------------------------------------
# Sentinel-1 radar block (phase D)
# ---------------------------------------------------------------------------------------------

def test_radar_block_is_nan_when_the_site_has_no_coverage():
    """A missing radar observation must yield NaN, never a substituted value.

    HistGradientBoosting routes NaN down its own branch; a fabricated backscatter silently claims
    the field was observed when it was not.
    """
    from argotech.features.agronomic import radar_block

    block = radar_block(None, [])
    assert set(block) == {"rvi", "vh_vv_ratio", "rvi_z_peer"}
    assert all(v != v for v in block.values()), "absent radar must be NaN, not 0.0"


def test_radar_block_computes_the_ratio_and_peer_anomaly():
    from argotech.features.agronomic import radar_block

    block = radar_block({"vv": 0.2, "vh": 0.05, "rvi": 0.8}, [0.4, 0.2])
    assert block["rvi"] == 0.8
    assert abs(block["vh_vv_ratio"] - 0.25) < 1e-9
    # The peer argument is the (mu, sigma) of the shared reference bucket, not a sample: the caller
    # looks it up, the block never computes it. (0.8 - 0.4) / 0.2 = 2.0.
    assert abs(block["rvi_z_peer"] - 2.0) < 1e-9
    # A constant reference carries no information, and 0.0 says so without becoming an infinity.
    assert radar_block({"vv": 0.2, "vh": 0.05, "rvi": 0.8}, [0.4, 0.0])["rvi_z_peer"] == 0.0
    # No bucket at all is a different statement from "average", and must stay NaN.
    assert np.isnan(radar_block({"vv": 0.2, "vh": 0.05, "rvi": 0.8}, None)["rvi_z_peer"])


def test_nearest_sar_pairs_within_tolerance_and_refuses_beyond_it():
    """S1 and S2 do not share an orbit, so the join is nearest-date with a tolerance."""
    from argotech.features.agronomic import nearest_sar

    sar = [
        {"sensing_date": "2025-03-01", "rvi": 0.1},
        {"sensing_date": "2025-03-20", "rvi": 0.2},
        {"sensing_date": "2025-05-30", "rvi": 0.3},
    ]
    assert nearest_sar(sar, "2025-03-18")["rvi"] == 0.2, "must pick the closest, not the first"
    assert nearest_sar(sar, "2025-04-30") is None, "26 days away is beyond tolerance"
    assert nearest_sar([], "2025-03-18") is None


def test_build_tolerates_a_satellite_block_with_no_radar_keys():
    """The serving path passes an optical-only block; it must still produce a full row."""
    from argotech.features.agronomic import FEATURE_COLUMNS, build

    daily = {
        "time": [f"2025-01-{d:02d}" for d in range(1, 32)] * 3,
        "temperature_2m_max": [30.0] * 93,
        "temperature_2m_min": [18.0] * 93,
        "precipitation_sum": [2.0] * 93,
        "et0_fao_evapotranspiration": [4.0] * 93,
        "relative_humidity_2m_mean": [60.0] * 93,
        "shortwave_radiation_sum": [20.0] * 93,
    }
    # `satellite_block` has already renamed ndwi -> ndmi by the time `build` sees it.
    sat = {"ndvi": 0.5, "ndmi": 0.2, "evi": 0.4, "vci": 50.0, "ndvi_z_peer": -0.5}
    row = build(daily, sat, {"latitude": 10.0, "longitude": 8.0, "elevation": 500.0,
                             "clim_rain_30": 60.0})
    assert set(FEATURE_COLUMNS) <= set(row), "row must cover the stored schema"
    assert row["rvi"] != row["rvi"], "radar-free row carries NaN, not a value"


def test_cluster_relative_keeps_missing_values_missing():
    """A missing input must not become a confident 0.0 twin.

    Radar and optical columns both carry real NaN (no coverage / cloud), and the constant-column
    fallback would otherwise fill them in as "exactly the cluster mean".
    """
    df = _frame()
    col = CLUSTER_RELATIVE[0]
    df.loc[df.index[:3], col] = np.nan
    out = add_cluster_relative(df)
    twin = out[col + CZ_SUFFIX]
    assert twin.iloc[:3].isna().all(), "missing input must give missing twin"
    assert twin.iloc[3:].notna().all(), "observed rows must still standardise"


def test_cluster_relative_still_zeroes_a_constant_column():
    """The other half of the three-way split: constant means no information, and 0.0 says so."""
    df = _frame()
    col = CLUSTER_RELATIVE[1]
    df[col] = 5.0
    out = add_cluster_relative(df)
    assert (out[col + CZ_SUFFIX] == 0.0).all()


def test_cohorts_reject_non_finite_peers():
    """A NaN peer poisons the whole cohort's mean and sd, and crashes pstdev on Python 3.12."""
    import statistics

    from argotech.features.agronomic import finite as _finite

    assert _finite(0.5) and _finite(0) and _finite(-1.2)
    assert not _finite(float("nan"))
    assert not _finite(float("inf"))
    assert not _finite(None)
    assert not _finite("0.5")

    # The failure this guards against, demonstrated rather than asserted in the abstract.
    poisoned = [0.4, 0.5, float("nan")]
    try:
        statistics.pstdev(poisoned)
        crashed = False
    except (AttributeError, TypeError):
        crashed = True
    clean = [v for v in poisoned if _finite(v)]
    assert statistics.pstdev(clean) == statistics.pstdev([0.4, 0.5])
    assert crashed or statistics.pstdev(poisoned) != statistics.pstdev(clean)


# ---------------------------------------------------------------------------------------------
# Ranking metric
# ---------------------------------------------------------------------------------------------

def test_spearman_sign_and_degenerate_cases():
    """The retired `training.train.rank_correlation(risk, forward_z)` and `lab.evaluate.spearman`
    are NOT the same function under a different name, so this is an adaptation, not a repoint:

    * `rank_correlation` scored a decision-theoretic *risk* score against `forward_z`, negating the
      target because risk runs opposite to the anomaly (low z = bad = high risk). `spearman` scores
      a point-*prediction* of `ztilde` directly against `ztilde` itself — both in the same units, low
      together on a good fit — so it does not negate. Confirmed by `lab/run.py`'s own call:
      `spearman(pred, test["ztilde"])`, unnegated.
    * `rank_correlation` returned NaN on a degenerate input (no ordering to measure). `spearman`
      returns 0.0 instead — see its own docstring — matching `precision_at_k`'s same move on a tied
      or constant score.
    * `rank_correlation` rounded to 4 decimal places; `spearman` returns the raw float, so an exact
      correlation compares with `pytest.approx` here rather than `==`.

    Both are monotone-invariant and drop (rather than propagate) a NaN target; that much carries
    over unchanged.
    """
    from argotech.lab.evaluate import spearman

    truth = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])   # ztilde: -2.0 is the worst-off field
    perfect = truth.copy()                          # a perfect point-prediction of ztilde itself
    assert spearman(perfect, truth) == pytest.approx(1.0)
    assert spearman(-perfect, truth) == pytest.approx(-1.0)

    # Monotone-invariant: rescaling the prediction must not move rho.
    assert spearman(perfect * 100 + 7, truth) == pytest.approx(1.0)

    # A degenerate input (no ordering to measure) reads as "no better than chance", not "undefined".
    assert spearman(np.zeros(5), truth) == 0.0
    assert spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0   # n < 3

    # A NaN target is dropped, not propagated: forward_z is absent for a handful of rows.
    assert spearman(perfect, np.array([-2.0, -1.0, 0.0, 1.0, float("nan")])) == pytest.approx(1.0)
