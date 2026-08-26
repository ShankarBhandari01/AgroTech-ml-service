"""The estimator behind the spec's headline number, and why it is not naive eta-squared.

E01 puts the field effect at 34.5% of forward_z variance. Naive eta-squared reports 36.0% on the
same data because a group mean over n_i observations carries sigma_eps^2 / n_i of pure noise that
eta-squared credits to the group. On small groups that gap is large, and the whole argument is a
share of variance, so the correction is load-bearing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from argotech.lab.variance import decompose, icc


def _planted(icc_true: float, n_groups: int = 60, n_obs: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    s_between = np.sqrt(icc_true)
    s_within = np.sqrt(1 - icc_true)
    alpha = rng.normal(0, s_between, n_groups)
    g = np.repeat(np.arange(n_groups), n_obs)
    return alpha[g] + rng.normal(0, s_within, n_groups * n_obs), g


def test_icc_recovers_a_planted_value():
    y, g = _planted(0.35)
    assert abs(icc(y, g)["icc"] - 0.35) < 0.06


def test_icc_is_zero_when_groups_carry_no_signal():
    y, g = _planted(0.0)
    assert icc(y, g)["icc"] < 0.03


def test_naive_eta_squared_overstates_it_on_small_groups():
    y, g = _planted(0.20, n_groups=80, n_obs=3)
    r = icc(y, g)
    assert r["eta2_naive"] > r["icc"] + 0.05, \
        "the small-group bias this estimator exists to correct did not appear"


def test_a_single_group_yields_no_estimate():
    assert np.isnan(icc(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0]))["icc"])


def test_decompose_reports_every_grouping_with_an_interval():
    # n_groups=150, not the 30 used elsewhere: the MoM estimator's sampling stdev scales as
    # ~1/sqrt(n_groups), so a single fixed seed's draw stays within the 0.08 tolerance reliably
    # (checked empirically: ~0.068 std at n_groups=30 vs ~0.033 at n_groups=150) without loosening
    # the tolerance and losing the power to catch a genuine estimator regression.
    y, g = _planted(0.30, n_groups=150, n_obs=12)
    df = pd.DataFrame({"forward_z": y, "site_id": [f"S{i}" for i in g],
                       "cluster": [f"C{i % 3}" for i in g],
                       "label_date": [f"2025-{1 + i % 12:02d}-01" for i in range(len(y))]})
    out = decompose(df, n_boot=200)
    assert abs(out["site"]["icc"] - 0.30) < 0.08
    lo, hi = out["icc_ci"]
    assert lo < out["site"]["icc"] < hi
    assert out["var_total"] > 0
