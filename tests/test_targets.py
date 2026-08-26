"""The estimand family, and the one property everything else rests on: alpha_hat sees no future.

The spec's whole argument is that `forward_z` retains a field effect the climatology baseline
collects for free. These pin the transform that removes it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from argotech.lab.targets import TARGET_KINDS, alpha_hat, build_target


def _panel(n_sites: int = 4, n_obs: int = 10) -> pd.DataFrame:
    """A synthetic panel with a real field effect: site k sits at level k - 1.5."""
    rows = []
    for k in range(n_sites):
        for j in range(n_obs):
            rows.append({"site_id": f"S{k}", "cluster": "C0",
                         "obs_date": f"2025-01-{j + 1:02d}",
                         "ndvi_z_peer": (k - 1.5) + 0.1 * ((j % 3) - 1),
                         "forward_z": (k - 1.5) + 0.1 * ((j % 3) - 1)})
    return pd.DataFrame(rows)


def test_alpha_hat_never_sees_the_future():
    df = _panel()
    before = alpha_hat(df)
    spiked = df.copy()
    # Plant an enormous value in every site's LAST observation.
    last = spiked.groupby("site_id").tail(1).index
    spiked.loc[last, "ndvi_z_peer"] = 999.0
    after = alpha_hat(spiked)
    # Every estimate except the ones at or after the spike must be untouched.
    assert np.allclose(before.drop(last), after.drop(last), equal_nan=True), \
        "a future observation moved an earlier alpha_hat: the estimator leaks"


def test_alpha_hat_excludes_the_current_rows_own_value():
    """The same-row check: perturbing row j must not move row j's own estimate.

    The sibling test above spikes a site's LAST row, so it only pins that the future cannot
    rewrite the past. Deleting `.shift(1)` — which puts today's value into today's mean — passes
    that test and every other one in this file. This is the test that fails when it is deleted.
    """
    df = _panel()
    before = alpha_hat(df, min_history=1)
    for row in (5, 12, 27):                       # mid-history rows, each with a real prior window
        perturbed = df.copy()
        perturbed.loc[df.index[row], "ndvi_z_peer"] = 999.0
        after = alpha_hat(perturbed, min_history=1)
        assert before.iloc[row] == after.iloc[row], (
            f"row {row}: alpha_hat moved when that row's OWN value changed — "
            "the estimate includes the current observation")


def test_first_observation_has_no_reference():
    df = _panel()
    first = df.groupby("site_id").head(1).index
    a = alpha_hat(df, min_history=1)
    assert a.loc[first].isna().all(), "a site's first row has no prior history and must be NaN"


def test_min_history_drops_rows_rather_than_defaulting_them():
    df = _panel(n_obs=10)
    out, _ = build_target(df, "within_y", features=["ndvi_z_peer"], min_history=3)
    # Each site loses its first 3 rows; nothing is silently filled with 0.
    assert len(out) == 4 * (10 - 3)
    assert out["alpha_hat"].notna().all()


def test_alpha_hat_recovers_the_planted_field_effect():
    df = _panel()
    a = alpha_hat(df, min_history=5)
    got = df.assign(a=a).dropna(subset=["a"]).groupby("site_id")["a"].mean()
    assert np.allclose(got.to_numpy(), [-1.5, -0.5, 0.5, 1.5], atol=0.05)


def test_within_y_removes_the_field_effect():
    df = _panel()
    out, _ = build_target(df, "within_y", features=["ndvi_z_peer"], min_history=3)
    per_site = out.groupby("site_id")["ztilde"].mean().abs()
    assert (per_site < 0.15).all(), f"field effect survived demeaning: {per_site.to_dict()}"
    assert out["ztilde"].var() < out["forward_z"].var()


def test_level_z_is_the_untransformed_target():
    df = _panel()
    out, _ = build_target(df, "level_z", features=["ndvi_z_peer"], min_history=3)
    assert np.allclose(out["ztilde"], out["forward_z"])


def test_shrink_zero_equals_the_raw_field_mean():
    df = _panel()
    assert np.allclose(alpha_hat(df, shrink=0.0, min_history=1).dropna(),
                       alpha_hat(df, min_history=1).dropna())


def test_shrink_pulls_short_histories_toward_zero():
    df = _panel()
    raw = alpha_hat(df, min_history=1)
    shrunk = alpha_hat(df, shrink=5.0, min_history=1)
    both = df.assign(raw=raw, shrunk=shrunk).dropna(subset=["raw"])
    assert (both["shrunk"].abs() <= both["raw"].abs() + 1e-9).all(), \
        "shrinkage must never move an estimate away from zero"
    # The effect must be strongest where history is shortest.
    early = both.groupby("site_id").head(1)
    late = both.groupby("site_id").tail(1)
    assert (early["shrunk"].abs() / early["raw"].abs()).mean() < \
           (late["shrunk"].abs() / late["raw"].abs()).mean()


def test_within_xy_demeans_the_features_too():
    df = _panel()
    out, feats = build_target(df, "within_xy", features=["ndvi_z_peer"], min_history=3)
    assert feats == ["ndvi_z_peer_w"], "within_xy must hand the arm the demeaned feature names"
    assert abs(out["ndvi_z_peer_w"].mean()) < 0.15
    assert "ndvi_z_peer" in out.columns, "the level column stays available for the baselines"


def test_within_xy_forwards_shrink_to_the_feature_side():
    """build_target demeans the target with cfg['shrink'] but must demean the features with the
    same weight: FWL requires both sides transformed the same way, and a non-zero shrink that only
    reaches the target would silently stop within_xy being a coherent two-way transform."""
    df = _panel()
    unshrunk, _ = build_target(df, "within_xy", features=["ndvi_z_peer"],
                               min_history=3, shrink=0.0)
    shrunk, _ = build_target(df, "within_xy", features=["ndvi_z_peer"],
                             min_history=3, shrink=5.0)
    assert not np.allclose(unshrunk["ndvi_z_peer_w"], shrunk["ndvi_z_peer_w"]), \
        "shrink never reached the feature-side demeaning"


def test_within_xy_drops_a_time_invariant_column(capsys):
    """A column that never varies within a field (e.g. elevation) demeans to exactly zero under
    within_xy — correct FWL behaviour, but dead weight handed to a model as a live feature. It must
    be excluded from the returned feature names, and the exclusion must be visible, not silent."""
    df = _panel()
    df["elevation"] = df["site_id"].map({f"S{k}": float(100 * k) for k in range(4)})
    out, feats = build_target(df, "within_xy", features=["ndvi_z_peer", "elevation"],
                              min_history=3)
    assert feats == ["ndvi_z_peer_w"], f"the degenerate column must be dropped, got {feats}"
    assert "elevation_w" not in out.columns
    assert "elevation" in capsys.readouterr().out, "the drop must be named, not silent"


def _panel_two_clusters() -> pd.DataFrame:
    """`quirky` varies month-to-month for cluster A's sites, but is a hard constant (0.0, every
    month) for cluster B's — the exact shape of the real defect: `heat_stress_days` varies across
    the panel but is identically zero, every month, in `Kaduna_Grain_Belt`."""
    rows = []
    for cluster in ("A", "B"):
        for k in range(3):
            for j in range(6):
                rows.append({"site_id": f"{cluster}{k}", "cluster": cluster,
                             "obs_date": f"2025-01-{j + 1:02d}",
                             "ndvi_z_peer": (k - 1.0) + 0.1 * j,
                             "forward_z": (k - 1.0) + 0.1 * j,
                             "quirky": float(j) if cluster == "A" else 0.0})
    return pd.DataFrame(rows)


def test_within_xy_keeps_a_column_that_only_varies_in_train():
    """BUG-1's fix: degeneracy is decided from `train_mask` alone. A column varying in train (here
    cluster A) but constant across every held-out test row (cluster B) must still be kept — the
    demeaning is legitimate there even though this particular test sample happens not to move."""
    df = _panel_two_clusters()
    train_mask = (df["cluster"] == "A").to_numpy()
    _, feats = build_target(df, "within_xy", features=["ndvi_z_peer", "quirky"],
                            min_history=3, train_mask=train_mask)
    assert "quirky_w" in feats, "a column varying in train must not be dropped for test's sake"


def test_within_xy_drops_a_column_thats_constant_in_train_even_if_test_varies():
    """The other half of the same rule: deciding from the UNION (or from test) would let test data
    alone earn a column a vote it should never get — the same transduction `fit_peer_stats` refuses
    by only ever seeing training rows."""
    df = _panel_two_clusters()
    train_mask = (df["cluster"] == "B").to_numpy()
    _, feats = build_target(df, "within_xy", features=["ndvi_z_peer", "quirky"],
                            min_history=3, train_mask=train_mask)
    assert "quirky_w" not in feats, "a column constant in train must be dropped even if test varies"


def test_delta_z_is_the_first_difference():
    df = _panel()
    out, _ = build_target(df, "delta_z", features=["ndvi_z_peer"], min_history=0)
    assert np.allclose(out["ztilde"], out["forward_z"] - out["ndvi_z_peer"])


def test_every_declared_kind_builds():
    df = _panel()
    for kind in TARGET_KINDS:
        out, feats = build_target(df, kind, features=["ndvi_z_peer"], min_history=2)
        assert "ztilde" in out.columns and len(out) > 0 and feats


def test_persistence_pred_is_the_right_quantity_per_target_kind():
    """"Carry the current reading forward" means a different quantity per estimand: the raw level
    under level_z, the within-deviation under within_y/within_xy, and "no change" under delta_z,
    whose target is already a difference."""
    df = _panel()

    level, _ = build_target(df, "level_z", features=["ndvi_z_peer"], min_history=2)
    assert np.allclose(level["persistence_pred"], level["ndvi_z_peer"])

    for kind in ("within_y", "within_xy"):
        out, _ = build_target(df, kind, features=["ndvi_z_peer"], min_history=2)
        assert np.allclose(out["persistence_pred"], out["ndvi_z_peer"] - out["alpha_hat"]), kind

    delta, _ = build_target(df, "delta_z", features=["ndvi_z_peer"], min_history=0)
    assert np.allclose(delta["persistence_pred"], 0.0)
