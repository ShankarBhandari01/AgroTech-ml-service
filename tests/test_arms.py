"""Every arm behind one interface, including the ones that fit nothing.

The point of the reformulation is that the baselines stop winning by proxy. `climatology` is a
real fitted arm — each field's mean `ztilde` over training — and under the reformulated target it
converges on the zero predictor as a MEASURED consequence of the target having removed the field
effect, not by construction. `zero` is the arm E02 has to beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argotech.lab.arms import ARMS, resid_sd
from argotech.lab.targets import build_target

FEATS = ["ndvi_z_peer", "rain_30"]


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for k in range(8):
        for j in range(12):
            level = k - 3.5
            rows.append({"site_id": f"S{k}", "cluster": f"C{k % 2}",
                         "obs_date": f"2025-{j + 1:02d}-01",
                         "ndvi_z_peer": level + rng.normal(0, 0.2),
                         "rain_30": rng.normal(50, 10),
                         "forward_z": level + rng.normal(0, 0.2)})
    return pd.DataFrame(rows)


def _split():
    df, feats = build_target(_panel(), "within_y", features=FEATS, min_history=2)
    return df.iloc[:60], df.iloc[60:], feats


def test_every_declared_arm_fits_and_predicts_the_right_shape():
    train, test, feats = _split()
    for name, make in ARMS.items():
        pred = make(42).fit(train, feats).predict(test, feats)
        assert pred.shape == (len(test),), name
        assert np.isfinite(pred).all(), f"{name} emitted a non-finite prediction"


def test_the_zero_arm_predicts_zero():
    train, test, feats = _split()
    pred = ARMS["zero"](42).fit(train, feats).predict(test, feats)
    assert np.allclose(pred, 0.0)


def test_climatology_collapses_onto_zero_under_the_reformulated_target():
    """Under `within_y` the field effect is gone, so "this field is usually weak" predicts nothing.

    Climatology is a real fitted arm (each field's mean ztilde), so this is a measured property of
    the target, not an identity. If build_target stopped subtracting alpha_hat, the per-field means
    would drift off zero and this test would fail — which is the regression it exists to catch.
    """
    train, test, feats = _split()
    clim = ARMS["climatology"](42).fit(train, feats).predict(test, feats)
    assert np.abs(clim).max() < 0.35, (
        f"climatology still carries a per-field signal (max |pred| = {np.abs(clim).max():.3f}); "
        "the target has stopped removing the field effect")


def test_climatology_is_a_real_predictor_under_the_level_target():
    df, feats = build_target(_panel(), "level_z", features=FEATS, min_history=2)
    train, test = df.iloc[:60], df.iloc[60:]
    clim = ARMS["climatology"](42).fit(train, feats).predict(test, feats)
    assert np.abs(clim).max() > 0.1, "under level_z climatology must carry the field effect"


def test_persistence_carries_the_current_within_deviation_forward():
    train, test, feats = _split()
    pred = ARMS["persistence"](42).fit(train, feats).predict(test, feats)
    assert np.allclose(pred, test["ndvi_z_peer"] - test["alpha_hat"])


def test_the_fitted_arms_are_deterministic_given_a_seed():
    train, test, feats = _split()
    for name in ("linear", "boosted"):
        a = ARMS[name](42).fit(train, feats).predict(test, feats)
        b = ARMS[name](42).fit(train, feats).predict(test, feats)
        assert np.allclose(a, b), f"{name} is not reproducible at a fixed seed"


def test_the_fitted_arms_tolerate_a_missing_feature_value():
    train, test, feats = _split()
    holed = test.copy()
    holed.loc[holed.index[0], feats[0]] = np.nan
    for name in ("linear", "boosted"):
        pred = ARMS[name](42).fit(train, feats).predict(holed, feats)
        assert np.isfinite(pred).all(), f"{name} propagated a NaN into its prediction"


def test_resid_sd_is_positive_and_finite():
    train, _, feats = _split()
    sd = resid_sd(ARMS["boosted"](42).fit(train, feats), train, feats)
    assert np.isfinite(sd) and sd > 0


def test_resid_sd_raises_on_a_degenerate_fit():
    """A residual spread of exactly zero cannot become an event probability. It must abort, not
    hand back a plausible-looking placeholder scale."""
    train, _, feats = _split()

    class _Perfect:
        """Predicts the training target exactly, so every residual is 0."""

        def predict(self, frame, features):
            return frame["ztilde"].to_numpy()

    with pytest.raises(ValueError, match="degenerate fit"):
        resid_sd(_Perfect(), train, feats)
