"""Every arm behind one interface, including the ones that fit nothing.

The point of the reformulation is that the baselines stop winning by proxy. `zero` is the
climatology baseline under the new target *by construction*, so it is here as an arm rather than a
special case, and it is the one E02 has to beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    assert np.allclose(pred, 0.0), "under ztilde the climatology baseline IS the zero predictor"


def test_the_climatology_arm_is_the_zero_arm_under_the_reformulated_target():
    train, test, feats = _split()
    assert np.allclose(ARMS["climatology"](42).fit(train, feats).predict(test, feats),
                       ARMS["zero"](42).fit(train, feats).predict(test, feats)), \
        "if these differ, the target has not removed the field effect"


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
