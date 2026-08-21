"""The regressor arm's wiring: hyperparameters matched to the classifier's base estimator, and
the arm table shaped so `evaluate_fold` knows which target each arm is fitted against."""

from __future__ import annotations

import numpy as np


def test_make_regressor_matches_the_classifier_base_hyperparameters():
    """Both arms must differ only in what they predict, or the comparison measures the wrong thing."""
    from argotech.training.train import make_model, make_regressor

    reg = make_regressor(42)
    base = make_model(42).estimator

    for param in ("max_iter", "learning_rate", "max_depth", "min_samples_leaf",
                  "l2_regularization", "early_stopping", "validation_fraction", "random_state"):
        assert getattr(reg, param) == getattr(base, param), f"{param} differs between arms"


def test_arms_declare_which_target_each_is_fitted_against():
    from argotech.training.train import ARMS, PRODUCTION_ARM

    assert PRODUCTION_ARM == "hgbr"
    assert ARMS[PRODUCTION_ARM][1] == "forward_z"
    assert ARMS["hgb"][1] == "label"
    assert ARMS["linear"][1] == "label"
    assert set(ARMS) == {"hgbr", "hgb", "linear"}, "the classifier stays as a control arm"


def test_regressor_ranks_a_worse_field_higher():
    """risk = -predicted_z, so a field predicted to fall further behind must rank above one that
    does not. Gets the sign right, which is invisible in aggregate metrics until the queue inverts."""
    from argotech.training.train import make_regressor

    x = np.arange(200, dtype=float).reshape(-1, 1)
    z = -x[:, 0] / 100.0                      # higher x -> lower z -> worse field
    fitted = make_regressor(42).fit(x, z)
    risk = -fitted.predict(np.array([[10.0], [190.0]]))
    assert risk[1] > risk[0], "the field with the lower predicted z must carry the higher risk"
