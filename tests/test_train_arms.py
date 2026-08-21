"""The regressor arm's wiring: hyperparameters matched to the classifier's base estimator, and
the arm table shaped so `evaluate_fold` knows which target each arm is fitted against."""

from __future__ import annotations


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

# The `risk = -predicted_z` sign convention is pinned by
# `test_the_model_path_moves_the_vegetation_hazard_the_right_way_and_far_enough` in
# tests/test_e2e_predict.py, on the real serving path. The test that used to live here fitted a
# regressor inside the test body and asserted it was monotone on monotone data — sklearn's property,
# touching neither train.py nor pipeline.py.
