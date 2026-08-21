"""The guard between a rollback and a silent nationwide zero.

A classifier's `.predict()` returns class labels 0/1/2. Fed to regressor-shaped serving code those
labels are read as z values, and `vegetation_hazard_from_anomaly(0..2)` returns 0.002-0.119 — so
every field in the country reads "almost no canopy hazard" with no exception and no log line.
`artifact_path`'s servability guard cannot catch it: the feature set is identical. Only an explicit
contract key can.
"""

from __future__ import annotations

import joblib
import pytest


def _bundle(**overrides) -> dict:
    bundle = {
        "model": object(),
        "feature_columns": ["a", "b"],
        "version": "agro-test",
        "target": "forward_z",
        "n_samples": 10,
        "trained_on": "2024-01-01..2024-02-01",
    }
    bundle.update(overrides)
    return bundle


@pytest.mark.parametrize("bad", [{"target": "label"}, {"target": None}])
def test_an_artifact_with_the_wrong_target_is_refused(tmp_path, monkeypatch, bad):
    from argotech.config import settings
    from argotech.models.registry import ModelManager

    path = tmp_path / "wrong.joblib"
    joblib.dump(_bundle(**bad), path)
    monkeypatch.setattr(settings, "AGRONOMIC_MODEL_PATH", str(path))

    with pytest.raises(ValueError, match="forward_z"):
        ModelManager().agronomic_model()


def test_an_artifact_missing_the_target_key_is_refused(tmp_path, monkeypatch):
    """The pre-change artifact shape. It must fail loudly, not be assumed to be a regressor."""
    from argotech.config import settings
    from argotech.models.registry import ModelManager

    bundle = _bundle()
    del bundle["target"]
    path = tmp_path / "legacy.joblib"
    joblib.dump(bundle, path)
    monkeypatch.setattr(settings, "AGRONOMIC_MODEL_PATH", str(path))

    with pytest.raises(ValueError, match="forward_z"):
        ModelManager().agronomic_model()


def test_the_production_artifact_declares_the_regression_target():
    """The other half of the contract: what training writes is what serving demands."""
    import joblib as jl

    from argotech.config import settings

    bundle = jl.load(settings.AGRONOMIC_MODEL_PATH)
    assert bundle["target"] == "forward_z"
    assert hasattr(bundle["model"], "predict")
    assert not hasattr(bundle["model"], "predict_proba"), "a regressor has no posterior"
