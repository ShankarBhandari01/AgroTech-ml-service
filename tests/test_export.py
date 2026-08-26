"""The bundle `lab.export` writes must load through `registry.py`'s OWN loader and pass its
contract check — not merely land on disk. A test that only checks for a file is the specific
failure mode this session hit repeatedly: an artifact that loads fine and then raises `KeyError`
on the first live prediction, because nobody exercised the actual load-and-predict path.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from argotech.features.agronomic import CLUSTER_RELATIVE
from argotech.lab.export import export_artifact, load_config

FEATURES = ["rain_30", "gdd_90_cz", "ndvi"]


def _panel(n_clusters: int = 2, per_cluster: int = 30) -> pd.DataFrame:
    """Every `CLUSTER_RELATIVE` raw column, since `add_cluster_relative`/`cluster_stats`/`peer_stats`
    all iterate that fixed list regardless of which features the config actually trains on."""
    rng = np.random.default_rng(0)
    rows = []
    for c in range(n_clusters):
        for i in range(per_cluster):
            row = {col: float(rng.normal()) for col in CLUSTER_RELATIVE}
            row.update({
                "site_id": f"C{c}-{i % 5}",
                "cluster": f"cluster_{c}",
                "obs_date": f"2024-{(i % 12) + 1:02d}-01",
                "forward_z": float(rng.normal()),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def _cfg(**overrides) -> dict:
    cfg = {"arm": "linear", "seed": 0, "features": FEATURES}
    cfg.update(overrides)
    return cfg


def test_export_writes_a_bundle_registry_loads_and_can_predict_with(tmp_path, monkeypatch):
    from argotech.config import settings
    from argotech.models.registry import ModelManager

    out = export_artifact(_panel(), _cfg(), tmp_path / "agronomic_risk.joblib")
    monkeypatch.setattr(settings, "AGRONOMIC_MODEL_PATH", str(out))

    model, columns, version, bounds, stats, peer_ref = ModelManager().agronomic_model()

    assert columns == FEATURES
    assert version  # non-empty
    assert isinstance(bounds, dict) and bounds  # the real production CLUSTERS, always carried
    assert isinstance(stats, dict) and "cluster_0" in stats and "cluster_1" in stats
    assert isinstance(peer_ref, dict) and peer_ref, "peer_stats must not be empty"

    # The exact call shape serving/pipeline.py makes: one row, sliced by the artifact's own
    # feature_columns. This is what a missing/misnamed feature would KeyError on.
    row = pd.DataFrame([{c: 0.1 for c in FEATURES}])
    pred = model.predict(row[columns])
    assert pred.shape == (1,)
    assert np.isfinite(pred).all()


def test_export_rejects_a_feature_serving_cannot_build():
    with pytest.raises(ValueError, match="not buildable"):
        export_artifact(_panel(), _cfg(features=["not_a_real_feature"]), "unused.joblib")


def test_export_rejects_a_non_sklearn_arm():
    with pytest.raises(ValueError, match="Sklearn-backed"):
        export_artifact(_panel(), _cfg(arm="climatology"), "unused.joblib")


def test_a_bundle_missing_peer_stats_is_rejected_by_registry(tmp_path, monkeypatch):
    """The failure mode this session hit: an export that quietly drops a required key must not load
    silently. Start from a genuine export.py bundle, strip the key a broken export would have
    dropped, and confirm registry.py's loader — not a hand-rolled assertion — refuses it."""
    from argotech.config import settings
    from argotech.models.registry import ModelManager

    good = export_artifact(_panel(), _cfg(), tmp_path / "good.joblib")
    bundle = joblib.load(good)
    del bundle["peer_stats"]
    broken = tmp_path / "broken.joblib"
    joblib.dump(bundle, broken)
    monkeypatch.setattr(settings, "AGRONOMIC_MODEL_PATH", str(broken))

    with pytest.raises(ValueError, match="peer_stats"):
        ModelManager().agronomic_model()


def test_load_config_defaults_to_model_features_and_the_production_arm(tmp_path):
    from argotech.features.agronomic import MODEL_FEATURES

    config = tmp_path / "minimal.yaml"
    config.write_text("name: minimal\n")
    cfg = load_config(config)
    assert cfg["arm"] == "boosted"
    assert cfg["seed"] == 42
    assert cfg["features"] == MODEL_FEATURES
