"""Model loading.

One model, one path. The previous registry carried a local-pickle branch, an MLflow-registry branch,
a `patch_model` shim that walked stacking estimators restoring a `multi_class` attribute across
scikit-learn versions, and `_align_features` reindexing the frame to whatever columns the estimator
happened to remember. All of that existed to paper over an artifact whose column contract was
unknown at load time. This artifact carries its own `feature_columns`, so the contract is explicit
and the shims are unnecessary.
"""

import logging
import os

import joblib

from argotech.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self._agronomic = None

    def agronomic_model(self):
        """`(model, feature_columns, version, cluster_bounds, cluster_stats)`, cached in-process.

        The version travels into every persisted prediction, so a stored row can always be traced
        back to the artifact that produced it.

        The two cluster members are the reference distribution for the `_cz` twins. They default to
        empty for an artifact trained before they were recorded: `cluster_relative_row` then yields
        NaN twins, which a model that has no `_cz` columns never asks for anyway.
        """
        if self._agronomic is None:
            path = settings.AGRONOMIC_MODEL_PATH
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Agronomic model '{path}' not found. Build it with "
                    "`python -m argotech.lab.panel.panel` then "
                    "`python -m argotech.lab.arms.export experiments/export-production.yaml`.")
            bundle = joblib.load(path)
            target = bundle.get("target")
            if target != "forward_z":
                raise ValueError(
                    f"Artifact '{path}' declares target {target!r}; serving requires 'forward_z'. "
                    "This is refused rather than adapted to: a classifier's `.predict()` returns "
                    "class labels 0/1/2, which the hazard map reads as near-zero anomalies, so "
                    "every field would silently report almost no canopy hazard. Retrain with "
                    "`python -m argotech.lab.panel.panel` then "
                    "`python -m argotech.lab.arms.export experiments/export-production.yaml`."
                )
            if not bundle.get("peer_stats"):
                raise ValueError(
                    f"Artifact '{path}' carries no peer_stats; serving cannot standardise the peer "
                    "anomaly without the reference the model was fitted against. Standardising "
                    "against anything else is how `ndvi_z_peer` came to mean two different things. "
                    "An artifact from before this contract must be retrained.")
            version = bundle.get("version") or f"n{bundle['n_samples']}@{bundle['trained_on']}"
            self._agronomic = (bundle["model"], bundle["feature_columns"], version,
                               bundle.get("cluster_bounds") or {}, bundle.get("cluster_stats") or {},
                               bundle["peer_stats"])
            logger.info("Loaded agronomic model %s from %s: %s on %s, %d features, %d clusters, "
                        "trained on %s", version, os.path.abspath(path),
                        type(bundle["model"]).__name__, target, len(bundle["feature_columns"]),
                        len(self._agronomic[3]), bundle.get("trained_on"))
        return self._agronomic
