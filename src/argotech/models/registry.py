"""Model loading.

One model, one path. The previous registry carried a local-pickle branch, an MLflow-registry branch,
a `patch_model` shim that walked stacking estimators restoring a `multi_class` attribute across
scikit-learn versions, and `_align_features` reindexing the frame to whatever columns the estimator
happened to remember. All of that existed to paper over an artifact whose column contract was
unknown at load time. This artifact carries its own `feature_columns`, so the contract is explicit
and the shims are unnecessary.
"""

import os

import joblib

from argotech.config import settings


class ModelManager:
    def __init__(self):
        self._agronomic = None

    def agronomic_model(self):
        """`(model, feature_columns)` for the agronomic risk model, cached in-process."""
        if self._agronomic is None:
            path = settings.AGRONOMIC_MODEL_PATH
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Agronomic model '{path}' not found. Build it with "
                    "`python -m argotech.training.dataset` then `python -m argotech.training.train`.")
            bundle = joblib.load(path)
            self._agronomic = (bundle["model"], bundle["feature_columns"])
            print(f"Loaded agronomic model: {bundle['n_samples']} samples, {bundle['trained_on']}")
        return self._agronomic
