"""Every arm behind one interface: fit on a training frame, predict `ztilde` on a test frame.

The baselines are arms rather than special cases because the whole argument turns on comparing
against them honestly. Under the reformulated target `climatology` collapses onto `zero` by
construction, which is the point: the baseline that beat the incumbent can no longer win by
supplying a field effect the target left in.

Regressors, not classifiers. `ztilde` is continuous; the three-class discretisation in the
incumbent existed to serve a control classifier and is not carried over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Arm:
    """Fit-and-predict. Subclasses override `_fit` and `_predict`."""

    def fit(self, train: pd.DataFrame, features: list[str]) -> Arm:
        self._fit(train, features)
        return self

    def predict(self, test: pd.DataFrame, features: list[str]) -> np.ndarray:
        return np.asarray(self._predict(test, features), dtype=float)

    def _fit(self, train, features): ...

    def _predict(self, test, features): raise NotImplementedError


class Zero(Arm):
    """Predict no deviation from the field's own norm. Climatology, under this target."""

    def _predict(self, test, features):
        return np.zeros(len(test))


class Persistence(Arm):
    """Carry the field's current within-deviation forward."""

    def _predict(self, test, features):
        return (test["ndvi_z_peer"] - test["alpha_hat"]).fillna(0.0).to_numpy()


class Sklearn(Arm):
    """A scikit-learn estimator, with NaN handling stated rather than assumed."""

    def __init__(self, estimator):
        self.estimator = estimator

    def _fit(self, train, features):
        X, y = train[features], train["ztilde"]
        ok = y.notna()
        self.estimator.fit(X[ok], y[ok])

    def _predict(self, test, features):
        return self.estimator.predict(test[features])


def _linear(seed: int) -> Arm:
    """Ridge over standardised features, median-imputed.

    A linear arm is here because the cross-region literature finds simpler models transfer better
    under shift, and because on the incumbent it did not merely match the boosted trees out of
    cluster — it beat them, 0.421 to 0.408 (docs/model-design.md section 9.1 C). It handicaps
    itself by needing imputation where the boosted arm routes NaN natively, which makes the result
    stronger rather than weaker.
    """
    return Sklearn(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 Ridge(alpha=1.0, random_state=seed)))


def _boosted(seed: int) -> Arm:
    """One gradient-boosted tree. Not a stack: on tabular data this size a stack buys a fraction of
    a point and costs interpretability, latency and four times the retraining surface.

    NaN is routed down its own branch rather than filled. A substituted value would silently claim
    the field was observed.
    """
    return Sklearn(HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_depth=None, min_samples_leaf=25,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=seed))


ARMS = {
    "zero": lambda seed: Zero(),
    "persistence": lambda seed: Persistence(),
    "climatology": lambda seed: Zero(),
    "linear": _linear,
    "boosted": _boosted,
}


def resid_sd(arm: Arm, train: pd.DataFrame, features: list[str]) -> float:
    """In-sample residual spread, used to turn a point prediction into an event probability.

    Deliberately in-sample and deliberately simple: it is a scale for the normal CDF in
    `evaluate.prob_event`, not an uncertainty claim. Distribution-free intervals with a coverage
    guarantee are conformal's job and are out of scope for this plan.
    """
    resid = train["ztilde"].to_numpy() - arm.predict(train, features)
    sd = float(np.nanstd(resid))
    return sd if np.isfinite(sd) and sd > 0 else 1.0
