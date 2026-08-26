"""Every arm behind one interface: fit on a training frame, predict `ztilde` on a test frame.

The baselines are arms rather than special cases because the whole argument turns on comparing
against them honestly. `climatology` is a real fitted arm — each field's mean `ztilde` over
training — not a stand-in for `zero`. Under the reformulated target it converges on the zero
predictor as a MEASURED consequence of the target having removed the field effect, which is the
point: the baseline that beat the incumbent can no longer win that comparison by supplying a field
effect the target left in.

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
    """Predict no deviation from the field's own norm.

    What `climatology` converges to once `ztilde` has actually removed the field effect — see
    `Climatology` below for why that arm is fitted rather than this one reused by name.
    """

    def _predict(self, test, features):
        return np.zeros(len(test))


class Climatology(Arm):
    """Each field's mean `ztilde` over the TRAINING rows, carried forward.

    "Is this field usually weak", as opposed to persistence's "is it weak right now". Under
    `level_z` this is the baseline that beat the incumbent model (docs/model-design.md §9.1 A),
    which is why it must be a real fitted quantity rather than a constant: a hardcoded zero would
    make that comparison a strawman.

    Under a correct within-transform it converges on the zero predictor — but as a MEASURED
    consequence of the target having removed the field effect, not by construction. That is what
    makes the equivalence test in test_arms.py meaningful.
    """

    def _fit(self, train, features):
        self.means = train.groupby("site_id")["ztilde"].mean()
        # A site absent from training has no history; the training mean is the neutral answer.
        self.default = float(train["ztilde"].mean())

    def _predict(self, test, features):
        return test["site_id"].map(self.means).fillna(self.default).to_numpy(dtype=float)


class Persistence(Arm):
    """Carry the field's current reading forward, in whatever units `ztilde` is: `persistence_pred`
    is defined per target kind in `targets.build_target` because "current reading" means a
    different quantity for each estimand.
    """

    def _predict(self, test, features):
        return test["persistence_pred"].to_numpy(dtype=float)


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
                                 # Ridge's default (cholesky) solver is a closed-form fit — no
                                 # randomness to seed. `random_state` is accepted but inert there.
                                 Ridge(alpha=1.0)))


def _boosted(seed: int) -> Arm:
    """One gradient-boosted tree. Not a stack: on tabular data this size a stack buys a fraction of
    a point and costs interpretability, latency and four times the retraining surface.

    NaN is routed down its own branch rather than filled. A substituted value would silently claim
    the field was observed.
    """
    return Sklearn(HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_depth=None, min_samples_leaf=25,
        l2_regularization=1.0,
        # sklearn's internal early-stopping split is IID-random within whatever training frame it
        # gets. The outer protocol (splits.py) is spatially blocked and forward-chained in time; a
        # random 15% here can put a site's temporally-adjacent rows on both sides of the internal
        # split, so the stopping iteration would be chosen on leaked, autocorrelated signal inside
        # a protocol whose entire purpose is blocking that. It would also make this arm's comparison
        # against linear (no internal split at all) unfair. Refused: run the full max_iter instead.
        early_stopping=False,
        random_state=seed))


ARMS = {
    "zero": lambda seed: Zero(),
    "persistence": lambda seed: Persistence(),
    "climatology": lambda seed: Climatology(),
    "linear": _linear,
    "boosted": _boosted,
}


def resid_sd(arm: Arm, train: pd.DataFrame, features: list[str]) -> float:
    """In-sample residual spread, used to turn a point prediction into an event probability.

    Deliberately in-sample and deliberately simple: it is a scale for the normal CDF in
    `evaluate.prob_event`, not an uncertainty claim. Distribution-free intervals with a coverage
    guarantee are conformal's job and are out of scope for this plan.

    Raises rather than substituting a placeholder scale on a degenerate fit: a fallback of 1.0 would
    be indistinguishable downstream from a genuine spread of 1.0, and this repository exists to stop
    publishing numbers it cannot stand behind — a degenerate fold should abort a run, not score it.
    """
    resid = train["ztilde"].to_numpy() - arm.predict(train, features)
    sd = float(np.nanstd(resid))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(
            f"{type(arm).__name__}: residual spread is {sd!r}. A degenerate fit cannot be turned "
            "into event probabilities; returning a placeholder scale would produce smooth, "
            "plausible, meaningless numbers.")
    return sd
