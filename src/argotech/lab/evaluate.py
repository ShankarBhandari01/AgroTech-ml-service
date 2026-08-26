"""What "good" means, once ranking and classification stop agreeing.

The incumbent reports macro F1 and precision@25, they disagree, and docs/model-design.md section 6
leaves the choice as a decision "someone has to make consciously" that nobody has made. Net benefit
(Vickers & Elkin, Medical Decision Making 26:565-574, 2006) makes it unnecessary: a model is
evaluated by the consequences of acting on it, across the whole range of cost ratios, against
visiting every field and visiting none.

    NB(t) = TP/n - (FP/n) * (t / (1 - t))

The threshold t is the probability at which an extension visit becomes worthwhile, so t/(1-t) is
exactly the missed-outbreak-to-wasted-trip cost ratio. Reporting the curve rather than one point is
what turns section 9.1 D's alert-rate table into an answer.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr

# The incumbent's severe cut, reused so an event means the same thing across documents.
DEFAULT_TAU = -1.0
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def prob_event(pred: np.ndarray, resid_sd: float, tau: float = DEFAULT_TAU) -> np.ndarray:
    """P(ztilde <= tau | prediction), as a normal CDF over the training residual spread.

    Net benefit needs probabilities, and the arms emit point predictions on a continuous target.
    Assuming normal residuals is the cheapest defensible bridge; it is an assumption and it is
    stated here rather than buried. Distribution-free coverage is conformal's job, out of scope.
    """
    sd = resid_sd if resid_sd > 0 else 1.0
    z = (tau - np.asarray(pred, dtype=float)) / sd
    return np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in z])


def net_benefit(y: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    """Net benefit of acting on `prob` at `threshold`, in true-positives-per-case units."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0 or not 0.0 < threshold < 1.0:
        return 0.0
    flagged = np.asarray(prob, dtype=float) >= threshold
    tp = float(np.sum(flagged & (y == 1)))
    fp = float(np.sum(flagged & (y == 0)))
    return tp / n - (fp / n) * (threshold / (1.0 - threshold))


def decision_curve(y: np.ndarray, prob: np.ndarray,
                    thresholds=DEFAULT_THRESHOLDS) -> list[dict]:
    """The model against both default strategies, at every threshold.

    `visit_none` is 0 by definition. `visit_all` flags everything, so its net benefit is
    prevalence - (1 - prevalence) * odds(t). A model is worth deploying only where it clears both.
    """
    y = np.asarray(y, dtype=float)
    ones = np.ones(len(y))
    return [{"threshold": float(t),
             "model": net_benefit(y, prob, t),
             "visit_all": net_benefit(y, ones, t),
             "visit_none": 0.0}
            for t in thresholds]


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    """Share of the top-k ranked cases that were events — an agent visits k farms this week."""
    y = np.asarray(y, dtype=float)
    k = min(int(k), len(y))
    if k <= 0:
        return float("nan")
    top = np.argsort(-np.asarray(score, dtype=float), kind="stable")[:k]
    return float(np.mean(y[top] == 1))


def spearman(score: np.ndarray, truth: np.ndarray) -> float:
    """Rank correlation, NaN-safe and 0.0 rather than NaN on a degenerate input."""
    s, t = np.asarray(score, dtype=float), np.asarray(truth, dtype=float)
    ok = np.isfinite(s) & np.isfinite(t)
    if ok.sum() < 3 or np.std(s[ok]) == 0 or np.std(t[ok]) == 0:
        return 0.0
    rho = spearmanr(s[ok], t[ok]).statistic
    return float(rho) if np.isfinite(rho) else 0.0


def bootstrap_ci(values, alpha: float = 0.05, n: int = 2000,
                  seed: int = 0) -> tuple[float, float]:
    """Percentile CI on the mean of per-fold scores.

    Every headline number in artifacts/metrics.json is a mean over four or six folds reported
    without one, and section 9.1 G records a precision@25 seed spread of 0.14 on a metric computed
    over 25 items. Fold means without an interval are how that goes unnoticed.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
