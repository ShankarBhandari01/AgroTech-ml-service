"""Train and honestly evaluate the agronomic risk model.

Evaluation protocol, per docs/model-design.md §6. Three things are reported, and the third is the
one that matters:

1. **Leave-one-cluster-out** — spatially blocked. Neighbouring fields share weather cells and
   satellite scenes, so a shuffled K-fold leaks and inflates. This answers "does it work in a
   district we have never seen".
2. **Forward-chaining temporal split** — train on observations up to a cutoff date, test after it.
   This answers "does it work next month".
3. **Baselines it must beat** — majority class, and persistence (carry the current peer anomaly
   forward). A model that cannot beat persistence has learned nothing about dynamics, and saying so
   is more useful than a headline F1.

Run: `python -m argotech.training.train`
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

from argotech.features.agronomic import FEATURE_COLUMNS
from argotech.training.dataset import ELEVATED_Z, SEVERE_Z

ARTIFACT = Path("artifacts/agronomic_risk.joblib")
METRICS = Path("artifacts/metrics.json")


def make_model(seed: int = 42) -> CalibratedClassifierCV:
    """One gradient-boosted tree, cross-fit calibrated.

    Not a four-estimator stack under a logistic meta-learner: on tabular data at this scale the
    stack bought a fraction of a point and cost interpretability, latency and four times the
    retraining surface. `cv=5` here means calibration is fitted on held-out folds — the previous
    pipeline used `cv="prefit"` on the training data itself, which taught the sigmoid to reproduce
    in-sample scores and made the output *more* overconfident, not less.
    """
    base = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=5,
        min_samples_leaf=25,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )
    # Sigmoid rather than isotonic: isotonic needs more data per fold than the minority classes have
    # here, and overfits the calibration map.
    #
    # No `class_weight="balanced"` here, deliberately. Reweighting the base estimator and then
    # calibrating it are opposed operations — calibration maps the inflated minority scores back
    # onto the observed prior, and argmax collapses onto class 0 entirely (balanced accuracy 0.333,
    # every permutation importance 0.0). Honest probabilities plus an explicit decision rule is the
    # correct pairing; see `decide` below.
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)


def expected_severity(proba: np.ndarray) -> np.ndarray:
    """Ordinal risk score in [0, 2]. The single number to rank by."""
    return proba[:, 1] + 2.0 * proba[:, 2]


def decide(risk: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    """Turn the ordinal risk score into a class by matching the training prior.

    The label is ordinal (0 < 1 < 2) and heavily imbalanced, so `argmax` over calibrated
    probabilities is the wrong decision rule: it is optimal for 0-1 loss on a balanced problem and
    for nothing else here. Cutting the risk score at the training prior's quantiles keeps the
    predicted class distribution honest and lets the operating point move with the cost ratio when
    the product decides what a missed outbreak costs relative to a wasted visit.
    """
    prior = np.bincount(train_labels, minlength=3) / len(train_labels)
    q1, q2 = 1.0 - prior[1] - prior[2], 1.0 - prior[2]
    t1, t2 = np.quantile(risk, q1), np.quantile(risk, q2)
    return np.where(risk >= t2, 2, np.where(risk >= t1, 1, 0))


# ---------------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------------

def baseline_majority(y_train: np.ndarray, n: int) -> np.ndarray:
    return np.full(n, np.bincount(y_train).argmax())


def persistence_risk(X_test: pd.DataFrame) -> np.ndarray:
    """Ranking score for the persistence baseline: lower peer anomaly means higher risk."""
    return -X_test["ndvi_z_peer"].to_numpy()


def baseline_persistence(X_test: pd.DataFrame) -> np.ndarray:
    """Carry the current peer anomaly forward, thresholded exactly as the label is.

    Vegetation is strongly autocorrelated month to month, so this is a genuinely strong baseline and
    the one the learned model has to justify itself against.
    """
    z = X_test["ndvi_z_peer"].to_numpy()
    return np.where(z <= SEVERE_Z, 2, np.where(z <= ELEVATED_Z, 1, 0))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    """ECE over the predicted-class confidence."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def precision_at_k(y_true: np.ndarray, risk: np.ndarray, k: int) -> float:
    """Share of the top-k highest-risk fields that really were stressed.

    The operational metric: an agent visits k farms this week, and what matters is how many of those
    visits land on a field that needed one. Weighted F1 does not answer that question.
    """
    if k <= 0 or k > len(risk):
        return float("nan")
    top = np.argsort(-risk)[:k]
    return float((y_true[top] >= 1).mean())


def _score(name: str, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None,
           risk: np.ndarray | None = None) -> dict:
    out = {
        "split": name,
        "n": int(len(y_true)),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
    }
    if proba is not None:
        out["ece"] = round(expected_calibration_error(y_true, proba), 4)
    if risk is not None:
        for k in (10, 25, 50):
            out[f"precision_at_{k}"] = round(precision_at_k(y_true, risk, min(k, len(risk))), 4)
    return out


# ---------------------------------------------------------------------------------------------
# Evaluation protocols
# ---------------------------------------------------------------------------------------------

def evaluate_spatial(df: pd.DataFrame) -> list[dict]:
    """Leave-one-cluster-out."""
    results = []
    for cluster in sorted(df.cluster.unique()):
        train, test = df[df.cluster != cluster], df[df.cluster == cluster]
        if len(test) < 30 or train.label.nunique() < 3:
            continue
        Xtr, ytr = train[FEATURE_COLUMNS], train.label.to_numpy()
        Xte, yte = test[FEATURE_COLUMNS], test.label.to_numpy()

        model = make_model().fit(Xtr, ytr)
        proba = model.predict_proba(Xte)
        risk = expected_severity(proba)

        results.append({
            **_score(f"held-out cluster: {cluster}", yte, decide(risk, ytr), proba, risk),
            "baseline_majority_f1": round(float(f1_score(
                yte, baseline_majority(ytr, len(yte)), average="macro", zero_division=0)), 4),
            "baseline_persistence_f1": round(float(f1_score(
                yte, baseline_persistence(Xte), average="macro", zero_division=0)), 4),
            "baseline_persistence_p25": round(precision_at_k(yte, persistence_risk(Xte), min(25, len(yte))), 4),
        })
    return results


def evaluate_temporal(df: pd.DataFrame, n_folds: int = 3) -> list[dict]:
    """Forward chaining over observation dates."""
    dates = sorted(df.obs_date.unique())
    if len(dates) < n_folds + 4:
        return []
    results = []
    for i in range(n_folds):
        cut = dates[int(len(dates) * (0.55 + 0.12 * i))]
        train, test = df[df.obs_date < cut], df[df.obs_date >= cut]
        # Only the immediately following period, so each fold tests one step ahead.
        horizon = sorted(test.obs_date.unique())[:2]
        test = test[test.obs_date.isin(horizon)]
        if len(test) < 30 or train.label.nunique() < 3:
            continue
        Xtr, ytr = train[FEATURE_COLUMNS], train.label.to_numpy()
        Xte, yte = test[FEATURE_COLUMNS], test.label.to_numpy()

        model = make_model().fit(Xtr, ytr)
        proba = model.predict_proba(Xte)
        risk = expected_severity(proba)

        results.append({
            **_score(f"train < {cut}, test {horizon[0]}..", yte, decide(risk, ytr), proba, risk),
            "baseline_majority_f1": round(float(f1_score(
                yte, baseline_majority(ytr, len(yte)), average="macro", zero_division=0)), 4),
            "baseline_persistence_f1": round(float(f1_score(
                yte, baseline_persistence(Xte), average="macro", zero_division=0)), 4),
            "baseline_persistence_p25": round(precision_at_k(yte, persistence_risk(Xte), min(25, len(yte))), 4),
        })
    return results


def permutation_importance_blocked(df: pd.DataFrame, seed: int = 0, repeats: int = 3) -> dict:
    """Feature importance measured on a held-out cluster, by permutation.

    Split importances (what the previous pipeline reported) describe how a tree used a column during
    fitting, including columns that only helped it memorise. Permutation importance on unseen ground
    describes what actually carries transferable signal.
    """
    cluster = sorted(df.cluster.unique())[-1]
    train, test = df[df.cluster != cluster], df[df.cluster == cluster]
    ytr = train.label.to_numpy()
    model = make_model().fit(train[FEATURE_COLUMNS], ytr)
    Xte, yte = test[FEATURE_COLUMNS].copy(), test.label.to_numpy()

    def macro(X):
        return f1_score(yte, decide(expected_severity(model.predict_proba(X)), ytr),
                        average="macro", zero_division=0)

    base = macro(Xte)
    rng = np.random.default_rng(seed)
    drops = {}
    for col in FEATURE_COLUMNS:
        deltas = []
        original = Xte[col].to_numpy().copy()
        for _ in range(repeats):
            Xte[col] = rng.permutation(original)
            deltas.append(base - macro(Xte))
        Xte[col] = original
        drops[col] = round(float(np.mean(deltas)), 4)
    return dict(sorted(drops.items(), key=lambda kv: -kv[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training_set.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    print(f"{len(df)} samples, {df.site_id.nunique()} sites, {df.cluster.nunique()} clusters, "
          f"{df.obs_date.min()}..{df.obs_date.max()}")
    print("class balance:", df.label.value_counts(normalize=True).sort_index().round(3).to_dict())

    print("\n=== Spatially blocked (leave-one-cluster-out) ===")
    spatial = evaluate_spatial(df)
    for r in spatial:
        print(f"  {r['split']:<42} n={r['n']:<5} macroF1={r['macro_f1']:.3f}  "
              f"balAcc={r['balanced_accuracy']:.3f}  ECE={r['ece']:.3f}  "
              f"P@25={r['precision_at_25']:.3f}  "
              f"[F1 base: majority {r['baseline_majority_f1']:.3f} / persistence {r['baseline_persistence_f1']:.3f}"
              f" | P@25 persistence {r['baseline_persistence_p25']:.3f}]")

    print("\n=== Forward-chaining temporal ===")
    temporal = evaluate_temporal(df)
    for r in temporal:
        print(f"  {r['split']:<42} n={r['n']:<5} macroF1={r['macro_f1']:.3f}  "
              f"balAcc={r['balanced_accuracy']:.3f}  ECE={r['ece']:.3f}  "
              f"P@25={r['precision_at_25']:.3f}  "
              f"[F1 base: majority {r['baseline_majority_f1']:.3f} / persistence {r['baseline_persistence_f1']:.3f}"
              f" | P@25 persistence {r['baseline_persistence_p25']:.3f}]")

    print("\n=== Permutation importance on a held-out cluster (top 12) ===")
    importance = permutation_importance_blocked(df)
    for name, drop in list(importance.items())[:12]:
        print(f"  {name:<24} {drop:+.4f}")

    # Final artifact: fitted on everything, since the estimates above already tell us what it is worth.
    print("\nFitting final model on all data ...")
    y_all = df.label.to_numpy()
    final = make_model().fit(df[FEATURE_COLUMNS], y_all)
    holdout = df[df.cluster == sorted(df.cluster.unique())[-1]]
    held_risk = expected_severity(final.predict_proba(holdout[FEATURE_COLUMNS]))
    print("(in-sample for the held-out cluster — the blocked numbers above are the honest ones)")
    print(classification_report(holdout.label, decide(held_risk, y_all), zero_division=0, digits=3))

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    version = f"agro-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    joblib.dump({
        "model": final,
        "version": version,
        "feature_columns": FEATURE_COLUMNS,
        "classes": [0, 1, 2],
        "label": "peer-standardised NDVI anomaly 30 days ahead",
        "thresholds": {"severe_z": SEVERE_Z, "elevated_z": ELEVATED_Z},
        "n_samples": int(len(df)),
        "trained_on": f"{df.obs_date.min()}..{df.obs_date.max()}",
    }, ARTIFACT)

    METRICS.write_text(json.dumps({
        "spatial_blocked": spatial,
        "temporal_forward": temporal,
        "permutation_importance": importance,
        "n_samples": int(len(df)),
        "class_balance": df.label.value_counts(normalize=True).sort_index().round(4).to_dict(),
    }, indent=2, default=str))
    print(f"Saved {ARTIFACT} ({version}) and {METRICS}")


if __name__ == "__main__":
    main()
