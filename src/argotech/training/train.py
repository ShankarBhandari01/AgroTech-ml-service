"""Train and honestly evaluate the agronomic risk model.

Evaluation protocol, per docs/model-design.md §6. Three things are reported, and the third is the
one that matters:

1. **Leave-one-cluster-out** — spatially blocked. Neighbouring fields share weather cells and
   satellite scenes, so a shuffled K-fold leaks and inflates. This answers "does it work in a
   district we have never seen".
2. **Forward-chaining temporal split** — train on observations up to a cutoff date, test after it.
   This answers "does it work next month".
3. **Baselines it must beat** — three, because naming only the easiest one is how a weak model gets
   promoted: majority class; persistence (carry the current peer anomaly forward, "is this field
   weak *now*"); and site climatology (its mean prior anomaly, "is this field *usually* weak"). A
   model that cannot beat all three has learned nothing, and saying so is more useful than a
   headline F1.

Alongside those, two things that make the numbers interpretable rather than merely reported:

* **A linear arm** beside the boosted trees, because the cross-region literature finds simpler
  models transfer better under distribution shift. It is a ceiling check, not a candidate.
* **A decision-rule sweep**, because `decide` cuts at the training prior by choice, not derivation,
  and the operating point belongs to whoever owns the cost of a missed outbreak.

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from argotech.features.agronomic import (
    CLUSTER_RELATIVE,
    CZ_SUFFIX,
    FEATURE_COLUMNS,
    MODEL_FEATURES,
    RADAR_FEATURES,
    UNINFORMATIVE,
    add_cluster_relative,
    cluster_stats,
)
from argotech.training.dataset import CLUSTERS, ELEVATED_Z, SEVERE_Z

PRODUCTION_ARTIFACT = Path("artifacts/agronomic_risk.joblib")
EXPERIMENT_DIR = Path("artifacts/experimental")


# What `serving/pipeline.py` can put in front of the model: the raw row from `agronomic.build`, plus
# the cluster-relative twins it now derives from the `cluster_stats` snapshot carried in the bundle.
SERVABLE_FEATURES = frozenset(FEATURE_COLUMNS) | {c + CZ_SUFFIX for c in CLUSTER_RELATIVE}


def artifact_path(features: list[str]) -> Path:
    """Where this model may be written, decided by whether serving can actually feed it.

    `serving/pipeline.py` builds one row from `FEATURE_COLUMNS`, appends the `_cz` twins via
    `cluster_relative_row`, and slices the result by the artifact's own `feature_columns`. A model
    trained on anything outside that set would load fine and then raise a KeyError on the first live
    prediction. That is a train/serve skew of exactly the kind documented as P0-2 in
    docs/model-design.md, so it is a guard rather than a comment: a feature set serving cannot build
    does not get the production path.
    """
    if set(features) <= SERVABLE_FEATURES:
        return PRODUCTION_ARTIFACT
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    return EXPERIMENT_DIR / "agronomic_risk_clusterrel.joblib"


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


def make_linear(seed: int = 42):
    """A multinomial logistic head over the same features — the shift-robustness arm.

    Not an ensemble member and not a candidate for production on its own. It is here because the
    cross-region literature consistently finds simpler models transfer better: in the leave-one-
    country-out maize study (arXiv 2605.08113) ridge showed the smallest random-CV -> LOCO gap
    (0.207 R^2 units) and the tree ensembles the largest (0.284). If this arm closes on the boosted
    trees out-of-cluster, that is a statement about the ceiling of the feature set, obtained cheaply.

    The median imputer is the arm's handicap, not a preprocessing detail: 13 rows have no usable
    Sentinel-2 observation and 210 have no peer cohort, and `HistGradientBoostingClassifier` routes
    those down its own missing-value branch rather than guessing a value. Imputing to the median
    tells the linear model a cloudy field is an average field. That is a real disadvantage for this
    arm and it is the honest comparison, since removing the rows would change the test set.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5, random_state=seed),
    )


# The arms evaluated side by side in every fold. The first is the production candidate.
ARMS = {"hgb": make_model, "linear": make_linear}

# Rebound by `--seed`. Only the boosted arm is genuinely stochastic: `HistGradientBoostingClassifier`
# draws its own early-stopping validation split from `random_state`, whereas lbfgs logistic
# regression and the leave-one-cluster-out splits are deterministic. So a seed sweep measures the
# boosted arm's variance — which is exactly what a single 6-fold mean cannot tell you.
SEED = 42


def expected_severity(proba: np.ndarray) -> np.ndarray:
    """Ordinal risk score in [0, 2]. The single number to rank by.

        s(x) = E[Y | x] = SUM_k k * P(Y = k | x) = P(Y=1|x) + 2 * P(Y=2|x)

    The expectation of an ordinal label under the calibrated posterior. Using the expectation rather
    than argmax is what makes the score rankable: two fields can both be argmax-class-0 and still
    differ by an order of magnitude in expected severity.
    """
    return proba[:, 1] + 2.0 * proba[:, 2]


def decide(risk: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    """Turn the ordinal risk score into a class by matching the training prior.

    The label is ordinal (0 < 1 < 2) and heavily imbalanced, so `argmax` over calibrated
    probabilities is the wrong decision rule: it is optimal for 0-1 loss on a balanced problem and
    for nothing else here. Cutting the risk score at the training prior's quantiles keeps the
    predicted class distribution honest and lets the operating point move with the cost ratio when
    the product decides what a missed outbreak costs relative to a wasted visit.

        pi_k  = n_k / n                        (training prior for class k)
        q1    = 1 - pi_1 - pi_2                (quantile of the "at least elevated" cut)
        q2    = 1 - pi_2                       (quantile of the "severe" cut)
        t_j   = Quantile(s, q_j)               (empirical quantile of the risk score)
        y_hat = 2 if s >= t2 else 1 if s >= t1 else 0
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

        z_t        = ndvi_z_peer at the prediction date
        y_hat      = 2 if z_t <= SEVERE_Z else 1 if z_t <= ELEVATED_Z else 0

    i.e. assume z_{t+30} = z_t and apply the label's own thresholds (-1.0, -0.35).
    """
    z = X_test["ndvi_z_peer"].to_numpy()
    return np.where(z <= SEVERE_Z, 2, np.where(z <= ELEVATED_Z, 1, 0))


CLIM_COLUMN = "clim_z_prior"


def add_site_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Each row's site's mean peer anomaly over its own *strictly earlier* observations.

    This is the climatology baseline, adapted to our target. A seasonal NDVI climatology — the
    reference used in the African subseasonal drought literature (arXiv 2605.05255), which chose it
    over persistence precisely because it scored higher — is not directly meaningful here: our label
    is already standardised against the concurrent peer cohort, so its seasonal climatology is ~0 by
    construction and would collapse onto the majority baseline.

    What *is* meaningful is that some fields sit persistently below their cluster's peers. This
    baseline asks "is this field usually weak", where persistence asks "is this field weak right
    now". They are different questions and the first is often the harder one to beat.

    For site s with observations ordered by date and z_j = ndvi_z_peer at its j-th observation:

        c_j = (1 / j) * SUM_{m < j} z_m        (expanding mean over strictly earlier observations)
        c_0 = 0                                (no history yet -> cluster-neutral)

    then thresholded exactly as the label is, and negated for the ranking score.

    Uses only past values of a feature, never a label, so it is leak-free under both protocols.
    """
    out = df.sort_values(["site_id", "obs_date"]).copy()
    prior_mean = (out.groupby("site_id")["ndvi_z_peer"]
                     .transform(lambda s: s.expanding().mean().shift(1)))
    # A site's first observation has no history; 0.0 is the cluster-neutral anomaly.
    out[CLIM_COLUMN] = prior_mean.fillna(0.0)
    return out.loc[df.index]


def climatology_risk(X_test: pd.DataFrame) -> np.ndarray:
    """Ranking score for the climatology baseline: a chronically weak field ranks as high risk."""
    return -X_test[CLIM_COLUMN].to_numpy()


def baseline_climatology(X_test: pd.DataFrame) -> np.ndarray:
    z = X_test[CLIM_COLUMN].to_numpy()
    return np.where(z <= SEVERE_Z, 2, np.where(z <= ELEVATED_Z, 1, 0))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    """ECE over the predicted-class confidence.

        conf_i = max_k P(Y=k | x_i)
        pred_i = argmax_k P(Y=k | x_i)
        acc(B) = (1/|B|) * SUM_{i in B} 1[pred_i = y_i]
        cnf(B) = (1/|B|) * SUM_{i in B} conf_i
        ECE    = SUM_B (|B| / n) * | acc(B) - cnf(B) |

    with B ranging over `bins` equal-width bins of confidence on (0, 1]. Zero means the model's
    stated confidence matches its observed accuracy at every confidence level.
    """
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

        T_k       = indices of the k largest values of the risk score s
        P@k       = (1/k) * SUM_{i in T_k} 1[y_i >= 1]

    Note the >= 1: both elevated and severe count as a visit worth making.
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

def _macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)


def evaluate_fold(name: str, train: pd.DataFrame, test: pd.DataFrame, oof: list | None = None) -> dict:
    """Score every arm and every baseline on one train/test split.

    One implementation for both protocols. The baselines are recomputed per fold rather than once
    globally because `decide` and `baseline_majority` both depend on the *training* prior, which
    differs by fold.
    """
    ytr, yte = train.label.to_numpy(), test.label.to_numpy()
    Xtr, Xte = train[MODEL_FEATURES], test[MODEL_FEATURES]

    result = {"split": name, "n": int(len(yte))}
    for arm, factory in ARMS.items():
        proba = factory(SEED).fit(Xtr, ytr).predict_proba(Xte)
        risk = expected_severity(proba)
        scored = _score(name, yte, decide(risk, ytr), proba, risk)
        for key, value in scored.items():
            if key not in ("split", "n"):
                result[f"{key}_{arm}" if arm != "hgb" else key] = value
        if oof is not None and arm == "hgb":
            oof.append((yte, risk, persistence_risk(test), climatology_risk(test), ytr))

    result.update({
        "baseline_majority_f1": _macro(yte, baseline_majority(ytr, len(yte))),
        "baseline_persistence_f1": _macro(yte, baseline_persistence(test)),
        "baseline_persistence_p25": round(precision_at_k(yte, persistence_risk(test), min(25, len(yte))), 4),
        "baseline_climatology_f1": _macro(yte, baseline_climatology(test)),
        "baseline_climatology_p25": round(precision_at_k(yte, climatology_risk(test), min(25, len(yte))), 4),
    })
    return result


def evaluate_spatial(df: pd.DataFrame, oof: list | None = None) -> list[dict]:
    """Leave-one-cluster-out."""
    results = []
    for cluster in sorted(df.cluster.unique()):
        train, test = df[df.cluster != cluster], df[df.cluster == cluster]
        if len(test) < 30 or train.label.nunique() < 3:
            continue
        results.append(evaluate_fold(f"held-out cluster: {cluster}", train, test, oof))
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
        results.append(evaluate_fold(f"train < {cut}, test {horizon[0]}..", train, test))
    return results


def sweep_decision_rule(oof: list) -> list[dict]:
    """Is the prior-matching cut in `decide` actually the best operating point?

    `decide` cuts the risk score at the training prior's quantiles. That keeps the predicted class
    distribution honest, but it is a choice, not a derivation — the right cut depends on what a
    missed outbreak costs relative to a wasted visit. This sweeps a multiplier on the alert rate:
    1.0 is the current rule, 2.0 flags twice as many fields as the prior implies.

        p_k(m) = min(m * pi_k, 0.98)           for k in {1, 2}, m the multiplier
        q1     = 1 - p_1(m) - p_2(m)
        q2     = 1 - p_2(m)

    and then the same quantile cut as `decide`. At m = 1 this reproduces `decide` exactly, which is
    the control row of the sweep.
    """
    y_true = np.concatenate([o[0] for o in oof])
    risk = np.concatenate([o[1] for o in oof])
    ytr_all = np.concatenate([o[4] for o in oof])
    prior = np.bincount(ytr_all, minlength=3) / len(ytr_all)

    rows = []
    for mult in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        p1, p2 = min(prior[1] * mult, 0.98), min(prior[2] * mult, 0.98)
        q1, q2 = 1.0 - p1 - p2, 1.0 - p2
        if q1 <= 0 or q1 >= q2:
            continue
        t1, t2 = np.quantile(risk, q1), np.quantile(risk, q2)
        pred = np.where(risk >= t2, 2, np.where(risk >= t1, 1, 0))
        rows.append({
            "alert_rate_multiplier": mult,
            "flagged_share": round(float((pred >= 1).mean()), 4),
            "macro_f1": _macro(y_true, pred),
            "recall_severe": round(float((pred[y_true == 2] == 2).mean()), 4),
            "recall_elevated_or_worse": round(float((pred[y_true >= 1] >= 1).mean()), 4),
        })
    return rows


def permutation_importance_blocked(df: pd.DataFrame, seed: int = 0, repeats: int = 3) -> dict:
    """Feature importance measured on a held-out cluster, by permutation.

    Split importances (what the previous pipeline reported) describe how a tree used a column during
    fitting, including columns that only helped it memorise. Permutation importance on unseen ground
    describes what actually carries transferable signal.
    """
    cluster = sorted(df.cluster.unique())[-1]
    train, test = df[df.cluster != cluster], df[df.cluster == cluster]
    ytr = train.label.to_numpy()
    model = make_model().fit(train[MODEL_FEATURES], ytr)
    Xte, yte = test[MODEL_FEATURES].copy(), test.label.to_numpy()

    def macro(X):
        return f1_score(yte, decide(expected_severity(model.predict_proba(X)), ytr),
                        average="macro", zero_division=0)

    base = macro(Xte)
    rng = np.random.default_rng(seed)
    drops = {}
    for col in MODEL_FEATURES:
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
    ap.add_argument("--no-radar", action="store_true",
                    help="ablate the Sentinel-1 features; the control arm for the radar experiment")
    ap.add_argument("--embeddings", default=None,
                    help="parquet of frozen Presto embeddings to concatenate (see training.embed)")
    ap.add_argument("--seed", type=int, default=42,
                    help="model seed; vary it to measure run-to-run variance of a reported gain")
    ap.add_argument("--folds-only", action="store_true",
                    help="blocked + temporal folds only: skip permutation importance, the decision "
                         "sweep and the final fit. For seed sweeps, where only the folds matter.")
    args = ap.parse_args()

    global SEED
    SEED = args.seed

    # Rebinding the module global rather than threading a `features` argument through five call
    # sites. It is set once, before anything reads it, and only from the CLI.
    global MODEL_FEATURES
    if args.no_radar:
        MODEL_FEATURES = [c for c in MODEL_FEATURES if c not in RADAR_FEATURES]

    raw = pd.read_parquet(args.data)
    # Both transforms are label-free and derived from columns already in the parquet, so neither
    # needs a dataset rebuild. See `add_cluster_relative` and `add_site_climatology`.
    df = add_site_climatology(add_cluster_relative(raw))

    if args.embeddings:
        # Left join: a sample with no embedding keeps its tabular features and gets NaN for the 128
        # dims, which the boosted trees handle natively. Dropping those rows would change the test
        # set and make the comparison against the no-embedding run invalid.
        emb = pd.read_parquet(args.embeddings)
        before = len(df)
        df = df.merge(emb, on=["site_id", "obs_date"], how="left")
        emb_cols = [c for c in emb.columns if c.startswith("presto_")]
        covered = df[emb_cols[0]].notna().mean() if emb_cols else 0.0
        assert len(df) == before, "embedding merge must not duplicate rows"
        MODEL_FEATURES = MODEL_FEATURES + emb_cols
        print(f"merged {len(emb_cols)} Presto dims, {covered * 100:.1f}% of samples covered")
    print(f"{len(df)} samples, {df.site_id.nunique()} sites, {df.cluster.nunique()} clusters, "
          f"{df.obs_date.min()}..{df.obs_date.max()}")
    print(f"{len(MODEL_FEATURES)} model features "
          f"({len(CLUSTER_RELATIVE)} cluster-relative, {len(UNINFORMATIVE)} dropped)")
    print("class balance:", df.label.value_counts(normalize=True).sort_index().round(3).to_dict())

    def report(rows):
        for r in rows:
            print(f"  {r['split']:<42} n={r['n']:<5} macroF1={r['macro_f1']:.3f}  "
                  f"balAcc={r['balanced_accuracy']:.3f}  ECE={r['ece']:.3f}  "
                  f"P@25={r['precision_at_25']:.3f}  linF1={r['macro_f1_linear']:.3f}")
            print(f"  {'':<42} baselines F1: majority {r['baseline_majority_f1']:.3f} / "
                  f"persistence {r['baseline_persistence_f1']:.3f} / "
                  f"climatology {r['baseline_climatology_f1']:.3f}   "
                  f"P@25: persistence {r['baseline_persistence_p25']:.3f} / "
                  f"climatology {r['baseline_climatology_p25']:.3f}")

    print("\n=== Spatially blocked (leave-one-cluster-out) ===")
    oof: list = []
    spatial = evaluate_spatial(df, oof)
    report(spatial)

    print("\n=== Forward-chaining temporal ===")
    temporal = evaluate_temporal(df)
    report(temporal)

    if args.folds_only:
        # A seed sweep needs the fold estimates and nothing else. Permutation importance alone
        # refits and re-scores once per feature per repeat, which at 168 features dominates the
        # runtime and answers a question the sweep is not asking.
        print(f"\n(--folds-only, seed {args.seed}: skipping sweep, importance and final fit)")
        return

    print("\n=== Decision-rule sweep (pooled out-of-fold, spatial) ===")
    sweep = sweep_decision_rule(oof)
    for r in sweep:
        print(f"  alert x{r['alert_rate_multiplier']:<4} flagged={r['flagged_share']:.3f}  "
              f"macroF1={r['macro_f1']:.3f}  recall(severe)={r['recall_severe']:.3f}  "
              f"recall(>=elevated)={r['recall_elevated_or_worse']:.3f}")

    print("\n=== Permutation importance on a held-out cluster (top 12) ===")
    importance = permutation_importance_blocked(df)
    for name, drop in list(importance.items())[:12]:
        print(f"  {name:<24} {drop:+.4f}")

    # Final artifact: fitted on everything, since the estimates above already tell us what it is worth.
    print("\nFitting final model on all data ...")
    y_all = df.label.to_numpy()
    final = make_model().fit(df[MODEL_FEATURES], y_all)
    holdout = df[df.cluster == sorted(df.cluster.unique())[-1]]
    held_risk = expected_severity(final.predict_proba(holdout[MODEL_FEATURES]))
    print("(in-sample for the held-out cluster — the blocked numbers above are the honest ones)")
    print(classification_report(holdout.label, decide(held_risk, y_all), zero_division=0, digits=3))

    artifact = artifact_path(MODEL_FEATURES)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact != PRODUCTION_ARTIFACT:
        print(f"\n!! {len(set(MODEL_FEATURES) - SERVABLE_FEATURES)} features are not buildable by "
              f"serving/pipeline.py; writing to {artifact} instead of the production path.")
        print(f"   Not buildable: {sorted(set(MODEL_FEATURES) - SERVABLE_FEATURES)}")
    # Metrics live beside their artifact, or an experimental run silently overwrites the record of
    # what production is actually doing.
    metrics_path = artifact.with_name("metrics.json" if artifact == PRODUCTION_ARTIFACT
                                      else artifact.stem + "_metrics.json")
    version = f"agro-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    joblib.dump({
        "model": final,
        "version": version,
        "feature_columns": MODEL_FEATURES,
        # The serving path reproduces the twins before calling the model. It needs both halves:
        # which cluster a field falls in (`cluster_bounds`) and the (mu, sigma) that cluster's
        # columns were standardised against here (`cluster_stats`). Recomputing sigma from serving
        # traffic instead would feed the model a differently-scaled feature under the same name.
        "cluster_relative": CLUSTER_RELATIVE,
        "cluster_stats": cluster_stats(df),
        "cluster_bounds": {c["name"]: {"lat": list(c["lat"]), "lon": list(c["lon"])}
                           for c in CLUSTERS},
        "classes": [0, 1, 2],
        "label": "peer-standardised NDVI anomaly 30 days ahead",
        "thresholds": {"severe_z": SEVERE_Z, "elevated_z": ELEVATED_Z},
        "n_samples": int(len(df)),
        "trained_on": f"{df.obs_date.min()}..{df.obs_date.max()}",
    }, artifact)

    metrics_path.write_text(json.dumps({
        "spatial_blocked": spatial,
        "temporal_forward": temporal,
        "decision_rule_sweep": sweep,
        "permutation_importance": importance,
        "model_features": MODEL_FEATURES,
        "n_samples": int(len(df)),
        "class_balance": df.label.value_counts(normalize=True).sort_index().round(4).to_dict(),
    }, indent=2, default=str))
    print(f"Saved {artifact} ({version}) and {metrics_path}")


if __name__ == "__main__":
    main()
