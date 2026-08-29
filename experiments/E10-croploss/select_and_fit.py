"""E10 step 2 — feature selection and fit for crop loss, blocked by enumeration area.

SELECTION is done in three stages, in this order, because the cheap ones must run first:

  1. **By rule** -- the leakage screen in `build_dataset.py`. No statistic can rescue a feature that
     is computed from the label.
  2. **By coverage** -- a feature present for under half the rows is dropped. Imputing it would mean
     the model learns the imputation, and `missing` is itself informative here (a plot with no
     fertiliser record usually had no fertiliser).
  3. **By stability, not by a single ranking** -- permutation importance is recomputed on every
     held-out fold, and a feature is kept only if it beats zero on a MAJORITY of folds. A one-shot
     ranking on one split reliably promotes noise at this sample size.

VALIDATION is grouped by EA throughout. Households in one enumeration area share weather, community
infrastructure and often planting calendars; a random row split puts neighbours on both sides of the
boundary and inflates every score. This is the spatial-blocking discipline the rest of the
repository already uses (`docs/CONCEPTS.md`, leave-one-cluster-out).

BASELINES are reported alongside, because an AUC means nothing on its own:
  * prevalence (0.5 by construction),
  * crop identity alone -- if a model only learns "cassava fails more often", that is worth knowing,
  * plot area alone -- section 12 of notebook 08 showed loss rate rises with area.

Run: venv/bin/python3 experiments/E10-croploss/select_and_fit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("experiments/E10-croploss")
SEED = 20260829
MIN_COVERAGE = 0.50


def cv_scores(X, y, groups, model_fn, n_splits=5):
    aucs, aps, briers, oof = [], [], [], np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = model_fn().fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p))
        briers.append(brier_score_loss(y[te], p))
    return np.array(aucs), np.array(aps), np.array(briers), oof


def main() -> None:
    d = pd.read_parquet(OUT / "dataset.parquet")
    y = d.crop_loss.to_numpy()
    groups = d.ea.to_numpy()
    print(f"rows {len(d)}  loss rate {y.mean():.1%}  EAs {len(np.unique(groups))}  "
          f"households {d.hhid.nunique()}")

    # ---- stage 2: coverage --------------------------------------------------------------------
    numeric = ["SR_hect", "gps_area_m2", "rent_paid", "plot_value_ngn", "plant_month",
               "has_extension_access", "asset_score", "market_access_score", "crop_diversity",
               "irrigated", "used_fertilizer", "animal_traction", "n_crops_on_plot"]
    cov = d[numeric].notna().mean()
    kept = [c for c in numeric if cov[c] >= MIN_COVERAGE]
    print("\nstage 2 -- coverage screen:")
    for c in numeric:
        print(f"  {c:22s} coverage {cov[c]:6.1%}  {'keep' if c in kept else 'DROP'}")

    # crop identity as a small set of one-hot columns for the majors only; a 37-level one-hot on
    # 9,376 rows would spend most of its capacity on crops with a handful of observations.
    majors = d.crop.value_counts().head(6).index.tolist()
    X = d[kept].astype(float).copy()
    for c in majors:
        X[f"crop_{c.lower().replace(' ', '_')}"] = (d.crop == c).astype(float)
    print(f"\nmajor crops one-hot: {majors}")

    def gb():
        return HistGradientBoostingClassifier(random_state=SEED, max_depth=3,
                                              learning_rate=0.06, max_iter=250,
                                              l2_regularization=1.0)

    # ---- stage 3: stability selection ---------------------------------------------------------
    print("\nstage 3 -- permutation importance on every held-out fold (stability, not one ranking):")
    per_fold = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = gb().fit(X.iloc[tr], y[tr])
        r = permutation_importance(m, X.iloc[te], y[te], n_repeats=8,
                                   random_state=SEED, scoring="roc_auc")
        per_fold.append(r.importances_mean)
    imp = pd.DataFrame(per_fold, columns=X.columns)
    stab = pd.DataFrame({"mean_importance": imp.mean(), "folds_positive": (imp > 0).sum()})
    stab = stab.sort_values("mean_importance", ascending=False)
    print(stab.to_string())
    selected = stab[(stab.folds_positive >= 3) & (stab.mean_importance > 0)].index.tolist()
    print(f"\nselected ({len(selected)} of {X.shape[1]}): {selected}")

    # ---- fit and compare against baselines ----------------------------------------------------
    results = {}
    def report(name, cols, model_fn):
        auc, ap, br, oof = cv_scores(X[cols], y, groups, model_fn)
        results[name] = {"features": cols, "auc": float(auc.mean()), "auc_sd": float(auc.std()),
                         "pr_auc": float(ap.mean()), "brier": float(br.mean())}
        print(f"  {name:26s} AUC {auc.mean():.3f} +/-{auc.std():.3f}   PR-AUC {ap.mean():.3f}   "
              f"Brier {br.mean():.4f}   ({len(cols)} features)")
        return oof

    print(f"\nGroupKFold(5) by EA. Prevalence baseline: PR-AUC {y.mean():.3f}, AUC 0.500")
    crop_cols = [c for c in X.columns if c.startswith("crop_")]
    report("crop identity only", crop_cols, gb)
    report("plot area only", ["SR_hect"], gb)
    report("all candidates", list(X.columns), gb)
    oof = report("SELECTED", selected, gb)
    # A linear comparator, to show how much of the signal is non-linear. It needs imputation --
    # HistGB handles NaN natively and the linear model cannot, so the median-impute is part of the
    # comparator, not of the selected model.
    report("SELECTED, logistic", selected,
           lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(max_iter=2000)))

    # calibration of the selected model
    q = pd.qcut(pd.Series(oof), 5, duplicates="drop")
    cal = pd.DataFrame({"pred": oof, "actual": y}).groupby(q, observed=True).agg(
        predicted=("pred", "mean"), observed=("actual", "mean"), n=("actual", "size"))
    print("\ncalibration of the selected model (out-of-fold):")
    print(cal.round(3).to_string())

    (OUT / "model_report.json").write_text(json.dumps(
        {"seed": SEED, "n_rows": int(len(d)), "loss_rate": float(y.mean()),
         "n_ea": int(len(np.unique(groups))), "coverage_screen": cov.round(4).to_dict(),
         "stability": stab.round(5).to_dict(orient="index"),
         "selected": selected, "results": results,
         "calibration": cal.round(4).to_dict(orient="records")}, indent=1))
    print(f"\nwrote {OUT}/model_report.json")


if __name__ == "__main__":
    main()
