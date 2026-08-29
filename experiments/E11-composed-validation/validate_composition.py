"""E11 - score the COMPOSED risk equation against a realised agronomic outcome.

Every term in `domain.risk` has been validated separately. The composition never has. This joins
E07's composed risk (per household, GHS-Panel Wave 5) to E10's realised crop-loss label (per
crop-plot, same survey, same wave) on `hhid` and asks the question the product actually turns on:

    does ranking farmers by `expected_loss` find the ones who lose crop?

WHAT THIS CAN AND CANNOT VALIDATE
---------------------------------
E07's `hazard` column is DRAWN from the panel's distribution -- Wave 5 ships no coordinates, so no
household has a real hazard. That is not a flaw to apologise for here, it is the design:

  * it makes this a test of the EXPOSURE x VULNERABILITY composition, which is 77.4% of
    var(log expected_loss) and the part actually under dispute (docs/CONCEPTS.md Rule 3);
  * it hands us a free NEGATIVE CONTROL. `hazard` alone must score at chance against `crop_loss`.
    If it does not, the harness is wrong and nothing else in this file may be believed.

It cannot validate the hazard model. Nothing in this repository can, until outcome capture runs.

UNIT OF ANALYSIS
----------------
The household. A visit goes to a farmer, not to a plot, and E07 composes per household. E10's
per-crop-plot label is aggregated up: `any_loss` (did this household lose crop on any plot) is the
event a visit would have caught. `share_loss` is carried as a continuous sensitivity.

Bootstrap resamples ENUMERATION AREAS, never households -- the same rule this repo already applies
to sites ("39 monthly observations of one site are not 39 independent facts"). `ea == 0` is an
unknown-EA code present in both files; it is kept as a single group, which makes the interval wider
rather than narrower.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from argotech.domain.risk import MAX_LOSS_FRACTION
from argotech.lab.eval.evaluate import net_benefit, precision_at_k, spearman

SEED = 20260829
ROOT = Path(__file__).resolve().parents[2]
E07 = ROOT / "experiments/E07-lsms/composed_risk.parquet"
E10 = ROOT / "experiments/E10-croploss/dataset.parquet"
OUT = Path(__file__).resolve().parent / "e11_result.json"

# An extension officer visits a few farms a week. These are the k the product actually operates at.
KS = (50, 100, 200)
THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50)

# Survey yield / FAO yield on sole-cropped plots, from experiments/E07-lsms/FINDINGS.md. Correcting
# a survey yield toward FAO means dividing by these. Cassava's 0.21x is not measurement noise -- it
# is harvested piecemeal over months, so a post-harvest visit captures harvest-to-date. Crops with
# no published comparison are left uncorrected rather than guessed.
FAO_RATIO = {"MAIZE": 0.44, "CASSAVA": 0.21, "RICE": 0.87, "GUINEA CORN (SORGHUM)": 0.84}

# GPS-measured plot area runs at 0.88x self-reported, spearman +0.549 (n=5,137). The CONSTANT is
# irrelevant to a ranking; the disagreement is what moves ranks, so scenario B calibrates
# multiplicative noise to reproduce that rank correlation rather than applying the 0.88.
AREA_RANK_CORR = 0.549


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def load() -> pd.DataFrame:
    """One row per household: the composed terms, the realised label, and the crop mix."""
    comp = pd.read_parquet(E07)
    comp.index = comp.index.astype(str)
    plots = pd.read_parquet(E10)
    plots["hhid"] = plots["hhid"].astype(str)

    hh = plots.groupby("hhid").agg(
        any_loss=("crop_loss", "max"),
        share_loss=("crop_loss", "mean"),
        n_crop_rows_e10=("crop_loss", "size"),
        ea_e10=("ea", "first"),
    )
    df = comp.join(hh, how="inner")

    # The crop-mix weighted FAO correction factor, per household.
    plots["ratio"] = plots["crop"].map(FAO_RATIO).fillna(1.0)
    df["fao_factor"] = 1.0 / plots.groupby("hhid")["ratio"].mean().reindex(df.index).fillna(1.0)

    # `ea` appears in both files. They must agree, or the two were built off different frames.
    disagree = int((df["ea"].astype(int) != df["ea_e10"].astype(int)).sum())
    if disagree:
        raise SystemExit(f"E07 and E10 disagree on `ea` for {disagree} households; join is unsafe")
    return df


def rankings(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every candidate queue order, including the controls that stop a headline being overstated."""
    loss_rate = df["hazard"] * (0.5 + 0.5 * df["vuln_score"]) * MAX_LOSS_FRACTION
    return {
        # The production queue.
        "expected_loss": df["expected_loss"].to_numpy(float),
        # Exposure-free: the loss RATE a farmer sees. docs/model-design.md 9.4's first alternative.
        "risk_score": df["risk_score"].to_numpy(float),
        # Its second alternative: keep exposure, stop letting it dominate by 227x.
        "expected_loss_log_exposure": (np.log1p(df["value_at_risk_usd"]) * loss_rate).to_numpy(float),
        # Single terms, to see which one the composite is actually tracking.
        "exposure_only": df["value_at_risk_usd"].to_numpy(float),
        "vulnerability_only": df["vuln_score"].to_numpy(float),
        # NEGATIVE CONTROL. Drawn, so it must land at chance. If it does not, stop reading.
        "hazard_only__NEGATIVE_CONTROL": df["hazard"].to_numpy(float),
    }


def ea_bootstrap_auc(y, score, groups, n=2000, seed=SEED) -> tuple[float, float]:
    """Percentile CI on AUC, resampling ENUMERATION AREAS with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_ea = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_ea[g] for g in drawn])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        out.append(roc_auc_score(yy, score[idx]))
    if len(out) < 2:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return (float(lo), float(hi))


def paired_delta_auc(y, a, b, groups, n=2000, seed=SEED) -> dict:
    """AUC(a) - AUC(b) on the SAME resampled EAs, so the two arms share their noise.

    Comparing two independent intervals is the wrong test: both are wide because EAs vary, and that
    variance is common to both arms. Pairing removes it, which is the only way to say whether one
    ranking is actually better than another rather than merely scoring higher once.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_ea = {g: np.flatnonzero(groups == g) for g in uniq}
    deltas = []
    for _ in range(n):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_ea[g] for g in drawn])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        deltas.append(roc_auc_score(yy, a[idx]) - roc_auc_score(yy, b[idx]))
    d = np.asarray(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta_auc": round(float(d.mean()), 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)],
            "p_positive": round(float((d > 0).mean()), 4)}


def tie_diagnostics(df: pd.DataFrame) -> dict:
    """Can each candidate queue key even ORDER the top 50?

    docs/model-design.md 9.3a already learned this on `hazard.combined` -- rounding to 3dp collapsed
    140 distinct top-half values to 8, so it is stored at 6dp. `risk_score` is `round(score, 1)` and
    is the field a queue would actually sort on. The same lesson was never applied to it.
    """
    out = {}
    for col in ("expected_loss", "risk_score", "vuln_score", "hazard", "value_at_risk_usd"):
        s = df[col].to_numpy(float)
        order = np.argsort(-s, kind="stable")
        out[col] = {"distinct": len(np.unique(s)),
                    "of_n": len(s),
                    "tied_at_k50": bool(np.isclose(s[order[49]], s[order[50]]))}
    return out


def oof_probability(score: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Platt-scale a score into a probability, out of fold by EA.

    A decision curve needs a probability and these rankings are not probabilities of THIS label.
    Fitting the scaling in-sample would let the curve report its own training fit, so it is fitted
    out of fold on the same EA blocks everything else here uses. The normal-CDF bridge in
    `evaluate.prob_event` is the equivalent move for a continuous target; this is the binary one,
    and like that one it is an assumption stated rather than buried.
    """
    x = np.log1p(np.clip(score, 0, None)).reshape(-1, 1)
    prob = np.zeros(len(y), dtype=float)
    for tr, te in GroupKFold(n_splits=5).split(x, y, groups):
        model = LogisticRegression(max_iter=2000).fit(x[tr], y[tr])
        prob[te] = model.predict_proba(x[te])[:, 1]
    return prob


def fitted_ceiling(df: pd.DataFrame) -> np.ndarray:
    """E10's model, out of fold, aggregated to the household - the ceiling the equation is judged against.

    Fitted at plot-crop level exactly as E10 did (GroupKFold(5) by EA, same estimator), then taken
    to the household by MAX: the household is at risk if any of its plots is. That matches
    `any_loss`, which is the event a visit catches.
    """
    plots = pd.read_parquet(E10)
    plots["hhid"] = plots["hhid"].astype(str)
    plots = plots[plots["hhid"].isin(df.index)].copy()

    feats = json.loads((ROOT / "experiments/E10-croploss/model_report.json").read_text())["selected"]
    crop_dummies = pd.get_dummies(plots["crop"].str.lower().str.replace(" ", "_"), prefix="crop")
    x_all = pd.concat([plots.drop(columns=["crop"]), crop_dummies], axis=1)
    usable = [c for c in feats if c in x_all.columns]
    x = x_all[usable].astype(float).to_numpy()
    y = plots["crop_loss"].to_numpy(int)
    groups = plots["ea"].to_numpy()

    prob = np.zeros(len(y), dtype=float)
    for tr, te in GroupKFold(n_splits=5).split(x, y, groups):
        model = HistGradientBoostingClassifier(random_state=SEED, max_depth=3,
                                               learning_rate=0.05, max_iter=300)
        model.fit(x[tr], y[tr])
        prob[te] = model.predict_proba(x[te])[:, 1]
    plots["oof"] = prob
    return plots.groupby("hhid")["oof"].max().reindex(df.index).to_numpy(float)


def calibrate_area_noise(exposure: np.ndarray, rng) -> float:
    """Find the lognormal sigma whose perturbation reproduces spearman 0.549 against the original."""
    lo, hi = 0.01, 5.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        perturbed = exposure * rng.lognormal(0.0, mid, size=len(exposure))
        if spearman(perturbed, exposure) > AREA_RANK_CORR:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def confound_check(df: pd.DataFrame) -> dict:
    """`any_loss` rises mechanically with plot count, and so does exposure. Separate them.

    A household with more crop-plot rows has more chances for at least one to lose, and
    `value_at_risk_usd` is area-driven, so it correlates with plot count too (spearman +0.23). Any
    edge exposure shows on `any_loss` could therefore be counting plots rather than finding risk.
    Two ways out, both reported: the continuous `share_loss` label, which is a rate and so cannot be
    inflated by count; and AUC within fixed plot-count strata.
    """
    y = df["any_loss"].to_numpy(int)
    exposure = df["value_at_risk_usd"].to_numpy(float)
    n_plots = df["n_crop_rows_e10"].to_numpy(float)
    share = df["share_loss"].to_numpy(float)

    strata = {}
    for k in (1, 2, 3):
        m = (df["n_crop_rows_e10"] == k).to_numpy()
        if m.sum() > 150 and len(np.unique(y[m])) > 1:
            strata[f"n_plots_{k}"] = {"n": int(m.sum()),
                                      "prevalence": round(float(y[m].mean()), 4),
                                      "auc_exposure": round(float(roc_auc_score(y[m], exposure[m])), 4)}
    return {
        "auc_plot_count_alone": round(float(roc_auc_score(y, n_plots)), 4),
        "auc_exposure_alone": round(float(roc_auc_score(y, exposure)), 4),
        "spearman_exposure_vs_plot_count": round(spearman(exposure, n_plots), 4),
        "spearman_vs_share_loss": {
            "exposure": round(spearman(exposure, share), 4),
            "expected_loss": round(spearman(df["expected_loss"].to_numpy(float), share), 4),
            "plot_count": round(spearman(n_plots, share), 4),
        },
        "auc_exposure_within_plot_count_strata": strata,
    }


def rank_stability(df: pd.DataFrame) -> dict:
    """How much does the top-k move under the input error this repo has already measured?

    For a triage queue this is the robustness question. Being wrong about a dollar figure matters
    far less than the ORDER moving, and every bias below is documented, not invented.
    """
    rng = np.random.default_rng(SEED)
    loss_rate = (df["hazard"] * (0.5 + 0.5 * df["vuln_score"]) * MAX_LOSS_FRACTION).to_numpy(float)
    exposure = df["value_at_risk_usd"].to_numpy(float)
    base = exposure * loss_rate
    sigma = calibrate_area_noise(exposure, np.random.default_rng(SEED))

    def retention(perturbed: np.ndarray) -> dict:
        out = {}
        for k in KS:
            top_a = set(np.argsort(-base, kind="stable")[:k])
            top_b = set(np.argsort(-perturbed, kind="stable")[:k])
            out[f"top{k}_retained"] = round(len(top_a & top_b) / k, 4)
        out["spearman_vs_base"] = round(spearman(perturbed, base), 4)
        return out

    scenarios = {
        # Systematic: correct each household's yield toward FAO using its own crop mix.
        "A_yield_corrected_to_fao": retention(exposure * df["fao_factor"].to_numpy(float) * loss_rate),
        # Random: plot-area measurement error at the documented rank agreement.
        "B_area_measurement_error": retention(
            exposure * rng.lognormal(0.0, sigma, size=len(df)) * loss_rate),
    }
    scenarios["C_both"] = retention(
        exposure * df["fao_factor"].to_numpy(float)
        * rng.lognormal(0.0, sigma, size=len(df)) * loss_rate)
    scenarios["area_noise_sigma"] = round(sigma, 4)
    return scenarios


def main() -> None:
    df = load()
    y = df["any_loss"].to_numpy(int)
    share = df["share_loss"].to_numpy(float)
    vuln = df["vuln_score"].to_numpy(float)
    groups = df["ea"].to_numpy()
    prevalence = float(y.mean())

    scores = rankings(df)
    scores["E10_fitted_model__CEILING"] = fitted_ceiling(df)

    rows = {}
    for name, s in scores.items():
        lo, hi = ea_bootstrap_auc(y, s, groups)
        entry = {
            "auc": round(float(roc_auc_score(y, s)), 4),
            "auc_ci_ea_bootstrap": [round(lo, 4), round(hi, 4)],
            "pr_auc": round(float(average_precision_score(y, s)), 4),
            "spearman_vs_share_loss": round(spearman(s, share), 4),
        }
        for k in KS:
            entry[f"p_at_{k}"] = round(precision_at_k(y, s, k), 4)
            top = np.argsort(-s, kind="stable")[:k]
            entry[f"mean_vulnerability_top{k}"] = round(float(vuln[top].mean()), 4)
        prob = oof_probability(s, y, groups)
        entry["decision_curve"] = [
            {"threshold": t,
             "model": round(net_benefit(y, prob, t), 4),
             "visit_all": round(net_benefit(y, np.ones(len(y)), t), 4)}
            for t in THRESHOLDS]
        rows[name] = entry

    contrasts = {
        # Does composing exposure with the other two terms beat exposure on its own?
        "expected_loss - exposure_only": paired_delta_auc(
            y, scores["expected_loss"], scores["exposure_only"], groups),
        # Sanity: the production queue against the drawn control.
        "expected_loss - hazard_only": paired_delta_auc(
            y, scores["expected_loss"], scores["hazard_only__NEGATIVE_CONTROL"], groups),
        # The ranking-target decision in docs/model-design.md 9.4, measured.
        "risk_score - expected_loss": paired_delta_auc(
            y, scores["risk_score"], scores["expected_loss"], groups),
        # How much a fitted model leaves the equation behind.
        "E10_ceiling - expected_loss": paired_delta_auc(
            y, scores["E10_fitted_model__CEILING"], scores["expected_loss"], groups),
    }

    result = {
        "git_sha": git_sha(),
        "seed": SEED,
        "unit": "household",
        "n_households": len(df),
        "n_ea": int(df["ea"].nunique()),
        "prevalence_any_loss": round(prevalence, 4),
        "mean_share_loss": round(float(share.mean()), 4),
        "mean_vulnerability_all": round(float(vuln.mean()), 4),
        "hazard_is_drawn": True,
        "rankings": rows,
        "paired_contrasts": contrasts,
        "tie_diagnostics": tie_diagnostics(df),
        "confound_check": confound_check(df),
        "rank_stability": rank_stability(df),
    }
    OUT.write_text(json.dumps(result, indent=1))

    print(f"n={len(df)} households, {df['ea'].nunique()} EAs, "
          f"prevalence(any_loss)={prevalence:.4f}, mean vulnerability={vuln.mean():.4f}\n")
    hdr = f"{'ranking':32s} {'AUC':>6s} {'95% CI (EA)':>18s} {'PR':>6s} {'P@50':>6s} {'P@200':>6s} {'vuln@50':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for name, e in rows.items():
        lo, hi = e["auc_ci_ea_bootstrap"]
        print(f"{name:32s} {e['auc']:6.3f} [{lo:+.3f}, {hi:+.3f}] {e['pr_auc']:6.3f} "
              f"{e['p_at_50']:6.3f} {e['p_at_200']:6.3f} {e['mean_vulnerability_top50']:8.3f}")
    print(f"\nbase rate (P@k of a random queue) = {prevalence:.3f}; "
          f"vulnerability of the whole population = {vuln.mean():.3f}")

    print("\npaired EA bootstrap (same resample both arms)")
    for name, c in contrasts.items():
        print(f"  {name:32s} dAUC={c['delta_auc']:+.4f} "
              f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]  P(>0)={c['p_positive']:.3f}")

    print("\ncan the key even order a 50-farmer queue?")
    for name, t in result["tie_diagnostics"].items():
        print(f"  {name:22s} {t['distinct']:5d} distinct / {t['of_n']}  "
              f"tied at k=50: {t['tied_at_k50']}")

    cc = result["confound_check"]
    print("\nconfound: `any_loss` rises with plot count, and so does exposure")
    print(f"  AUC plot-count alone {cc['auc_plot_count_alone']:.4f} vs exposure "
          f"{cc['auc_exposure_alone']:.4f}  (spearman between them "
          f"{cc['spearman_exposure_vs_plot_count']:+.4f})")
    print("  on the count-free `share_loss` label, spearman: "
          + ", ".join(f"{k}={v:+.4f}" for k, v in cc["spearman_vs_share_loss"].items()))
    for k, v in cc["auc_exposure_within_plot_count_strata"].items():
        print(f"  {k}: n={v['n']:4d} prev={v['prevalence']:.3f} AUC(exposure)={v['auc_exposure']:.4f}")

    print("\nrank stability of `expected_loss` under documented input error")
    for name, s in result["rank_stability"].items():
        if isinstance(s, dict):
            print(f"  {name:28s} " + "  ".join(f"{k}={v}" for k, v in s.items()))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
