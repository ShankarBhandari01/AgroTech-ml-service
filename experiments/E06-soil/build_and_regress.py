"""E06 — does static soil explain the field effect?

Joins SoilGrids v2.0 covariates (fetched by fetch_soilgrids.py into
`.cache/soilgrids/{site_id}-{property}.json`) onto each site's field effect
(`alpha_hat`, min_history=5, averaged over its own rows — identical to
notebooks/03-the-field-effect.ipynb), regresses with cluster fixed effects (within-cluster
demeaning, since the effect is entirely within-cluster per E06's design doc), and reports:

  - coverage: how many of 122 sites got a complete soil row, and whether gaps are cluster-structured
  - per-covariate within-cluster r^2 (same metric the elevation/lat/lon null was measured with)
  - the joint within-cluster R^2 (Ridge, matching lab.arms.arms's own regularised-linear convention,
    since 14 soil covariates are collinear by construction — same property at two adjacent depths),
    scored OUT OF FOLD over sites with the cluster demeaning fitted inside each fold
  - a site-level bootstrap interval on that out-of-fold R^2 (resample sites, never rows)
  - a within-cluster permutation null, which is what makes the in-sample number interpretable:
    14 collinear covariates on ~119 sites return R^2 ~= 0.09 in-sample on pure noise

Run: venv/bin/python3 experiments/E06-soil/build_and_regress.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from argotech.lab.estimand.targets import alpha_hat  # noqa: E402
from argotech.lab.panel.panel import manifest as panel_manifest  # noqa: E402

CACHE_DIR = REPO / ".cache" / "soilgrids"
PROPERTIES = ["clay", "sand", "silt", "soc", "phh2o", "cec", "bdod"]
DEPTHS = ["0-5cm", "5-15cm"]
SEED = 20260829
N_BOOT = 2000


def load_soil_row(site_id: str) -> dict | None:
    """One site's soil covariates, unit-converted via the API's own unit_measure/d_factor.
    Returns None if any of the 7 properties is missing for this site — a hole, not a zero."""
    row = {}
    for prop in PROPERTIES:
        path = CACHE_DIR / f"{site_id}-{prop}.json"
        if not path.exists():
            return None
        layer = json.loads(path.read_text())
        d_factor = layer["unit_measure"]["d_factor"]
        by_depth = {d["label"]: d["values"].get("mean") for d in layer["depths"]}
        for depth in DEPTHS:
            v = by_depth.get(depth)
            if v is None:
                return None
            row[f"{prop}_{depth}"] = v / d_factor
    return row


def within_cluster_r2(df: pd.DataFrame, ycol: str, xcol: str) -> tuple[float, float]:
    """Bivariate within-cluster r/r^2 — identical construction to the elevation/lat/lon null in
    notebooks/03-the-field-effect.ipynb."""
    y_c = df.groupby("cluster")[ycol].transform(lambda s: s - s.mean())
    x_c = df.groupby("cluster")[xcol].transform(lambda s: s - s.mean())
    r = np.corrcoef(y_c, x_c)[0, 1]
    return r, r ** 2


def _fit_pipe():
    """Ridge, matching lab.arms.arms's `linear` arm: the 14 covariates are collinear by
    construction (each property at two adjacent depths), so raw OLS is not an option."""
    return make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), Ridge(alpha=1.0))


def joint_within_r2_insample(df: pd.DataFrame, ycol: str, xcols: list[str]) -> float:
    """In-sample cluster-FE R^2. Reported ONLY as a diagnostic against the permutation null --
    never as the headline. With 14 collinear covariates on ~119 sites it is inflated by roughly
    0.09 even when the covariates carry no signal at all (measured; see `null_permutation` in the
    result file). Bootstrapping it does not remove the optimism, it only describes the sampling
    variability of an optimistic estimator."""
    y_c = (df.groupby("cluster")[ycol].transform(lambda s: s - s.mean())).to_numpy()
    X_c = df.groupby("cluster")[xcols].transform(lambda s: s - s.mean()).to_numpy()
    model = _fit_pipe().fit(X_c, y_c)
    pred = model.predict(X_c)
    ss_res = float(((y_c - pred) ** 2).sum())
    ss_tot = float(((y_c - y_c.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def joint_within_r2_cv(df: pd.DataFrame, ycol: str, xcols: list[str],
                       n_splits: int = 10, n_repeats: int = 5, seed: int = SEED) -> float:
    """Out-of-fold cluster-FE R^2 -- the headline figure.

    Two things this does that the in-sample version cannot:

    1. **Scores on held-out sites.** Folds are over sites, so no site contributes to the model that
       predicts it.
    2. **Demeans within the fold.** The cluster means used to demean `y` and `X` are computed on the
       TRAINING sites only and then applied to the held-out ones. Demeaning on the full frame first
       would let each test site's own value enter the mean that centres it -- the same
       fit-the-reference-on-everything defect the peer-standardisation leak was, in a new place.
       Small here (~30 sites per cluster), but it is the identical error and costs nothing to avoid.

    Averaged over `n_repeats` shuffles because a single 10-fold split has a standard deviation of
    about 0.024 on this data -- large relative to the effect being measured.
    """
    y_raw = df[ycol].to_numpy(float)
    X_raw = df[xcols].to_numpy(float)
    cl = df["cluster"].to_numpy()
    obs, prd = [], []
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for tr, te in kf.split(X_raw):
            mu_y, mu_X = {}, {}
            for c in np.unique(cl[tr]):
                sel = tr[cl[tr] == c]
                mu_y[c] = y_raw[sel].mean()
                mu_X[c] = np.nanmean(X_raw[sel], axis=0)
            # a held-out site whose cluster is absent from the training fold cannot be centred
            keep = np.array([c in mu_y for c in cl[te]])
            if not keep.any():
                continue
            y_tr = y_raw[tr] - np.array([mu_y[c] for c in cl[tr]])
            X_tr = X_raw[tr] - np.vstack([mu_X[c] for c in cl[tr]])
            model = _fit_pipe().fit(X_tr, y_tr)
            te_k = te[keep]
            y_te = y_raw[te_k] - np.array([mu_y[c] for c in cl[te_k]])
            X_te = X_raw[te_k] - np.vstack([mu_X[c] for c in cl[te_k]])
            obs.append(y_te)
            prd.append(model.predict(X_te))
    y_all = np.concatenate(obs)
    p_all = np.concatenate(prd)
    ss_res = float(((y_all - p_all) ** 2).sum())
    ss_tot = float(((y_all - y_all.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def permutation_null(df: pd.DataFrame, ycol: str, xcols: list[str], n: int = 300,
                     seed: int = SEED) -> dict:
    """Shuffle the field effect WITHIN cluster and recompute both statistics. This is what makes the
    in-sample number interpretable: it says what R^2 the same fit returns when the covariates carry
    no information about the target at all."""
    rng = np.random.default_rng(seed)
    cl = df["cluster"].to_numpy()
    ins, cvs = [], []
    for i in range(n):
        d = df.copy()
        y = d[ycol].to_numpy(float).copy()
        for c in np.unique(cl):
            idx = np.where(cl == c)[0]
            y[idx] = rng.permutation(y[idx])
        d[ycol] = y
        ins.append(joint_within_r2_insample(d, ycol, xcols))
        if i < 60:                      # CV is the expensive one; 60 draws bound the null adequately
            cvs.append(joint_within_r2_cv(d, ycol, xcols, n_repeats=1, seed=seed + i))
    return {"n_permutations": n, "n_permutations_cv": len(cvs),
            "insample_mean": float(np.mean(ins)), "insample_p95": float(np.percentile(ins, 95)),
            "cv_mean": float(np.mean(cvs)), "cv_p95": float(np.percentile(cvs, 95)),
            "cv_values": [float(v) for v in cvs]}


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    panel = pd.read_parquet(REPO / "data" / "training_set.parquet")
    a = alpha_hat(panel, column="ndvi_z_peer", unit="site_id", min_history=5)
    site_effect = (panel.assign(alpha=a).dropna(subset=["alpha"])
                   .groupby(["site_id", "cluster"]).alpha.mean().reset_index())
    site = (panel[["site_id", "latitude", "longitude", "elevation"]].drop_duplicates("site_id")
            .merge(site_effect, on="site_id"))
    assert len(site) == 122, f"expected 122 sites with an estimable field effect, got {len(site)}"

    soil_rows = {sid: load_soil_row(sid) for sid in site["site_id"]}
    covered = {sid: row for sid, row in soil_rows.items() if row is not None}

    coverage_by_cluster = (site.assign(has_soil=site["site_id"].isin(covered))
                            .groupby("cluster")["has_soil"].agg(["sum", "count"]))

    soil_df = pd.DataFrame.from_dict(covered, orient="index")
    xcols = list(soil_df.columns)
    merged = site.merge(soil_df, left_on="site_id", right_index=True, how="inner")

    print(f"coverage: {len(merged)}/122 sites have a complete soil row "
          f"({len(PROPERTIES)} properties x {len(DEPTHS)} depths each)")
    print("coverage by cluster:")
    print(coverage_by_cluster.to_string())

    print("\nper-covariate within-cluster r^2 (null to beat: elevation 0.017, latitude 0.020, "
          "longitude 0.034):")
    per_cov = []
    for col in xcols:
        r, r2 = within_cluster_r2(merged, "alpha", col)
        per_cov.append((col, r, r2))
    per_cov.sort(key=lambda t: -t[2])
    for col, r, r2 in per_cov:
        print(f"  {col:16s}: r = {r:+.3f}   r^2 = {r2:.3f}")

    r2_cv = joint_within_r2_cv(merged, "alpha", xcols)
    r2_in = joint_within_r2_insample(merged, "alpha", xcols)
    print(f"\njoint within-cluster R^2 (Ridge, all {len(xcols)} soil covariates)")
    print(f"  out-of-fold (10-fold x 5, fold-wise demeaning): {r2_cv:.4f}   <-- headline")
    print(f"  in-sample (diagnostic only, inflated):          {r2_in:.4f}")

    print("\npermutation null (field effect shuffled within cluster) ...")
    null = permutation_null(merged, "alpha", xcols)
    print(f"  in-sample under the null: mean {null['insample_mean']:.4f}  95th pct {null['insample_p95']:.4f}")
    print(f"  out-of-fold under the null: mean {null['cv_mean']:.4f}  95th pct {null['cv_p95']:.4f}")
    p_val = float(np.mean(np.array(null["cv_values"]) >= r2_cv))
    print(f"  permutation p-value (out-of-fold): {p_val:.3f}")

    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOT):
        sample = merged.sample(n=len(merged), replace=True, random_state=rng.integers(2**32 - 1))
        # a replicate that draws only one cluster (or leaves a cluster with n=1) makes cluster
        # demeaning degenerate for that cluster; skip such draws rather than let them distort the
        # interval — the four clusters are large enough (>=25 sites each) that this is rare.
        if sample["cluster"].nunique() < merged["cluster"].nunique():
            continue
        # bootstrap the CV estimate, not the in-sample one: an interval around an
        # optimistic point estimate is still centred on the optimism.
        boot.append(joint_within_r2_cv(sample, "alpha", xcols, n_repeats=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    result = {
        "panel_manifest": panel_manifest(panel),
        "git_sha": git_sha(),
        "seed": SEED,
        "n_sites_total": 122,
        "n_sites_with_soil": len(merged),
        "coverage_by_cluster": coverage_by_cluster.reset_index().to_dict(orient="records"),
        "properties": PROPERTIES,
        "depths": DEPTHS,
        "per_covariate_r2": [{"covariate": c, "r": r, "r2": r2} for c, r, r2 in per_cov],
        "joint_r2_cv": r2_cv,
        "joint_r2_insample_diagnostic": r2_in,
        "joint_r2_cv_bootstrap_ci": [float(lo), float(hi)],
        "permutation_p_value_cv": p_val,
        "null_permutation": null,
        "n_boot_requested": N_BOOT,
        "n_boot_used": len(boot),
        "null_r2": {"elevation": 0.017, "latitude": 0.020, "longitude": 0.034},
    }
    out = Path(__file__).parent / "e06_result.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\njoint within-cluster R^2 (out-of-fold): {r2_cv:.4f}   95% bootstrap CI (site-level, "
          f"n_boot={len(boot)}): [{lo:.4f}, {hi:.4f}]")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
