"""E06 — does static soil explain the field effect?

Joins SoilGrids v2.0 covariates (fetched by fetch_soilgrids.py into
`.cache/soilgrids/{site_id}-{property}.json`) onto each site's field effect
(`alpha_hat`, min_history=5, averaged over its own rows — identical to
notebooks/03-the-field-effect.ipynb), regresses with cluster fixed effects (within-cluster
demeaning, since the effect is entirely within-cluster per E06's design doc), and reports:

  - coverage: how many of 122 sites got a complete soil row, and whether gaps are cluster-structured
  - per-covariate within-cluster r^2 (same metric the elevation/lat/lon null was measured with)
  - the joint within-cluster R^2 (Ridge, matching lab.arms.arms's own regularised-linear convention,
    since 14 soil covariates are collinear by construction — same property at two adjacent depths)
  - a site-level bootstrap interval on that joint R^2 (resample sites, never rows)

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


def joint_within_r2(df: pd.DataFrame, ycol: str, xcols: list[str]) -> float:
    """Cluster-fixed-effects R^2 for the full covariate set: demean y and every x by their own
    cluster mean (FWL), fit a regularised linear model (Ridge, matching lab.arms.arms's `linear`
    arm) on the demeaned data, and report 1 - SS_res/SS_tot on the demeaned target. Ridge, not raw
    OLS, because the 14 covariates are collinear by construction (each property at two adjacent
    depths); the codebase's own linear arm already reaches for Ridge for exactly this reason."""
    y_c = (df.groupby("cluster")[ycol].transform(lambda s: s - s.mean())).to_numpy()
    X_c = df.groupby("cluster")[xcols].transform(lambda s: s - s.mean()).to_numpy()
    model = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(X_c, y_c)
    pred = model.predict(X_c)
    ss_res = float(((y_c - pred) ** 2).sum())
    ss_tot = float(((y_c - y_c.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


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

    r2_point = joint_within_r2(merged, "alpha", xcols)
    print(f"\njoint within-cluster R^2 (Ridge, all {len(xcols)} soil covariates): {r2_point:.4f}")

    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOT):
        sample = merged.sample(n=len(merged), replace=True, random_state=rng.integers(2**32 - 1))
        # a replicate that draws only one cluster (or leaves a cluster with n=1) makes cluster
        # demeaning degenerate for that cluster; skip such draws rather than let them distort the
        # interval — the four clusters are large enough (>=25 sites each) that this is rare.
        if sample["cluster"].nunique() < merged["cluster"].nunique():
            continue
        boot.append(joint_within_r2(sample, "alpha", xcols))
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
        "joint_r2_point": r2_point,
        "joint_r2_bootstrap_ci": [float(lo), float(hi)],
        "n_boot_requested": N_BOOT,
        "n_boot_used": len(boot),
        "null_r2": {"elevation": 0.017, "latitude": 0.020, "longitude": 0.034},
    }
    out = Path(__file__).parent / "e06_result.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\njoint within-cluster R^2: {r2_point:.4f}   95% bootstrap CI (site-level, "
          f"n_boot={len(boot)}): [{lo:.4f}, {hi:.4f}]")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
