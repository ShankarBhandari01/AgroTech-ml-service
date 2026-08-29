"""Variance decomposition of forward_z, promoted from the E01 probe to tested library code.

Question: what share of the target's variance is a time-invariant field effect? If the field
effect dominates, the climatology baseline wins by collecting it directly and the spec's diagnosis
holds. If it does not, the spec is wrong. `decompose` reproduces the 34.5% CI [0.236, 0.435] that
the spec cites as its headline number; `experiments/E01_variance_decomposition.out` is the
committed record of that run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def icc(y, g) -> dict:
    """Unbiased one-way random-effects variance components (method of moments).

    Naive eta^2 (between-group SS / total SS) is biased upward when groups are small:
    with n_i observations per group the group mean carries sigma_eps^2 / n_i of pure
    noise, which eta^2 credits to the group. This corrects for that.
    """
    d = pd.DataFrame({"y": y, "g": g}).dropna()
    n = d.groupby("g")["y"].size().to_numpy()
    m = d.groupby("g")["y"].mean().to_numpy()
    total_n, k = n.sum(), len(n)
    if k < 2 or total_n <= k:
        return dict(icc=np.nan, s2_between=np.nan, s2_within=np.nan, k=k, n=total_n)
    gm = d["y"].mean()
    ssb = float((n * (m - gm) ** 2).sum())
    ssw = float(((d["y"] - d.groupby("g")["y"].transform("mean")) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (total_n - k)
    n0 = (total_n - (n ** 2).sum() / total_n) / (k - 1)          # effective group size, unbalanced
    s2b = max(0.0, (msb - msw) / n0)
    return dict(icc=s2b / (s2b + msw) if s2b + msw > 0 else np.nan,
                s2_between=s2b, s2_within=msw, k=k, n=total_n, eta2_naive=ssb / (ssb + ssw))


def decompose(df: pd.DataFrame, target: str = "forward_z", seed: int = 0,
              n_boot: int = 1000) -> dict:
    """Variance components of `target` by site, cluster, and cluster|label_date cohort.

    The site-level ICC's confidence interval is bootstrapped over sites, not rows: the site is the
    independent unit, and resampling rows would treat 39 observations of one field as 39
    independent facts.

    Sorted by site_id first so the resample is reproducible from a seed regardless of input row
    order — E01's committed run sorted the frame before computing, and the bootstrap draw sequence
    depends on the order `site_id.unique()` returns.
    """
    df = df.sort_values("site_id")
    rng = np.random.default_rng(seed)
    sites = df["site_id"].unique()
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(sites, len(sites), replace=True)
        s = pd.concat([df[df["site_id"] == p].assign(site_id=f"{p}#{i}")
                        for i, p in enumerate(pick)])
        boot.append(icc(s[target], s["site_id"])["icc"])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"var_total": float(df[target].var()),
            "site": icc(df[target], df["site_id"]),
            "cluster": icc(df[target], df["cluster"]),
            "cohort": icc(df[target], df["cluster"].astype(str) + "|" + df["label_date"].astype(str)),
            "icc_ci": (float(lo), float(hi))}
