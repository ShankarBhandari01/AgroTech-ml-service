"""E01 - variance decomposition of forward_z.

Question: what share of the target's variance is a time-invariant field effect?
If the field effect dominates, the climatology baseline wins by collecting it directly
and the spec's diagnosis holds. If it does not, the spec is wrong.
"""
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
df = pd.read_parquet("data/training_set.parquet").sort_values(["site_id", "obs_date"])

def icc(y, g):
    """Unbiased one-way random-effects variance components (method of moments).

    Naive eta^2 (between-group SS / total SS) is biased upward when groups are small:
    with n_i observations per group the group mean carries sigma_eps^2 / n_i of pure
    noise, which eta^2 credits to the group. This corrects for that.
    """
    d = pd.DataFrame({"y": y, "g": g}).dropna()
    n = d.groupby("g")["y"].size().to_numpy()
    m = d.groupby("g")["y"].mean().to_numpy()
    N, k = n.sum(), len(n)
    if k < 2 or N <= k:
        return dict(icc=np.nan, s2_between=np.nan, s2_within=np.nan, k=k, N=N)
    gm = d["y"].mean()
    ssb = float((n * (m - gm) ** 2).sum())
    ssw = float(((d["y"] - d.groupby("g")["y"].transform("mean")) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)          # effective group size, unbalanced
    s2b = max(0.0, (msb - msw) / n0)
    return dict(icc=s2b / (s2b + msw) if s2b + msw > 0 else np.nan,
                s2_between=s2b, s2_within=msw, k=k, N=N, eta2_naive=ssb / (ssb + ssw))

print(f"panel: {len(df)} rows, {df.site_id.nunique()} sites, {df.cluster.nunique()} clusters, "
      f"{df.obs_date.min()} .. {df.obs_date.max()}")
print(f"obs/site: median {df.groupby('site_id').size().median():.0f}, "
      f"min {df.groupby('site_id').size().min()}, max {df.groupby('site_id').size().max()}")
print(f"Var(forward_z) = {df.forward_z.var():.4f}\n")

# --- sanity: is the cohort-date effect already removed by construction? -------------
coh = df.groupby(["cluster", "label_date"])["forward_z"].mean()
print(f"[check] mean forward_z within (cluster,label_date) cohort: "
      f"mean {coh.mean():+.4f}, sd {coh.std():.4f}  <- should be ~0 if gamma_t is removed")
print(f"[check] ICC of forward_z on (cluster,label_date): "
      f"{icc(df.forward_z, df.cluster.astype(str) + '|' + df.label_date.astype(str))['icc']:.4f}\n")

# --- the question ------------------------------------------------------------------
print("variance components of forward_z")
print(f"{'grouping':<26}{'ICC':>9}{'naive eta2':>12}{'between':>10}{'within':>9}{'groups':>8}")
for name, g in [("site_id (field effect)", df.site_id),
                ("cluster", df.cluster),
                ("site_id within cluster", df.cluster.astype(str) + "|" + df.site_id.astype(str))]:
    r = icc(df.forward_z, g)
    print(f"{name:<26}{r['icc']:>9.4f}{r['eta2_naive']:>12.4f}"
          f"{r['s2_between']:>10.4f}{r['s2_within']:>9.4f}{r['k']:>8d}")

# bootstrap over sites (the independent unit), not rows
sites = df.site_id.unique()
boot = []
for _ in range(1000):
    pick = rng.choice(sites, len(sites), replace=True)
    s = pd.concat([df[df.site_id == p].assign(site_id=f"{p}#{i}") for i, p in enumerate(pick)])
    boot.append(icc(s.forward_z, s.site_id)["icc"])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\nfield-effect ICC 95% CI over 1000 site bootstraps: [{lo:.4f}, {hi:.4f}]")

# --- what the climatology baseline actually collects --------------------------------
# Replicates train.add_site_climatology exactly.
df["alpha_hat"] = (df.groupby("site_id")["ndvi_z_peer"]
                     .transform(lambda s: s.expanding().mean().shift(1))).fillna(0.0)
d = df.dropna(subset=["forward_z"])
ss_res = float(((d.forward_z - d.alpha_hat) ** 2).sum())
ss_tot = float(((d.forward_z - d.forward_z.mean()) ** 2).sum())
print(f"\nexpanding-window alpha_hat (== the climatology baseline) vs forward_z:")
print(f"  Pearson r          {d.forward_z.corr(d.alpha_hat):+.4f}")
print(f"  Spearman rho       {d.forward_z.corr(d.alpha_hat, method='spearman'):+.4f}")
print(f"  out-of-sample R2   {1 - ss_res / ss_tot:+.4f}   (as a direct predictor, no fitting)")

# and the current level, which is what persistence uses
print(f"\ncurrent level ndvi_z_peer (== the persistence baseline) vs forward_z:")
print(f"  Pearson r          {d.forward_z.corr(d.ndvi_z_peer):+.4f}")
print(f"  field-effect ICC of ndvi_z_peer itself: {icc(df.ndvi_z_peer, df.site_id)['icc']:.4f}")

# --- what the reformulated target leaves behind -------------------------------------
d = d.assign(ztilde=d.forward_z - d.alpha_hat)
print(f"\nreformulated target ztilde = forward_z - alpha_hat:")
print(f"  Var(forward_z) {d.forward_z.var():.4f}  ->  Var(ztilde) {d.ztilde.var():.4f}"
      f"   ({100 * (1 - d.ztilde.var() / d.forward_z.var()):+.1f}% of variance removed)")
print(f"  field-effect ICC {icc(d.forward_z, d.site_id)['icc']:.4f}  ->  "
      f"{icc(d.ztilde, d.site_id)['icc']:.4f}   <- should collapse toward 0")
print(f"  corr(ztilde, alpha_hat) {d.ztilde.corr(d.alpha_hat):+.4f}   <- baseline can no longer win by proxy")
