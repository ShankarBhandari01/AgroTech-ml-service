# E03 — replicating the +0.0404 result

Spatial / `geo_month` / `within_xy` / `linear` at **+0.0404** (`experiments/E02/E02-within_xy-geo_month.result.json`,
`git_sha=dd59159`, `seed=42`) is the strongest result in this investigation. It rests on one run
over four leave-one-cluster-out folds. This directory replicates it — correctly, not by seed sweep.

## Step 1 — the four per-fold values, explicit

Pooling four folds into one mean and reporting it without its spread is the exact mistake
`docs/model-design.md` §9.1 G already records once (a precision@25 whose seed spread was twice the
reported effect). The fold-to-fold spread here, with four independent spatial units, IS the
uncertainty:

| held-out cluster | `linear` NB | `boosted` NB | n | event_rate | peer_coverage |
| --- | --- | --- | --- | --- | --- |
| Benue_River_Basin | +0.1268 | +0.0738 | 566 | 0.488 | 1.000 |
| Kaduna_Grain_Belt | -0.0103 | -0.0046 | 1043 | 0.079 | 1.000 |
| Kano_Sudan_Savannah | -0.0042 | -0.0056 | 1257 | 0.049 | 1.000 |
| Kenya_Rift_Valley | +0.0491 | +0.0234 | 748 | 0.287 | 0.472 |
| **mean (= reported figure)** | **+0.0404** | **+0.0218** | | | |

Read plainly: two of four folds are *negative* for `linear` (Kaduna, Kano), and the positive mean
is carried almost entirely by one fold, Benue, at +0.1268 — roughly 3x the reported headline and
about 12x the magnitude of either negative fold. The fold-to-fold spread (-0.0103 to +0.1268,
range 0.137) is more than three times the reported effect itself (+0.0404). This is exactly the
shape of number a fold mean alone hides and a per-fold table exposes.

## Step 2 — no seed sweep is possible, and why

**A seed sweep was not attempted, deliberately, because it cannot produce what a seed sweep is
for.** Measured (by the coordinator, prior to this task, and consistent with what this arm's
construction predicts): `max |pred(seed) - pred(seed=42)|` across seeds `{42, 7, 2024, 1, 99}` is
exactly `0.000e+00` for **both** the linear and boosted arms on this cell.

- `linear` (`argotech.lab.arms`) is a `Ridge` regression: a closed-form / deterministic convex
  solve. It has no stochastic component at all, at any seed. This is expected, not a bug.
- `boosted` is `HistGradientBoostingRegressor` with `early_stopping=False`, set deliberately
  (see `arms.py` / E01-era review notes) to stop sklearn's IID internal validation split from
  leaking across the spatial blocking this protocol depends on. That removes the arm's only
  stochastic component: with `early_stopping=False`, no internal train/validation resplit happens,
  and `HistGradientBoostingRegressor` does not subsample rows for histogram binning at this
  dataset's size (n ~ 3-4k per fold; subsampling only activates well above that). Nothing left in
  the fit path reads the seed.

Five seeds on this cell would therefore return five *identical* numbers for both arms. Reporting
that as "replicated across five seeds" would imply five independent draws where there is exactly
one — a false replication, not a conservative one. This is why the plan for this task explicitly
rules a seed sweep out rather than treating it as optional.

## Step 3 — the actual replication: a site-level bootstrap

Per `docs/superpowers/plans/2026-08-26-close-out-and-retire-incumbent.md` Task E and this
investigation's own `src/argotech/lab/variance.py` (which already bootstraps sites the same way
to interval the site ICC): resample **sites**, never rows — the panel has 39 monthly observations
per site on average, and treating those as 39 independent facts is exactly the mistake
`variance.decompose`'s docstring already warns against.

`bootstrap_sites.py` (this directory): for each of `N_BOOT=200` replicates, draws `len(sites)`
site IDs with replacement from the full 122-site pool, relabels each draw `"<site_id>#<i>"` so
`alpha_hat` and leave-one-cluster-out treat every draw as an independent pseudo-site (identical
technique to `variance.decompose`'s own resampling), rebuilds the panel, and reruns the **same,
unmodified** `argotech.lab.run.run_experiment` — full peer-transform-per-fold, target-build,
fit-and-score pipeline — restricted to `splits=[spatial]`, `arms=[linear, boosted]`. Each replicate
yields one net-benefit mean over its own (re-derived) four leave-one-cluster-out folds. The
distribution of those 200 replicate-level means gives a 2.5/97.5 percentile CI on the headline
number.

`bootstrap.result.json` (this directory) is the run's full output: every one of the 200 per-replicate
means for both arms, `boot_seed=20260826` (distinct from the model's own `seed=42`, used only to
drive the resampling draw), and `elapsed_seconds`.

### Result

| arm | point estimate | site-level bootstrap 95% CI | P(NB > 0) |
| --- | --- | --- | --- |
| `linear` | +0.0404 | **[-0.0026, +0.0781]** | 0.950 |
| `boosted` | +0.0218 | [-0.0016, +0.0424] | 0.950 |

200 of 200 replicates used, 0 skipped, `boot_seed=20260826` (distinct from the model's `seed=42`,
which drives nothing here since both arms are deterministic), 1634 s. Full per-replicate output in
`bootstrap.result.json`.

**This interval is narrower than the fold-level one, and that is not reassurance.** The fold-level
CI from `experiments/E02/` is [-0.0072, +0.0940], width 0.1012; this one is width 0.0808. The
controller predicted the opposite before seeing it, on the reasoning that resampling 122 sites
propagates more variance than resampling 4 fold scores. That reasoning was wrong, and the reason it
was wrong is the important part:

A site-level bootstrap resamples sites **within the four existing clusters**. Cluster identity is
held fixed in every replicate. So it measures *within-cluster* sampling variance — while the claim
under test is transferability to an *unseen region*, whose uncertainty is dominated by
*between-cluster* variance, which resampling sites cannot reach. Each replicate still averages the
same four clusters, and each fold score is itself a mean over many sites, so it is stable by
construction.

The site-level interval is therefore correctly computed and answers an easier question than the one
being asked. **The fold-level interval remains the relevant one for a transferability claim** — four
clusters are the independent units — and it is the wider of the two.

## Step 4 — does it survive

**No.**

On the more favourable of the two intervals, the site-level one, the 95% CI is [-0.0026, +0.0781]
and **includes zero**. On the fold-level interval, [-0.0072, +0.0940], it includes zero by a wider
margin. There is no interval here under which +0.0404 is distinguishable from the zero predictor.

The point estimate is positive and 95% of bootstrap draws are positive, so this is not evidence of
*no* effect — it is an absence of evidence *for* one, at four independent spatial units. The
distinction matters and should not be rounded either way.

Three facts, taken together, are the result:

1. **Two of four folds are negative** (Kaduna -0.0103, Kano -0.0042). The positive mean is carried
   by Benue at +0.1268, roughly 3x the headline.
2. **Benue is also the fold with the highest achievable ceiling.** Maximum attainable net benefit
   *is* the event rate; Benue's is 0.488 against Kano's 0.049, a factor of ten. Averaging net
   benefit across folds with that prevalence spread averages quantities that are not on a common
   scale, and the fold carrying the mean is the one with the most room to score.
3. **The fold spread (-0.0103 to +0.1268, range 0.137) is more than three times the reported effect
   (+0.0404).**

This is consistent with the rest of the investigation rather than an outlier: every spatial cell in
`experiments/E02/` has an interval crossing zero, and 0 of 12 cells in `experiments/E04-band-sweep/`
exclude zero. **No configuration tested in this investigation demonstrates out-of-region skill.**
Temporal skill is real and its intervals do exclude zero; spatial skill is not demonstrated.
