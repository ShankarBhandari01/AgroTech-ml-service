# Design — a two-way demeaned estimand, a lab, and a closed loop

Status: approved design, not yet implemented.
Supersedes the learned layer described in `docs/model-design.md` §4.4, §9 and §9.1. Leaves §4.3
(`domain/indices.py`), §4.5 (`domain/risk.py`) and §10 (serving architecture) standing.

Every empirical number below is recomputed from a committed file. Fold means come from
`artifacts/metrics.json` (4,596 samples, 4 held-out clusters, 40 features), not from prose. A claim
that cannot be reproduced from a committed file is marked `[UNVERIFIED]`, the convention
`docs/RESEARCH_SUMMARY.md` already uses.

---

## 1. The result being explained

From `artifacts/metrics.json`, means over the four leave-one-cluster-out folds:

| Blocked, 4-fold mean | Model | Persistence | Climatology |
| --- | --- | --- | --- |
| macro F1 | 0.4498 | 0.4783 | 0.4427 |
| precision@25 | 0.6000 | 0.4900 | **0.6400** |
| Spearman rho | 0.3976 | **0.4225** | 0.4145 |

Means over the three forward-chaining folds:

| Temporal, 3-fold mean | Model | Persistence | Climatology |
| --- | --- | --- | --- |
| macro F1 | 0.4631 | 0.4461 | **0.4803** |
| precision@25 | 0.6000 | 0.5733 | **0.7200** |
| Spearman rho | 0.4638 | 0.4020 | **0.5364** |

Temporally the model loses to climatology on all three metrics. Spatially it loses to climatology on
ranking and to persistence on rank correlation. Permutation importance on held-out ground is
dominated by a single feature: `ndvi_z_peer` at +0.0642, with the second-placed feature at +0.0051 —
a factor of 12.6.

Successive additions did not move this. Radar contributed +0.0066 macro F1 with ECE worsening by
0.0084; frozen Presto embeddings moved the boosted arm by −0.0228 and the linear arm by +0.0223; the
full ten-band variant moved precision@25 by +0.109 with a seed spread of 0.14 on a 25-item metric
(`docs/model-design.md` §9.1 E–G). A linear head beat gradient boosting out of cluster, 0.421 to
0.408 (§9.1 C).

Read together, these say the constraint is not capacity, not representation, and not sensing.

## 2. Diagnosis

The panel target decomposes as

    z_{i,t+1} = alpha_i + gamma_t + eps_{i,t+1}

where `alpha_i` is a time-invariant field effect, `gamma_t` a cohort-date effect, and `eps` the part
that weather and agronomy could in principle explain.

`forward_z` standardises each observation against its peers **at the same date within the same
cluster**, which removes `gamma_t`. It never standardises against the field's own history, so
`alpha_i` survives in the target.

`add_site_climatology` in `training/train.py` computes each field's mean prior peer anomaly. That is
an expanding-window estimator of `alpha_i`. So the "climatology baseline" and "the component the
target failed to remove" are the same quantity, and the baseline wins by collecting it directly
while the model must infer it from covariates. The permutation-importance profile is the signature:
the model's one useful feature, `ndvi_z_peer`, is the current-period level, which is the best
single-observation estimate of `alpha_i` available.

This is the panel-data within-transformation problem. Cross-sectional demeaning to remove
time-invariant unit characteristics is exactly the correction applied by "Forecasting Crop Yield
Anomalies on Panel Data via Spatially Demeaned Ensembles" (*J. Agric. Biol. Environ. Stat.*, 2026,
doi:10.1007/s13253-026-00743-8). Cited by title and DOI: the authorship shown by secondary indexes
could not be confirmed against the publisher record, so it is `[UNVERIFIED]` and deliberately
omitted rather than guessed. The repository performs one half of a two-way demeaning and is beaten by a
baseline that supplies the other half.

**Measured, E01 (run 2026-08-26 on `data/training_set.parquet`).** The field effect is **34.5% of
`forward_z` variance**, 95% CI [0.236, 0.435] over 1,000 site bootstraps, using an unbiased one-way
random-effects estimator rather than naive eta-squared. Two checks confirm the construction: the mean
of `forward_z` within a (cluster, label_date) cohort is +0.0014 with ICC 0.0000, so `gamma_t` is
indeed already removed; and the cluster ICC is exactly 0.0000 (§7).

That is substantial but **not** a majority share, so the diagnosis does not rest on it. The
discriminating comparison is what the field effect buys for nothing:

| Predictor | Fitted parameters | Spearman vs `forward_z` | Out-of-sample R-squared |
| --- | --- | --- | --- |
| `alphahat`, expanding-window field mean | **0** | **+0.437** | **+0.199** |
| 40-feature boosted model, blocked folds | ~thousands | +0.398 | -- |

A zero-parameter estimator attains rank correlation above what gradient boosting on 40 features
attains on held-out clusters (`metrics.json`: climatology 0.4145, model 0.3976). The model is not
merely failing to beat the field effect; it is failing to *capture* it. Consistently,
`ndvi_z_peer` -- the one feature carrying non-trivial permutation importance -- has a field-effect
ICC of 0.279, so the model's single useful signal is itself substantially a field-identity marker.

**Consequence for the literature, which is the research contribution.** Any NDVI-anomaly forecaster
evaluated against persistence but not against an estimated field effect may be reporting skill that
is a fixed effect in disguise. This repository has the negative result and the instrumentation to
demonstrate the mechanism.

## 3. Scope

**Retired.** `forward_z` as the primary target; `artifacts/experimental/` and the unpromotable
cluster-relative artifact; the `--no-radar` / `--no-bands` / `--seed` / `--folds-only` flag matrix as
the experiment interface; the manual copy-into-`artifacts/` promotion path; the unreferenced `mlflow.db` and `mlruns/` as they
stand today (no module imports mlflow; they are residue from an earlier iteration). They are reset,
not deleted -- §6 puts the same local store under management.

**Kept unchanged.** `domain/indices.py`, `domain/agronomy.py`, `domain/risk.py` — label-free
published relationships with nothing fitted, never validated against `forward_z`, and therefore
untouched by this result. They become the incumbent the learned layer must beat. `data/`,
`features/agronomic.py` (the single shared feature builder), the `/predict` and `/outcomes`
contracts.

**Refactored, not rewritten.** `training/dataset.py` carries the Sentinel and Open-Meteo backfill,
the 429 retry with backoff, the refuse-to-memoise-an-empty-result fix and the band cache namespace.
All of that survives as `lab/panel.py`. Only `training/train.py` is decomposed.

**Non-goals.** Field boundary polygons (`farms` stores points). A new geospatial foundation model.
Replacing FastAPI, Postgres, the CI/deploy path or the nightly job. Any change to the Kotlin
backend.

## 4. The estimand

Unit `i` is a field; time `t` is a 30-day bucket.

| Stage | Definition | Removes |
| --- | --- | --- |
| 1 — exists | `z_it = (y_it - mu_{c(i),t}) / sigma_{c(i),t}` over cluster peers at date `t`, min 5 peers, field excluded | `gamma_t` |
| 2 — new | `alphahat_it = shrink( mean_{s<t} z_is , m )`, expanding window, strictly prior | — |
| Target | `ztilde_{i,t+1} = z_{i,t+1} - alphahat_it` | `alpha_i` |

`alphahat` uses only observations strictly before `t`, so the existing leakage controls extend
unchanged. It requires a minimum history of `m` prior observations; `m` is a swept hyperparameter,
and fields below `m` are excluded from training and flagged at serving (§7).

**Shrinkage.** With few prior observations `alphahat` is noisy, and subtracting a noisy estimate
injects that noise into the target. `shrink` pulls the field mean toward **zero**, not toward a
computed cluster mean: `forward_z` is already standardised within cluster and date, the cluster ICC
was measured at exactly 0.0000 (§2), and a cluster mean computed over the whole frame would leak
future observations into a quantity defined as prior-only. The shrinkage weight is a config constant,
applied identically to every row before any split -- it is swept across experiments (E03), not
fitted per fold. `shrink` with weight 0 is the raw field mean, so the unshrunk case is a config value
rather than a separate code path.

**E01 quantifies what shrinkage is worth.** The field effect is 34.5% of variance, but subtracting an
unshrunk `alphahat` removes only 20.3% of it (Var 1.318 -> 1.051, recomputed via
`targets.build_target(min_history=1)`). The missing ~14 points is estimation noise injected by a
noisy `alphahat`, and it is the upper bound on what a better estimator can recover. E03 therefore has
a numeric objective rather than a hyperparameter to sweep: drive the realised variance reduction from
20.3% toward 34.5%. The transformation itself works as intended -- field-effect ICC falls
0.345 -> 0.073 and corr(`ztilde`, `alphahat`) is -0.061, so the baseline can no longer win by proxy.

**Frisch–Waugh–Lovell.** Residualising the target alone is not the within estimator; FWL requires
demeaning both sides. `targets.py` therefore implements both variants and E02 runs them
head-to-head:

- `within_y` — target demeaned, features left in levels.
- `within_xy` — target and the field-varying features both demeaned by their own prior field means.

The difference between them is itself a reportable quantity: it measures how much of the covariates'
apparent explanatory power was also a field effect.

**Baselines under `ztilde`.** The zero predictor is climatology by construction, so the baseline that
currently wins can no longer win by proxy. Persistence predicts `z_it - alphahat_it`. Majority and
the seasonal control are dropped: both were classification-era baselines, and a continuous target
has no majority class. Zero, persistence and climatology are recomputed inside every fold, as today.

**First deliverable, before any model is fitted — completed 2026-08-26.** E01 reports the variance
decomposition of `forward_z` on the panel with bootstrap intervals. Its result and the amended
reading are in §2.

The invalidation criterion originally written here — "the field effect is the majority share" — was
the wrong test, and E01 failed it (34.5%) while still supporting the diagnosis. It is replaced by the
test that discriminates: **does a zero-parameter fixed-effect estimator match or beat the fitted
model on held-out ground?** It does, on both Spearman and precision@25. A share-of-variance threshold
cannot separate "the target is mostly noise" from "the model cannot capture the structure that is
there", and only the second is actionable. This replacement is recorded rather than quietly applied,
because moving a criterion after seeing the result is exactly the move a reviewer should distrust —
the new criterion is stricter, and the old one is preserved above so the change is auditable.

## 5. Data and unit of analysis

Two populations, joined in one panel table.

**Lattice sites** — the existing 122–185 Halton-sampled points across 4–6 bounding boxes. They
supply statistical power and are what §1's numbers were measured on. They are not farms.

**Registered farms** — `backend_schema.list_fields` already returns every farm with usable
coordinates, reading `farms.latitude` / `farms.longitude` (the columns commit 95936ff corrected).
Sentinel-2, Sentinel-1 and Open-Meteo history can be backfilled retrospectively for any point, so
farm history is obtainable without waiting a season.

Farms are held out as an **external validation set**, never used for selection. This follows the
standard separation between blocked cross-validation for model selection and an independent set for
the transferability claim.

**Prerequisite task.** The farm count and geographic spread are not known from the repository. The
implementation plan's first task is a read-only census: how many farms, in which states, with what
coordinate spread. If the count is too small to support an external set, farms become a qualitative
case study and the lattice stays primary — that branch is decided by the census, not assumed here.

**Constraints carried forward, not solved.** `farms` stores points, not polygons, so the 500 m
Sentinel-2 support mismatch persists and remains a stated limitation. The Niger Delta — where the
registered farmers are — is humid forest with severe optical cloud loss, which promotes Sentinel-1
from the additive role §9.1 E measured to a load-bearing one there.

## 6. Lab architecture

```
src/argotech/lab/
  panel.py      build the (field_id, bucket) panel; emit parquet + a manifest sidecar
  targets.py    level_z | within_y | within_xy | delta_z; alphahat with shrinkage; min-history
  splits.py     leave-one-cluster-out, forward chaining, buffered blocking
  arms.py       zero, persistence, climatology, linear, boosted, +presto, +radar
  evaluate.py   net benefit, ranking with bootstrap CIs, ECE, conformal coverage, AOA
  run.py        entrypoint: python -m argotech.lab.run experiments/E02.yaml
experiments/
  E01-variance-decomposition.yaml ... one file per experiment
```

Six modules, each independently testable, replacing one 684-line file that currently holds dataset
loading, four arms, two split strategies, five metrics, a decision-rule sweep and a CLI.

**An experiment is a config file, not a flag.** Each YAML declares the panel manifest hash, target,
arms, splits, metrics and seed. `run.py` resolves it, executes it, and logs to MLflow: the resolved
config, the git SHA, the data manifest hash, the seed, every fold metric, and the artifact.

This closes a hole the repository already has. `docs/RESEARCH_SUMMARY.md` records that the README
and two source files cite a metrics table — 0.760 and 0.377 over six folds on 6,957 samples — that
exists in no committed file and is marked `[UNVERIFIED]`. Under this design, a number that cannot be
traced to a run id cannot reach a document.

**MLflow, local file store only.** `mlruns/` with the existing SQLite backend. No tracking server,
no new infrastructure. It is used for two things the paper needs — run comparison across dozens of
arms, and a model registry with stages — and for nothing else.

**Panel manifest.** A JSON sidecar beside the parquet holding the content hash, row count, site
count, cluster list, date range, and the code version that built it. An experiment records the hash
it ran against, so a rebuilt panel produces a different hash and the mismatch is visible rather than
silent. This is the controlled-comparison problem §9.1 E already ran into: the builder keys its
window off `date.today()`, so two builds are not comparable.

## 7. Evaluation and the release gate

**Primary metric: net benefit.** Decision curve analysis (Vickers & Elkin 2006) evaluates a model by
the consequences of acting on it, computed across the range of threshold probabilities, against
visit-all and visit-none. It replaces the unresolved macro-F1-versus-precision@k standoff that
`docs/model-design.md` §6 item 7 leaves as a decision "someone has to make consciously", and it
subsumes the §9.1 D alert-rate sweep: the operating point becomes a reported curve rather than a
constant chosen on principle.

The promotion criterion needs no invented threshold: the candidate must achieve net benefit greater
than or equal to the incumbent and both naive baselines -- where the incumbent is the currently
registered production model, or, when no learned model is registered, the `domain/` agronomy alone **across the whole declared interval of
cost ratios**, where the interval is recorded in `experiments/gate.yaml` and set by the product
owner, not by the modeller.

**Secondary metrics, all with intervals.** Precision@k and Spearman rho with bootstrap confidence
intervals over folds. This is a correction, not an addition: §9.1 G reports a precision@25 seed
spread of 0.14 on a metric computed over 25 items, and no interval currently accompanies any
reported number.

**Uncertainty.** Split-conformal intervals with weighted quantiles for non-exchangeable data
(Barber, Candès, Ramdas & Tibshirani 2023). Weighting is required rather than optional here: blocked
spatial folds and forward-chaining temporal folds both violate exchangeability by construction, so
standard split conformal has no coverage guarantee in this protocol. Empirical coverage on held-out
clusters is a hard gate.

**Applicability.** The Area of Applicability (Meyer & Pebesma 2021) computes a dissimilarity index in
predictor space and masks the region where cross-validated performance does not hold. This
formalises the open question in `docs/RESEARCH_SUMMARY.md` §7 — "a field outside every rectangle
gets no reference at all" — as a computed boolean rather than a caveat, and it is what makes the
Niger Delta case explicit instead of silent.

**Protocol.** Leave-one-cluster-out and forward chaining, both retained. Ploton et al. (2020)
establishes that non-spatial validation is overoptimistic for this class of model; the honest
counterweight, stated in the spec rather than discovered by a reviewer, is that four to six blocks
estimate transferability with very few degrees of freedom. The farm external set is the response to
that limitation.

**What the blocking does and does not control, measured (E01).** The cluster ICC of `forward_z` is
exactly 0.0000 — by construction, since `z` is standardised within cluster and date, so no target
variance survives at cluster level. Leave-one-cluster-out therefore blocks against **feature
distribution shift only**, not against target structure. This does not weaken the protocol; it
narrows what a blocked score is evidence *of*, and it should be stated whenever one is reported.
It also explains why cluster-relative feature twins (§9.1 B of `docs/model-design.md`) improved
ranking and calibration while leaving classification flat: they act on the axis the blocking
actually varies.

**Fairness.** The existing protected-attribute guard in `domain/risk.py` stands. Net benefit and
ranking metrics are additionally sliced by household headship, landholding size and district, and a
slice regression blocks promotion.

**CI gate.** A single job: net benefit over the declared interval, conformal coverage within
tolerance, no fairness-slice regression, AOA coverage reported. A model failing any of these does
not promote, and the failure names which one.

## 7a. The peer reference: fold-fitted, and what it costs to transfer

`ndvi_z_peer`/`rvi_z_peer` are model *features*, not the estimand target §4 describes, but they had
the same defect §2 diagnoses for the target, by a different route: `lab.panel.py` computed
`peer_stats` once over the whole frame and baked the result in as a static column, keyed
`cluster|MM` — calendar month pooled across all years, with no year boundary at all. 42 of the 45
buckets span multiple years; `Benue_River_Basin|01` alone pools 2023-01-08 through 2026-01-22.
Measured: across the lab's three forward-chaining boundaries, 74.0% / 49.9% / 24.3% of a training
row's peer cohort lay at or after the boundary (max 94.2%); under leave-one-cluster-out, the held-out
cluster's bucket drew from exactly one cluster — itself.

`argotech.lab.peers` now fits the reference per fold. `fit_peer_stats` sees training rows only;
`apply_peer_z` looks a row's bucket up in that fit, rewrites `ndvi_z_peer`/`rvi_z_peer` for both
train and test, and reports `peer_coverage` — the share of rows that found a bucket — as a first-
class fold metric, so a missing reference shows up in the record rather than silently defaulting.
Three cohort keys are offered:

- `leaky` — the pre-fix, whole-frame `panel.py` column, kept as the control that isolates how much
  of any reported skill the leak itself supplied.
- `cluster_month` — `cluster|MM`, fit honestly per fold. Cluster identity is exactly what a held-out
  region cannot supply: coverage of each held-out cluster is `[0.0, 0.0, 0.0, 0.0]`. An honest
  cluster-keyed fit leaves an unseen region with no reference at all — that is the correct answer,
  not a bug, and it is what the Applicability gate above exists to surface rather than paper over.
- `geo_month` — latitude/elevation bands in place of cluster identity, month unchanged. Coverage
  recovers to `[1.0, 1.0, 1.0, 0.472]`; the shortfall is Kenya, the panel's only highland cluster
  (1760 m mean), which borrows a reference only where its own sites fall below the 1000 m band.

Widening the geographic bands trades specificity for transfer, and the trade is measured rather than
assumed. Donor rows available to each held-out cluster (Benue 7.6°N/118 m, Kaduna 10.8°N/676 m, Kano
12.0°N/455 m, Kenya 0.5°N/1760 m), by band width:

| Band width | Benue | Kaduna | Kano | Kenya |
| --- | --- | --- | --- | --- |
| 5° / 500 m | 0 | 282 | 1098 | 0 |
| 10° / 1000 m | 36 | 1353 | 1098 | 647 |
| 20° / 1000 m | 2175 | 2799 | 1792 | 3139 |

At 5°/500 m, two of four held-out clusters get no donors at all. At 20°/1000 m every cluster is
fully served, but by then the key has stopped discriminating: the four clusters span only 0.5–12°N,
so a 20° band no longer separates any of them by latitude, and the key degrades to
elevation-and-month — pooling humid Benue with Sudan-savannah Kano, agroecologically distinct
regimes a cluster key would never have pooled.

A cohort tight enough to be agronomically meaningful — cluster identity, or a geographic band narrow
enough to still discriminate climate — does not transfer to an unseen region at this sample size (122
sites across 4 clusters); a band wide enough to transfer stops discriminating the thing the reference
is meant to control for. No fixed band width is simply correct here, so `lat_band`/`elev_band` are
swept experiment parameters (`experiments/E02-within-vs-level.yaml`), not a chosen constant. The
Applicability paragraph above already cites `docs/RESEARCH_SUMMARY.md` §7's open question — "How
should the reference cohort be defined so that it transfers to an unsampled region?" — as what the
AOA formalises; this section is the beginning of an answer to it, not a resolution: a measured trade
in place of an unexamined default.

**Only `level_z` supports a valid cross-`peer_key` comparison.** `within_y`'s target is `forward_z -
alpha_hat`, and `alpha_hat` is derived from `ndvi_z_peer`; `delta_z`'s target is `forward_z -
ndvi_z_peer` directly. Changing `peer_key` therefore changes `ndvi_z_peer`, which changes the TARGET
for both of those kinds — not just the features an arm gets to see. A table comparing `leaky` against
`cluster_month` against `geo_month` under `within_y` or `delta_z` is not comparing three models on
one problem; it is comparing three models each scored against its own, different problem. `level_z`'s
target is `forward_z` alone, which comes from the label's own (cluster, date) cohort at build time
(`lab/panel.py`) and never touches `ndvi_z_peer` — it is the one target `peer_key` leaves untouched,
and so the only one a cross-key table may legitimately be built from. Any such table must be
restricted to `level_z`, or must carry this caveat explicitly; `run.py` also emits a one-line warning
to stderr whenever a non-`level_z` target runs with a non-`leaky` `peer_key`, so generating one
without seeing this is not possible by accident.

For reference, the valid comparison — `level_z` net benefit, spatial / temporal — reads: `leaky`
+0.0064 / +0.0210; `cluster_month` −0.0019 / +0.0183; `geo_month` +0.0056 / +0.0185. Even here,
`cluster_month`'s spatial figure is negative and `geo_month`'s spatial figure is a fraction of
`leaky`'s — the fold-fitted reference costs real skill relative to the whole-frame leak it replaces,
which is the leak's own size, not a defect in the replacement.

**[UNVERIFIED against the current code]** The `cluster_month` spatial figure above (−0.0019) does
not reproduce under the harness as it stands today: `experiments/E02/E02-level_z-cluster_month`
(git_sha `dd59159`) shows every spatial fold skipped (0/4 scored), not a scored negative number.
97751bc ("Stop alpha_hat fabricating 0.0 when a fold has no peer reference at all") postdates
whatever run produced −0.0019 and is the likely reason: before that fix, a fold with no peer
reference silently fabricated `alpha_hat=0.0` rather than being dropped, which would produce a
real-looking (if meaningless) scored number where the current, corrected code reports "no target
could be built" instead. This paragraph's other five figures (`leaky` and `geo_month`, both
splits) do reproduce exactly against `experiments/E02/` — see that directory's README. Left as-is
here rather than silently edited, since resolving which number is right is outside this task's
scope; a reader citing `cluster_month`/spatial from this paragraph should use `experiments/E02/`
instead.

### Band sweep: what widening `lat_band`/`elev_band` actually buys and costs

Measured (`experiments/E04-band-sweep/`, `git_sha=dd59159`, `seed=42`, spatial folds,
`peer_key=geo_month`): `lat_band x elev_band` swept over `{5, 10, 20, 90} x {500, 1000, 2000}` on
both `level_z` and `within_xy`.

**Latitude stops discriminating at 20°, exactly, not approximately.** `lat_band=20` and
`lat_band=90` produce bit-identical net benefit, CI, donor-row counts and per-cluster coverage in
every one of 48 checked (cell x target x arm) rows. All four cluster mean latitudes (Benue 7.6°N,
Kaduna 10.8°N, Kano 12.0°N, Kenya 0.5°N) satisfy `floor(lat/20) == floor(lat/90) == 0`, so from 20°
up the key is `elevation-and-month` alone — confirming numerically what this section's paragraph
above already states from inspection of the band width.

**The binding constraint at every band width is Kenya's coverage, not Benue's or Kano's.** At
`elev_band=500` (lat >= 20), Benue/Kaduna/Kano each reach 0.86-1.00 peer coverage; Kenya sits at
0.25 — its 1760 m mean elevation is the outlier against the other three clusters' 118-676 m. Kenya
coverage rises 0.25 -> 0.47 -> 0.55 as `elev_band` widens 500 -> 1000 -> 2000, and net benefit
falls as it does: `within_xy`/`linear` (the arm carrying the +0.0404 result) goes
+0.0567 -> +0.0404 -> +0.0235; `boosted` goes +0.0296 -> +0.0218 -> +0.0186. Every point of Kenya
coverage bought past 500 m costs net benefit somewhere on the curve — a measured trade, not an
assumed one. Below 20°/500 m, tight bands can zero out coverage for *both* extreme-latitude
clusters at once (Benue and Kenya both reach 0 donor rows at 5-10°/500 m), collapsing
leave-one-cluster-out from 4 scored folds to 2.

**A thin donor pool can manufacture a spurious-looking peak.** `within_xy`/`linear` at
`10°/1000 m` reports net benefit +0.1580 — nominally the highest number in the sweep, well above
`20°/1000 m`'s +0.0404 — but it is a 4-fold mean dominated by one degenerate fold: Benue, whose
donor pool at that band is only 36 rows, scores net benefit +0.5989 with an event rate of 88.2%
(every other fold in the same run: 4-21%). A peer reference fit on 36 rows is a noisy, plausibly
biased estimate; applying it appears to have pushed nearly all of Benue's held-out rows below the
event threshold, and net benefit under a near-universal event rate is close to its ceiling almost
by construction. The 4-fold mean's own CI, [-0.0036, +0.4492], crosses zero and is wide enough that
this was reportable as a fold-level artifact rather than a genuine transfer improvement only
because per-fold values were inspected, not just the mean — see `experiments/E04-band-sweep/README.md`
Finding 3.

**Where the trade turns.** Coverage is "complete enough" (3 of 4 clusters >= 0.86, Kenya the only
partial one) starting at `20°/500 m`, which is also this sweep's highest defensible net benefit
(`within_xy` linear +0.0567). `20°/1000 m` — the width this investigation shipped — trades −0.0163
net benefit for +0.22 Kenya coverage (0.25 -> 0.47); `20°/2000 m` trades another −0.0169 for
another +0.08. `20°/1000 m` is not the point of maximum net benefit in this sweep; it is the point
past which every further coverage gain keeps costing net benefit, which is what a band width
argued from first principles ("the coarsest option that reaches every held-out cluster") should be
expected to be, and now is: measured, not assumed.

## 8. Serving

The `/predict/farmer`, `/predict/crop-health` and `/outcomes` contracts are preserved. Three
additions to the response:

- `applicability` — whether the request falls inside the AOA.
- `interval` — the conformal interval on the learned term.
- `reference_status` — whether the field has the `m` prior observations `alphahat` requires.

**Behaviour outside the AOA, or below minimum history: the learned term is withheld and the response
falls back to the `domain/` agronomy alone**, labelled as such. This is the honest answer for a
newly registered farm and for the Niger Delta, and it is strictly better than the current behaviour,
which is an undefined peer anomaly flowing into a hazard term.

`alphahat_i` is maintained per field by the nightly `jobs/precompute` run and stored on the
`field_features` row, so the request path gains no query and no latency. The 48-hour freshness
refusal in `store.is_fresh` covers it unchanged.

**Artifact contract.** `models/registry.py` currently enforces which feature columns an artifact may
require. It is extended to also declare required **per-field state** (`alphahat`, minimum history,
AOA training-space summary). This generalises the guard that already exists because promoting the
cluster-relative model would have raised a `KeyError` on the first live prediction — the same
failure class, caught at test time rather than remembered.

**Artifact format.** Export to ONNX, which removes the `scikit-learn==1.6.1` pin and the joblib
internal-module-layout coupling documented in `pyproject.toml`. Already on the §7 "still to do"
list; it is a precondition here because the registry pulls artifacts by version at startup rather
than loading a pickle committed to git.

## 9. The loop

1. **Ingest** — nightly `jobs/precompute` writes `field_features` as today, and additionally appends
   the night's observation to an append-only `panel` table. The training set accumulates as a side
   effect of serving rather than as a backfill job.
2. **Label** — `store.label_join` already pairs predictions with outcomes and is currently never
   called. It becomes a scheduled job writing T1 and T2 outcomes into the panel. This is the single
   change that makes the loop closed rather than drawn; `GET /outcomes/label-count` already exposes
   the counter that decides when supervised targets become viable.
3. **Train** — `lab.run` against the panel, every run logged.
4. **Gate** — §7, in CI.
5. **Register** — MLflow registry, staged, ONNX artifact.
6. **Serve** — pull by version at startup.
7. **Monitor** — per-feature drift (PSI), share of requests outside the AOA, conformal coverage
   against realised outcomes, and the existing `degraded_rate` (share of fields with no cloud-free
   scene). Because outcome labels arrive months late, these are the only near-real-time signals that
   the model has broken.
8. **Retrain** — triggered by label count crossing a threshold or by a drift alarm.

## 10. Experiment schedule

| Id | Question | Depends on |
| --- | --- | --- |
| E01 | Does a zero-parameter fixed-effect estimator match or beat the fitted model? | panel only — **done, §2** |
| E02 | Does `within_y` or `within_xy` beat the zero and persistence baselines where `level_z` did not? | E01 |
| E03 | What minimum history `m` and shrinkage weight minimise held-out error on `alphahat`? | E02 |
| E04 | Under the reformulated target, do radar and Presto still contribute nothing? | E02 |
| E05 | Does the linear arm still beat the boosted arm out of cluster once the field effect is removed? | E02 |
| E06 | What does the net benefit curve look like against visit-all and visit-none, and where does the model cross them? | E02 |
| E07 | Does conformal coverage hold on held-out clusters, and does weighting fix it if plain split conformal fails? | E02 |
| E08 | Does the result replicate on real registered farms as an external set? | farm census, E02 |

E01 is a function over a parquet file and needed no refactor. It ran on 2026-08-26; §2 carries the
result and §4 records the criterion change it forced. Implementation proceeds on that basis. E02 is
now the first experiment that can invalidate the design: if neither `within_y` nor `within_xy` beats
the zero predictor on held-out ground, the reformulated target carries no learnable signal at this
resolution and the recommendation becomes shipping the `domain/` agronomy alone.

## 11. Testing

The existing runnable self-check style is kept; no new frameworks.

- `test_targets.py` — `alphahat` uses no future data, verified on a synthetic panel with a planted
  future spike that must not move any earlier estimate; `within_y` has approximately zero mean per
  field over its own training window; a field below `m` yields no sample rather than a default.
- `test_evaluate.py` — net benefit reproduces the known closed form on a hand-computed case; the
  zero predictor scores exactly the visit-none reference.
- `test_aoa.py` — a point far outside the training predictor space is masked out.
- `test_artifact_contract.py` — extended: an artifact declaring per-field state it will not receive
  fails at test time, not at first prediction.
- `test_panel.py` — the manifest hash changes when the data changes and not when it does not.

Existing leakage, feature, peer-statistic and end-to-end tests are retained unchanged.

## 12. Risks

- **The residual may be near-unpredictable.** Removing `alpha_i` may leave noise that weather and
  agronomy cannot explain at this resolution. Given E01, that is a publishable result with a
  mechanism rather than a failure, but it is the most likely outcome and is accepted in advance.
- **Shrinkage is a hand-set constant baked into the target, not a fitted quantity.** The weight is a
  config value applied identically to every row before any split — arguably less defensible than
  something fitted per fold, since it is the modeller's choice rather than data-driven. E03 must
  report sensitivity to it, because a target that moves with a hyperparameter invites exactly the
  criticism this design is correcting.
- **Few blocks.** Four to six spatial folds remain few, whatever the target. The farm external set
  and reported intervals are the mitigation; they do not eliminate it.
- **Farm population unknown.** §5 makes the census a blocking first task rather than an assumption.
- **Support mismatch is unsolved.** Point coordinates, a 500 m satellite box, a ~9–25 km reanalysis
  cell, and a per-farm advisory. This design does not close that gap and does not claim to.

## 13. Sources

| Design choice | Source |
| --- | --- |
| Cross-sectional demeaning removes time-invariant unit effects in crop-anomaly forecasting | Forecasting Crop Yield Anomalies on Panel Data via Spatially Demeaned Ensembles, *J. Agric. Biol. Environ. Stat.*, 2026. https://link.springer.com/article/10.1007/s13253-026-00743-8 |
| Demeaning/detrending applied post hoc carries bias; estimate jointly | Comprehensive review of detrending methods for crop yields, *Field Crops Research*, 2026. https://www.sciencedirect.com/science/article/pii/S016819232600002X |
| Residualising one side only is not the within estimator | Frisch & Waugh, *Econometrica* 1(4), 1933; Lovell, *JASA* 58(304), 1963 |
| Evaluate by consequences of the decision, not by discrimination alone | Vickers & Elkin, Decision Curve Analysis, *Medical Decision Making* 26:565-574, 2006. https://journals.sagepub.com/doi/10.1177/0272989X06295361 |
| Non-spatial CV is overoptimistic for spatially structured data | Ploton et al., *Nature Communications* 11, 2020. https://www.nature.com/articles/s41467-020-18321-y |
| Masking the region where CV performance does not hold | Meyer & Pebesma, *Methods in Ecology and Evolution* 12, 1620–1633, 2021. https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13650 |
| Conformal intervals when exchangeability fails under shift | Barber, Candès, Ramdas & Tibshirani, *Annals of Statistics* 51(2), 816–845, 2023. https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.full |
| Conformal prediction applied to Earth observation | Uncertainty quantification for probabilistic ML in Earth observation using conformal prediction, arXiv:2401.06421 |
| Cross-validation design for real-world transferability of satellite vegetation models | Bringing cross-validation into the real world to evaluate transferability of satellite-based vegetation models, *Scientific Reports*, 2026. https://www.nature.com/articles/s41598-026-39866-w |
| Frozen pretrained encoder for remote-sensing pixel timeseries (already vendored) | Tseng et al., Presto, arXiv:2304.14065 |
| Vegetation Condition Index as the within-field reference | Kogan, 1990 (retained from `domain/indices.py`) |
| Hazard x Exposure x Vulnerability risk framing | IPCC AR5/AR6 (retained from `domain/risk.py`) |

## 14. What this design does not change

The FastAPI service and its three contracts; the Kotlin backend and its schema; the nightly job's
schedule and single-instance property; the CI deploy path and rollback-by-tag; the `domain/` layer;
`features/agronomic.py` as the single feature builder shared by both paths; the protected-attribute
guard.
