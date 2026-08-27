# Forecasting peer-relative canopy anomalies for extension triage

Every claim comes from code, data or metrics committed to this repository; a number I cannot
reproduce from a committed file is marked `[UNVERIFIED]`. Citations are in `docs/FACTS.md`.

## 1. Problem

The decision is triage: an extension officer can visit a few farms a week, and the question is which
ones. That is a ranking problem, not a classification one. It is hard here because no outcome labels
exist — nothing in this repository records a harvest, a diagnosis or a field visit — so the target
must be built from something observable. Absolute greenness does not serve, since an NDVI indicating
stress in irrigated maize is unremarkable in Sahelian sorghum.

## 2. Approach

Risk is decomposed as Hazard × Exposure × Vulnerability (IPCC AR5/AR6). Hazard is a noisy-OR over
four terms; drought, disease and heat are closed-form published agronomy with nothing fitted. Only vegetation is learned — a regressor predicting a canopy anomaly thirty days out,
mapped through a fixed logistic — so the model is one of four inputs to one of three terms.

## 3. Data

NDVI, NDMI and EVI from Sentinel-2 L2A via the Copernicus Statistical API: 30-day means over a 500 m
box at 60 m, excluding scenes above 40 % cloud, plus Sentinel-1 backscatter. Weather is Open-Meteo's
archive at daily resolution, six variables; the docstring calls it ERA5, but no model parameter is
passed, so the grid spacing is `[UNVERIFIED]`. Beyond elevation there is no soil product, and no
household data reaches the model.

The training set is 4,596 samples from 122 sites in four bounding boxes in northern Nigeria and the
Kenyan Rift Valley, 2022-09-10 to 2026-06-21, class balance 68.3 / 19.0 / 12.8 per cent. The sites
are lattice points, not farms.

## 4. Label construction

The target, `forward_z`, is the field's NDVI at the next 30-day bucket, standardised against the
other sites in the same cluster observed in that same bucket: take the peer values for that
(cluster, date), require at least five, exclude the field itself, and compute z = (NDVI − mean) / sd.
The model regresses this continuous z; the discretisation at −1.0 and −0.35 serves only a control
classifier and the classification metrics.

The reasoning for a peer-relative target is that it removes what the cohort shares — season, weather
regime, soil background — and leaves what distinguishes a field from its neighbours, which is what a
triage decision turns on. It carries a cost I return to below: it cancels exactly the covariates the
weather block supplies.

Lead time is one bucket. Since the upstream omits intervals with no cloud-free scene, the next
bucket can fall much further out, so a guard rejects pairs more than 45 days apart; in the committed
data the gap is exactly 30 days for all 4,596 rows. This cohort exists only in the training builder:
the peer anomalies used as model *features* take a coarser reference, cluster × calendar month,
shipped as constants in the artifact, because a serving request holds one row and cannot form a
cohort.

Leakage is controlled in six places, among them a strictly prior 90-day feature window and a
rainfall climatology excluding the sample's own year. One transform was not blocked: the
cluster-relative feature twins and the peer reference feeding `ndvi_z_peer`/`rvi_z_peer` were both
fitted over the whole frame, held-out clusters included, and defended as admissible because both are
label-free. That defence addressed label leakage only; it left transduction over the feature
distribution unmeasured. Measured, it was real: `peer_stats` pooled all years into one `cluster|MM`
bucket — 42 of 45 buckets span multiple years, `Benue_River_Basin|01` alone runs 2023-01-08 through
2026-01-22 — and across the lab's three forward-chaining boundaries, 74.0% / 49.9% / 24.3% of a
training row's peer cohort lay at or after the boundary (max 94.2%). Under leave-one-cluster-out the
held-out cluster's bucket drew from exactly one cluster: itself.

The peer reference is now fit per fold (`argotech.lab.peers`): training rows populate the fit, a
held-out row never contributes to its own standardisation, and a fold that cannot see a region
records no reference for it rather than fabricating one — coverage of a held-out cluster is
`[0.0, 0.0, 0.0, 0.0]` under an honest cluster-keyed fit. A geographic key built from latitude and
elevation bands instead of cluster identity restores partial coverage, `[1.0, 1.0, 1.0, 0.472]`, the
shortfall being Kenya, the panel's one highland cluster, which borrows a reference only below the
1000 m band — at the cost of pooling agroecologically distinct places once the bands widen enough to
reach every held-out cluster. That transfer-versus-specificity trade is measured in
`docs/superpowers/specs/2026-08-26-two-way-demeaned-estimand-design.md` §7a. The cluster-relative
feature twins (`cluster_stats`/`add_cluster_relative`) are a separate mechanism from the peer
reference and this fix did not touch them; they remain fitted over the whole frame and are not
defended as settled.

## 5. Evaluation

Spatially, leave-one-cluster-out, the block being the whole bounding box, roughly 120–180 km on a
side. There is no buffer parameter anywhere in the code and no within-cluster blocking; separation
rests on the clusters lying several hundred kilometres apart. Four clusters give four folds.
Temporally, purged and embargoed forward chaining over three folds: training rows take
`label_date < boundary`, test rows take `obs_date >= boundary`, and rows satisfying neither belong to
neither side (52/31/69 embargoed per fold). An earlier version of this document recorded cutting
training on the prediction date, `obs_date`, as the defect and reported it "not corrected"; a
subsequent change then shipped exactly that defect again, under a commit message claiming to fix it.
**It is now corrected.** Measured on the panel: the label lag is exactly 30 days on all 4,596 rows;
cutting training on `obs_date < boundary` gives 121 rows whose outcome postdates the first test
prediction, against zero under the fix. Three baselines are recomputed per fold: majority class,
persistence, and site climatology. The peer reference the label and features depend on is now fit
per fold rather than over the whole frame (`argotech.lab.peers`), under three keys — `leaky`, the
whole-frame control; `cluster_month`; and `geo_month`, latitude-and-elevation bands. Results below use
`level_z`, the only target that supports comparing across keys: `within_y`, `within_xy` and `delta_z`
are defined *through* `ndvi_z_peer`, so changing the key changes the target, not just the estimator.

**The finding that governs how to read every number below.** 165 spatial cells were tested across the
leave-one-cluster-out grid and the band sweep, and 8 cleared a 95% interval excluding zero — against
~4.1 expected by chance with no correction, a ratio of 1.9×. Temporally, 65 cells were tested and 13
cleared zero against ~1.6 expected, 8.0×. No formal correction (Bonferroni, Benjamini–Hochberg) is
applied; the raw ratio is reported instead of one. On that basis: temporal skill is established;
out-of-region (spatial) skill is not distinguishable from multiple-testing noise at four independent
spatial units. That does not mean no configuration shows spatial skill — 8 of 165 cells do clear
zero, and an earlier version of this document claimed none did; that claim was wrong and is corrected
here. The argument against reading those 8 cells as skill is the ratio above, not a denial that they
exist. The spatial hits are also internally incoherent: `level_z`/`geo_month`/boosted clears zero at a
500 m elevation band and at 2000 m but not at the 1000 m band between them, which is not the shape a
real effect produces.

Read against that: under the temporal protocol `level_z` clears zero at every peer key measured —
`leaky` +0.0194 [+0.0182, +0.0216], `cluster_month` +0.0179 [+0.0152, +0.0202], `geo_month` +0.0172
[+0.0162, +0.0182]. Under the spatial protocol it does not clear zero at any key that could be
evaluated — `leaky` +0.0064 [-0.0014, +0.0142], `geo_month` +0.0066 [-0.0020, +0.0157] — and
`cluster_month` could not be evaluated at all: every fold skipped, because an honest cluster-keyed fit
leaves a held-out region with no reference to compute against. The single largest cell in the matrix
is spatial / `geo_month` / `within_xy` / linear, net benefit +0.0404, 95% CI [-0.0072, +0.0940]
(fold-level bootstrap) and [-0.0026, +0.0781] (a 200-replicate site-level bootstrap, P(NB>0)=0.950) —
both cross zero. Per fold: Benue +0.1268, Kaduna -0.0103, Kano -0.0042, Kenya +0.0491. Two of four
folds are negative, and the mean is carried by Benue at roughly 3× the headline figure. Maximum
attainable net benefit is the event rate, and folds are therefore not on a common scale — Benue's
ceiling is 0.488 against Kano's 0.049, a factor of ten — so the fold carrying the mean is also the
fold with the most room to score. No seed sweep could sharpen this further: predictions are
bit-identical across five seeds for both arms (`linear` is a closed-form ridge solve; `boosted` runs
with `early_stopping=False`), so a sweep would report five copies of one number, not five draws.

**The mechanism behind that split, which matters more than any single number above.** Comparing the
temporal column across two targets isolates it: `level_z`, which *retains* the field effect, clears
zero for a learned arm and for the field-mean baseline alike — `boosted` +0.0194 [+0.0182, +0.0216],
`climatology` +0.0086 [+0.0044, +0.0116]. `within_xy`, which *removes* it by demeaning the target and
the features per Frisch–Waugh–Lovell, does not — `boosted` -0.0011 [-0.0051, +0.0031], `linear`
-0.0145 [-0.0173, -0.0129], `persistence` -0.0468 [-0.0693, -0.0234]: every learned arm at or below
zero, several significantly so. Remove the field effect and the temporal skill goes with it. A
separate field-effect decomposition put that effect at 34.5% of `forward_z`'s variance directly
(95% CI [0.2357, 0.4353], 1,000 site bootstraps); this reaches the same place from the opposite
direction and licenses a stronger claim — at this resolution the field effect is essentially all of
the predictable signal. That single mechanism accounts for three things otherwise reported as
separate observations: why climatology was so hard to beat, why the model behaved like a smoothed
persistence model, and why the reformulated target has nothing left to predict. Read this way, the
negative result is not "the model does not work." It is that the reformulation did exactly what it
was designed to do, and what remains after removing the field effect is not predictable from these
features, on this data, at four independent spatial units.

Robust losses were tested against the heavy tails described in §4 (`linear_huber`, `boosted_abs`, run
alongside the existing arms, not replacing them) and do not help: equal or worse everywhere, clearly
worse where signal exists — `level_z` temporal `boosted` +0.0194 against `boosted_abs` +0.0126. The
tails are real but were not hurting the fit; down-weighting them costs signal rather than recovering
any.

Every number above was computed on the panel exactly as committed, and that panel is known to contain
233 rows (5.07%) with physically invalid NDVI (NDVI < 0, i.e. NIR ≤ Red — water, cloud, shadow or
snow, never vegetation), which drives `forward_z`'s tail: rows with |z| > 3 are 8× enriched for a
negative label-date NDVI. A fix at the source (`MIN_VALID_NDVI = 0.0`, `MIN_COHORT_SD = 0.005`,
replacing a `1e-6` guard) landed in `argotech/lab/panel.py`, but **the panel was deliberately not
rebuilt**: the meteo cache is keyed on `date.today()` and the committed panel was assembled across at
least five distinct fetch windows, so a rebuild today would both invalidate 38 committed result files
pinned to the current `content_hash` and confound the NDVI fix with a change in weather data. Stated
plainly: every result in this section, without exception, was computed on a panel that still contains
the invalid observations.

From `artifacts/metrics.json`, written beside the shipped artifact: over four leave-one-cluster-out
folds the regressor takes mean precision-at-25 of 0.600 and Spearman ρ 0.398, against climatology's
0.640 and 0.415 and persistence's 0.490 and 0.423. Over three forward-chaining folds it takes 0.600
and 0.464 against climatology's 0.720 and 0.536. On the committed numbers the model does not beat
the stronger baseline under either protocol. Permutation importance on the held-out cluster is
dominated by one feature, the current peer anomaly, at +0.0642; the second is +0.0051 and 23 of 40
are negative. That feature is what persistence uses on its own. The README and two source files cite
a different table — 0.760 and 0.377 over six folds on 6,957 samples — attributed to this same file.
It is not in the repository, and I mark it `[UNVERIFIED]`.

## 6. Limitations

The sample size is smaller than 4,596 suggests: the independent units are 122 sites in four
clusters, and monthly observations of one site are strongly autocorrelated, so the spatial protocol
yields four estimates rather than thousands. No site is a real farm.

The label is a proxy licensing one claim — that a field's canopy will fall behind its cluster peers
over thirty days. It says nothing about yield, disease incidence, income, or whether a visit would
have helped, and no outcome data exists here to measure the gap. The standardisation is also
self-defeating for the weather block, since a z-score within a cluster-and-date cohort cancels
whatever that cohort shares — and a reanalysis cell covering several sites is such a quantity. The
importances agree: the model behaves close to a smoothed persistence model.

There is a resolution mismatch at every layer — a reanalysis cell of unstated size `[UNVERIFIED]`
probably spanning several sites, a satellite mean over a 500 m box, and a per-farm advisory as
output. A smallholder plot is smaller than any of them. Coverage is four savannah or highland zones
over just under four years, none in the Niger Delta, which the code names as where the only
registered farms are and where the cluster lookup returns nothing, leaving the peer anomaly
undefined.

## 7. Open questions

Does a forecast of peer-relative canopy state carry information beyond the field's current
peer-relative state? On the committed evidence it does not clearly do so, and the question is
whether that reflects the task or the target formulation.

What spatial support must a meteorological covariate have before it contributes skill at field scale
under a within-cohort standardised target?

How should the reference cohort be defined so that it transfers to an unsampled region? Partly
answered this session: an honest cluster-keyed fit gives a held-out region no reference at all —
measured coverage `[0.0, 0.0, 0.0, 0.0]`. A geographic key (latitude and elevation bands) restores
coverage — `[1.0, 1.0, 1.0, 0.472]`, the shortfall being Kenya's highland band — but restoring
coverage is not the same as demonstrating skill: under `level_z`/`geo_month` the spatial interval
still crosses zero, and the 24-cell band sweep found net benefit did not separate from zero at any
band width tested. What remains open is whether a better-chosen key would change that, or whether
out-of-region skill genuinely is not there to find. The current cohort is still a hand-drawn
rectangle, and a field outside every rectangle gets no reference at all.

What is the relationship between a peer-relative canopy anomaly and a realised agronomic outcome?
That decides whether the proxy is worth optimising, and cannot be approached without a prospective
outcome record.
