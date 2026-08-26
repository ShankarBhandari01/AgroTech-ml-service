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
Temporally, forward chaining over three folds, cut on the prediction date rather than the label
date, so the last month of training outcomes becomes known after the first test predictions are
made. That is not corrected. Three baselines are recomputed per fold: majority class, persistence,
and site climatology.

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

How should the reference cohort be defined so that it transfers to an unsampled region? The current
cohort is a hand-drawn rectangle, and a field outside every rectangle gets no reference at all.

What is the relationship between a peer-relative canopy anomaly and a realised agronomic outcome?
That decides whether the proxy is worth optimising, and cannot be approached without a prospective
outcome record.
