# Concepts, formulas, and why — with the warrant for each

Every entry gives four things: **what it is**, **the formula as implemented** (with the file that
implements it, so the document cannot drift from the code), **why it is used here**, and **the
warrant** — what makes the claim valid.

## On warrants, and what "no source" means

An uncited claim is not automatically invalid. Different claims are validated differently, and
naming which kind a claim is matters more than attaching a citation to everything:

| Kind of claim | What validates it | Citation? |
| --- | --- | --- |
| **Borrowed science** — NDVI's formula, FAO-56, IPCC risk framing, Kogan's VCI | a citation to the source | **required** |
| **Measurement from this repository** — the field-effect ICC, the leak percentages | reproducibility: data manifest hash + git SHA + seed | no — a citation would be *wrong*, nobody published our ICC |
| **Design choice** — shrink toward zero, the alert threshold, band widths | a stated rationale, ideally with a measurement behind it | only if the choice is borrowed |
| **Novel finding** — the target/reference pairing, the noise argument | the argument and the evidence for it | no — it is the contribution |

What is *not* acceptable is a claim with no warrant of any kind. That is what `[UNVERIFIED]` marks
in this repository, and it is what the metrics table cited in the README but present in no committed
file was.

Citations below are marked **[verified]** where the source was checked against the publisher or an
authoritative record during this work, and **[UNVERIFIED]** where it is asserted from memory or
from a secondary index and has not been confirmed. Treat the second category as a to-do before
submission, not as a fact.

---

# Part 1 — Spectral and radar indices

Implemented in `src/argotech/domain/indices.py`. All are pure functions over reflectance values.

### NDVI — Normalised Difference Vegetation Index
**Formula.** `NDVI = (NIR − Red) / (NIR + Red)`
**Why.** Chlorophyll absorbs red and reflects near-infrared, so the contrast tracks green biomass.
It is the most widely used vegetation index and the basis of this project's target.
**Range and interpretation.** Bounded [−1, 1]. Above ~0.2 is vegetation; **at or below 0 is not
vegetation at all** — it means NIR ≤ Red, which is water, cloud, shadow or snow. This matters here:
233 rows (5.07%) of the committed panel have NDVI < 0, two of them exactly −1.0 (NIR = 0, a no-data
sentinel), and they drive the target's extreme tail (Part 4).
**Warrant.** Rouse et al. 1974, ERTS symposium. **[UNVERIFIED]** — universally attributed, not
checked here.

### SAVI — Soil-Adjusted Vegetation Index
**Formula.** `SAVI = ((NIR − Red) / (NIR + Red + L)) × (1 + L)`, with `L = 0.5`.
**Why.** NDVI reads bare soil as crop failure. Smallholder plots are soil-dominated for the first
weeks of a season, so on sparse canopy SAVI is the honest index and NDVI is misleading.
**Warrant.** Huete, A.R. (1988), *A soil-adjusted vegetation index (SAVI)*, Remote Sensing of
Environment 25:295–309. **[verified]** — formula and the `L = 0.5` default confirmed.

### EVI — Enhanced Vegetation Index
**Formula.** `EVI = G · (NIR − Red) / (NIR + C1·Red − C2·Blue + L)` with `G = 2.5, C1 = 6,
C2 = 7.5, L = 1`.
**Why.** NDVI saturates on dense canopy and is sensitive to aerosols and soil background. EVI
corrects both and keeps responding where NDVI has flattened.
**Warrant.** Huete et al. (2002), Remote Sensing of Environment 83:195–213 (the MODIS VI paper).
**[verified]** — the implemented coefficients are exactly the published MODIS values.

### NDMI — Normalised Difference Moisture Index
**Formula.** `NDMI = (NIR − SWIR1) / (NIR + SWIR1)`
**Why.** Canopy water content, the earliest visible drought signal — it moves before greenness does.
**A naming correction carried in this repo:** what the codebase originally called "NDWI" is this
formula. McFeeters' NDWI is `(Green − NIR)/(Green + NIR)` and is a *water-body* index, not a crop
index. The rename was a real fix, not cosmetics.
**Warrant.** Gao, B.-C. (1996), Remote Sensing of Environment 58:257–266. **[UNVERIFIED]**.

### NDRE — Normalised Difference Red Edge
**Formula.** `NDRE = (NIR − RedEdge) / (NIR + RedEdge)`
**Why.** NDVI saturates above roughly LAI 3; NDRE does not, and it tracks nitrogen status mid-season
when NDVI has stopped discriminating.
**Warrant.** Barnes et al. 2000 / Gitelson & Merzlyak red-edge literature. **[UNVERIFIED]**.

### VCI — Vegetation Condition Index
**Formula.** `VCI = 100 × (NDVI − NDVI_min) / (NDVI_max − NDVI_min)`, clamped to [0, 100].
**Why.** Absolute NDVI is meaningless across crops and agro-ecologies — an NDVI indicating stress in
irrigated maize is unremarkable in Sahelian sorghum. VCI places the current value within *the
field's own* historical range, which is comparable across places. Below 35 is conventionally read as
drought stress.
**Warrant.** Kogan, F.N. (1990), *Remote sensing of weather impacts on vegetation in
non-homogeneous areas*, International Journal of Remote Sensing 11:1405–1419,
doi:10.1080/01431169008955102. **[verified]** — formula confirmed identical to the implementation.

### RVI — Radar Vegetation Index, and the cross-pol ratio
**Formulas.** `RVI = 4·VH / (VV + VH)`; `vh_vv_ratio = VH / VV`.
**Why.** Optical sensing fails under cloud, which in sub-Saharan Africa means it fails during the
rainy season — exactly when disease and water-stress signals matter most. C-band radar does not care
about cloud. RVI responds to volume scattering, so it reads canopy *structure* where NDVI reads
greenness: ~0 over bare soil, ~1 under dense canopy.
**One implementation detail that is load-bearing:** backscatter is averaged in **linear power, not
decibels**. Averaging dB averages logarithms, which is not the mean backscatter.
**Warrant.** Kim & van Zyl radar vegetation index literature. **[UNVERIFIED]**. The linear-vs-dB
point is a mathematical fact, not a citation.

---

# Part 2 — Agronomy

Implemented in `src/argotech/domain/agronomy.py`. Published closed-form relationships with nothing
fitted, which is why they transfer across regions and need no training data.

### GDD — Growing Degree Days
**Formula.** `GDD = max(0, (min(Tmax, cap) + Tmin)/2 − base)`, with `base = 10 °C`, `cap = 30 °C`
for maize.
**Why.** Plant development tracks accumulated thermal time, not calendar days. The upper cap exists
because development does not keep accelerating above ~30 °C, and an uncapped sum overstates
progress in a heatwave.
**Warrant.** Standard base-10 °C maize scale; McMaster & Wilhelm (1997) is the usual methods
reference for the cap variants. **[UNVERIFIED]**.

### Phenology stage
**Definition.** Cumulative GDD mapped to a named stage (emergence → vegetative → flowering →
grain-fill → maturity).
**Why.** Stage determines everything downstream: the crop coefficient, heat sensitivity, and whether
an advisory is even actionable. The incumbent hard-coded the stage string `"Vegetative / Flowering"`
for every field on every date, which made every stage-dependent quantity fictitious.
**Warrant.** Crop-specific GDD thresholds, FAO and extension sources. **[UNVERIFIED]** per crop.

### FAO-56 water balance
**Formula.** `deficit = Kc(stage) × ET0 − rainfall − irrigation`, accumulated over the window.
**Why.** Total rainfall alone cannot express water stress: 40 mm is generous at emergence and a
drought at flowering. Kc scales reference evapotranspiration to what *this* canopy at *this* stage
actually transpires. Open-Meteo returns `et0_fao_evapotranspiration` directly, so this costs one
query parameter rather than a Penman–Monteith implementation.
**Warrant.** Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998), *Crop Evapotranspiration —
Guidelines for Computing Crop Water Requirements*, FAO Irrigation and Drainage Paper 56.
**[verified]** — authors, title and the single-Kc approach confirmed.

### Longest dry spell
**Definition.** The longest run of consecutive days below a rain threshold within the window.
**Why.** Seasonal totals hide the two-week gap at silking that actually destroys a maize crop. A
total and a distribution are different facts and only the second one predicts loss.
**Warrant.** Agronomic argument, not a borrowed formula.

### Heat stress days
**Definition.** Count of days with Tmax above a pollen-viability threshold (35 °C default),
**counted only during flowering**.
**Why.** Heat above the pollen threshold is destructive at flowering and largely irrelevant at other
stages, so an unconditional count measures the wrong thing.
**Warrant.** Pollen viability thresholds, crop physiology literature. **[UNVERIFIED]**.

### DSV — Daily Severity Value (late blight)
**Definition.** A lookup on (mean temperature during the wet period, hours of wetness) returning
0–4, accumulated to a spray threshold of **18**.
**Why.** This is the validated form of what the incumbent's `rh_85_consecutive_hrs` and
`incubation_hours` were groping toward — same inputs, published thresholds, and an output that maps
to an action rather than to a number.
**Warrant.** Wallin, J.R. (1962), severity values from duration of RH ≥ 90% and mean temperature
during that period; BLITECAST (Krause, Massie & Hyre 1975) combines Wallin's severity values with
Hyre's favourable-days rule; the 18–20 SV spray threshold is Wallin's. **[verified]** — including
the threshold value the code uses.
**Deviation to record:** the implementation drives DSV from **leaf-wetness hours** while Wallin
specified **RH ≥ 90% duration**. The substitution is common in the literature but it is a
substitution and should be stated as one.

### Fall armyworm degree-days
**Formula.** Generations `= cumulative GDD(base 10 °C) / 390`.
**Why.** *Spodoptera frugiperda* has been the dominant maize pest across sub-Saharan Africa since
2016, and degree-day accumulation is how scouting is timed.
**Warrant.** ~390 DD per generation at base 10 °C. **[UNVERIFIED]** — widely quoted, not confirmed
here.

---

# Part 3 — Risk composition

Implemented in `src/argotech/domain/risk.py`.

### Risk = Hazard × Exposure × Vulnerability
**Why.** Each term is computed and validated separately, then combined by arithmetic anyone can
check. A single fused score is unvalidatable (the weather half is checkable in days, the
socio-economic half only against harvest), unactionable ("risk 78" does not say whether to bring
fungicide or a loan form), and unauditable.
**Warrant.** IPCC AR5/AR6 risk framing; the standard in agricultural early warning. **[verified]**
as a body of work, though no single page is cited here.

### Noisy-OR hazard combination
**Formula.** `H = 1 − Π(1 − h_i)` over the component hazards.
**Why.** Hazards compound. Two independent hazards of 0.4 give 0.64, not 0.4 — a field facing both
drought and blight really is worse off than one facing either. A max() or a mean would say
otherwise.
**Warrant.** Standard probabilistic construction for independent causes; not specific to agronomy.

### Vulnerability as a modulation, not a multiplier
**Formula.** `loss_rate = H × (0.5 + 0.5·v) × MAX_LOSS_FRACTION`
**Why.** A raw multiplier would let a well-resourced farm facing a severe hazard score near zero. It
should not: a farmer with irrigation and credit still loses the crop to late blight if nobody tells
them to spray. The `(0.5 + 0.5v)` form bounds vulnerability's influence to a factor of two.
**Warrant.** Design choice, argued not borrowed. The constant 0.5 is a judgement.

### Exposure, and two outputs
**Formula.** `exposure = area × expected_yield × price`; `expected_loss = exposure × loss_rate`.
**Why.** `risk_score` (0–100, a loss *rate*) is comparable across farms of any size and is what a
farmer sees. `expected_loss` (currency) is what a triage queue ranks by and what makes the return on
an extension visit measurable. They answer different questions and collapsing them loses one.
**Warrant.** Design choice.

### The protected-attribute guard
**Definition.** `head_gender` and `household_max_education` raise a `ValueError` if used as
vulnerability inputs; they are retained only for *measuring* disparate impact.
**Why.** The output ranks farmers for scarce extension visits, credit referral and input subsidy.
Trained on real survey data, a protected attribute would encode the historical correlation between
household headship and disadvantage and then *act* on it. A constraint that lives only in a comment
gets violated the first time someone adds a column.
**Warrant.** Fairness argument, enforced in code rather than asserted.

---

# Part 4 — The estimand (the research core)

Implemented in `src/argotech/lab/targets.py` and `src/argotech/lab/peers.py`.

### Peer standardisation — the label
**Formula.** For field *i* at bucket *t*: `z_it = (NDVI_it − μ_{c(i),t}) / σ_{c(i),t}`, over other
sites in the same cluster observed in the same bucket, minimum 5 peers, field excluded.
**Why.** Absolute greenness is not comparable across agro-ecologies. Standardising against a
concurrent cohort removes what the cohort shares — season, weather regime, soil background — and
leaves what distinguishes a field from its neighbours, which is what a triage decision turns on.
**The cost, which is real:** it also cancels exactly the covariates the weather block supplies, since
a reanalysis cell covering several sites is precisely a quantity the cohort shares.
**Warrant.** Design choice with a stated cost.

### `forward_z` — the incumbent target
**Definition.** `z` at the next 30-day bucket. A guard rejects pairs more than 45 days apart; in the
committed panel the gap is exactly 30 days for all 4,596 rows.
**Warrant.** Measurement, reproducible from the panel.

### The field effect, and why the target was wrong
**Decomposition.** `z_{i,t+1} = α_i + γ_t + ε_{i,t+1}` — a time-invariant field effect, a cohort-date
effect, and the residual that weather and agronomy could explain.
**The defect.** Stage 1 removes `γ_t`. Nothing removed `α_i`, so it survived in the target — and the
"climatology" baseline, which is an expanding-window estimate of exactly `α_i`, beat the fitted
model by supplying the half the target left in.
**Measured.** Field-effect ICC **0.345**, 95% CI **[0.2357, 0.4353]** (1,000 site bootstraps).
Cluster ICC **0.0000**. Cohort-date ICC **0.0000**, confirming stage 1 works. A zero-parameter
expanding field mean reaches Spearman **+0.437** and out-of-sample R² **+0.199** against the fitted
40-feature model's **+0.398**.
**Warrant.** Measurement — `experiments/E01_variance_decomposition.out`, reproducible via
`argotech.lab.variance`.

### `alpha_hat` — the field-effect estimator
**Formula.** `α̂_it = w · (1/n_j) Σ_{s<t} z_is` with `w = n_j / (n_j + shrink)`, over observations
**strictly before** *t*, requiring `min_history` priors.
**Why shrink toward zero rather than a cluster mean.** `z` is already standardised within cluster and
date, so the cluster mean is ~0 by construction — and the measured cluster ICC is exactly 0.0000. A
cluster mean computed over the whole frame would also leak future observations into a quantity
defined as prior-only.
**Why NaN and not 0.0 when there is no reference.** A default of 0.0 asserts "exactly average" about
a field nothing is known of. That fabrication was present and shipped: with zero peer coverage it
made `ztilde == forward_z` exactly, so `within_*` results were `level_z` results under a `within`
label.
**Warrant.** Design choice + measurement.

### The within transformation, and Frisch–Waugh–Lovell
**Formula.** `z̃_{i,t+1} = z_{i,t+1} − α̂_it` (`within_y`); `within_xy` also demeans the features by
their own prior field means.
**Why both variants exist.** Residualising the target alone is **not** the within estimator. FWL
requires demeaning both sides, so the difference between the two is itself a measurable quantity:
how much of the covariates' apparent explanatory power was also a field effect.
**Warrant.** Frisch & Waugh (1933), *Econometrica* 1(4):387–401; Lovell (1963), *JASA*
58(304):993–1010. **[UNVERIFIED]** — standard attribution, not checked here.
Cross-sectional demeaning applied to crop-yield anomalies: *Forecasting Crop Yield Anomalies on
Panel Data via Spatially Demeaned Ensembles*, J. Agric. Biol. Environ. Stat. (2026),
doi:10.1007/s13253-026-00743-8. **[verified DOI; authorship UNVERIFIED]** — cited by title and DOI
deliberately rather than guessing the authors.

### The peer reference, and the three cohort keys
**Definition.** `leaky` = the panel's whole-frame baked column (the control). `cluster_month` =
fitted per fold, keyed `<cluster>|MM`. `geo_month` = fitted per fold, keyed on fixed latitude and
elevation bands plus month.
**Why fitted per fold.** Fitting over the whole frame leaks. **Measured:** 42 of 45 `cluster|MM`
buckets span multiple years; across the three forward-chaining boundaries **74.0% / 49.9% / 24.3%**
of a training row's peer cohort lies at or after the boundary. Under leave-one-cluster-out the
held-out cluster's bucket drew from exactly one cluster — itself.
**Why fixed band cut points and never quantiles.** A quantile fitted over the frame would put the
leak straight back in through the binning.
**The consequence, which is a finding.** An honest cluster-keyed fit gives a held-out cluster **no
reference at all** — coverage `[0.0, 0.0, 0.0, 0.0]`. It is not *worse* out-of-region, it is
*unevaluable*. A geographic key restores coverage `[1.0, 1.0, 1.0, 0.472]`.
**Warrant.** Measurement + design choice.

### `forward_z`'s tails, and what is and is not a fix
**Measured.** Range **−11.35 to +7.08**, sd 1.144, on a quantity that is a z-score by construction.
Arithmetic verified correct (independent recomputation, r = 0.998). Cohorts are *not* small or tight
(sizes ~32, sds 0.07–0.33).
**Cause.** Physically invalid NDVI. 233 rows (5.07%) have NDVI < 0; rows with |z| > 3 are **8×
enriched** for a negative label-date NDVI (39.7% against a 5.1% base rate).
**Why filtering is a validity fix and NOT a tail fix.** Simulated: excluding NDVI < 0 moves the
minimum −11.35 → −4.56 but pushes the **maximum** +7.08 → +7.59, because removing low members
shrinks the cohort sd. Overall sd barely moves (1.144 → 1.142).
**Why the remaining tails must not be filtered.** A field genuinely at NDVI 0.9 in a cohort
averaging 0.28 with sd 0.11 really is z = +5.6. That is an unusual field — which is precisely what a
triage system exists to find. Deleting it would delete the signal.
**What was actually changed** (`src/argotech/lab/panel.py`, commit `2d4a595`):
- `MIN_VALID_NDVI = 0.0` — applied to both the cohort and the label observation. Physics, not a
  tuning knob: NDVI ≤ 0 means NIR ≤ Red, which no canopy produces.
- `MIN_COHORT_SD = 0.005`, replacing a `1e-6` guard. Measured cohort sds run 0.07–0.33, so 0.005 sits
  over an order of magnitude below any real cohort while being ~5,000× stricter than the old value.
- **A separate bug fixed at the same site:** peers were excluded by *value*
  (`v != label_obs["ndvi"]`), so any peer that happened to share the field's NDVI was silently
  dropped, shrinking and biasing the cohort. Exclusion is now by site identity.

**The panel was deliberately NOT rebuilt.** See Part 9.

**Warrant.** Measurement + physical argument.

---

# Part 5 — Validation protocol

Implemented in `src/argotech/lab/splits.py`.

### Spatially blocked CV — leave-one-cluster-out
**Why.** Neighbouring fields share weather cells and satellite scenes, so a shuffled K-fold leaks and
inflates. The question the product needs answered is "does this work in a district we have never
seen", and random K-fold cannot answer it.
**What it does and does not control, measured.** The cluster ICC of `forward_z` is exactly 0.0000, so
blocking by cluster controls **feature distribution shift** and nothing about target structure. That
narrows what a blocked score is evidence *of*; it does not make the protocol wrong.
**Warrant.** Ploton et al. (2020), *Spatial validation reveals poor predictive performance of
large-scale ecological mapping models*, Nature Communications 11. **[verified]**.

### Forward chaining, purged and embargoed
**Definition.** Training rows take `label_date < boundary`; test rows take `obs_date >= boundary`.
Rows satisfying neither are **embargoed** and belong to neither side (52/31/69 per fold).
**Why.** Cutting the training side on the *prediction* date is the leak, not the fix. **Measured:**
the lag is exactly 30 days on all rows; cutting train on `obs_date < boundary` gives 2,931 rows of
which **121** have outcomes that postdate the first test prediction; cutting on `label_date` gives
2,810 with **zero**.
**Warrant.** Measurement. The purged/embargoed construction is standard in financial ML
(López de Prado 2018) **[UNVERIFIED]**.

### Fold boundaries computed on the raw panel
**Why.** Boundaries taken from the post-`build_target` frame depend on `min_history`, the target kind
and the peer key, so every configuration would get its own temporal partition and a comparison would
be measuring partitions rather than treatments. **Measured:** raw panel gives
`[2023-09-05, 2024-08-30, 2025-08-25]`; the target frame gives
`[2023-11-04, 2024-09-29, 2025-08-25]` — two of three differ.
**Warrant.** Measurement + argument.

### Area of Applicability
**Definition.** A dissimilarity index in predictor space marking the region where cross-validated
performance does not hold.
**Why.** It turns "a field outside every rectangle gets no reference" from a caveat into a computed
boolean, and it is what makes "route low-confidence predictions to a human" a rule rather than a
hope.
**Warrant.** Meyer & Pebesma (2021), *Predicting into unknown space?*, Methods in Ecology and
Evolution 12:1620–1633. **[verified]**.

---

# Part 6 — Metrics

Implemented in `src/argotech/lab/evaluate.py`.

### Net benefit / decision curve analysis
**Formula.** `NB(t) = TP/n − (FP/n) · (t / (1 − t))`
**Why.** `t` is the probability at which a visit becomes worthwhile, so `t/(1−t)` is exactly the
missed-outbreak-to-wasted-trip cost ratio. Reporting the curve across `t` replaces an unresolved
argument between macro-F1 and precision@k with a reported function, and makes the operating point a
product decision rather than a hidden constant.
**Its ceiling, which governs every comparison here.** The maximum attainable NB **is the event
rate**. Cells and folds with different prevalence are therefore *not on a common scale* — measured
event rates range 0.049 (Kano) to 0.488 (Benue), a factor of ten.
**Warrant.** Vickers & Elkin (2006), *Decision Curve Analysis: A Novel Method for Evaluating
Prediction Models*, Medical Decision Making 26:565–574. **[verified]**.

### Precision@k
**Definition.** Share of the top-k ranked cases that were events — an agent visits k farms this week.
**A defect fixed here.** On tied scores the top-k is an artifact of row order: a constant-predicting
arm scored P@25 up to 0.40 against event rates of 0.075–0.145. It now returns the prevalence when
the selection is arbitrary.
**Warrant.** Standard ranking metric; the tie handling is a design choice with a measurement behind
it.

### Bootstrap confidence intervals
**Definition.** Percentile CI over per-fold scores; separately, a site-level bootstrap resampling
sites and rebuilding folds.
**Why resample sites and never rows.** 39 monthly observations of one site are not 39 independent
facts.
**What the site-level interval does *not* answer.** It holds cluster identity fixed, so it measures
*within-cluster* variance — while transferability depends on *between-cluster* variance. Measured, it
is therefore **narrower** (0.0808) than the fold-level interval (0.1012), against a prediction that
it would be wider. It is correct and answers an easier question. Four clusters remain the independent
units.
**Warrant.** Efron (1979) for the bootstrap **[UNVERIFIED]**; the site-vs-row and within-vs-between
points are arguments, and the widths are measurements.

### ICC and variance components
**Formula.** One-way random effects, method of moments: `σ²_a = (MSB − MSW)/n₀`,
`ICC = σ²_a/(σ²_a + σ²_e)`, with `n₀` the unbalanced effective group size.
**Why not naive eta-squared.** A group mean over `n_i` observations carries `σ²_e/n_i` of pure noise,
which eta-squared credits to the group. Measured on this panel: eta-squared reports 0.360 where the
unbiased estimator reports 0.345.
**Warrant.** Standard variance-components estimation (Searle, Casella & McCulloch)
**[UNVERIFIED]**; the 0.360-vs-0.345 gap is a measurement.

### Expected Calibration Error
**Definition.** Binned gap between predicted confidence and realised accuracy.
**Why.** A risk number that is confidently wrong is worse than one that is uncertain and says so.
**Warrant.** Standard; **[UNVERIFIED]** for a specific source.

---

# Part 7 — Statistical concepts that govern the conclusions

### Multiple comparisons
**Why it dominates here.** Many cells were tested and some clear a 95% interval by luck.
**Measured.** Spatial: **165 cells tested, 8 significant-positive, ~4.1 expected by chance** with no
correction — a ratio of 1.9×. Temporal: **65 tested, 13 significant-positive, ~1.6 expected** —
8.0×. The spatial hits are additionally incoherent: significant at 500 m and 2000 m but not at the
1000 m between them.
**Conclusion it licenses.** Temporal skill is established; out-of-region skill is not distinguishable
from multiple-testing noise at four independent spatial units.
**Warrant.** Measurement + standard argument. No formal correction (Bonferroni, Benjamini–Hochberg)
has been applied — the raw ratio is reported instead, and a pre-registered comparison would be the
proper fix.

### Absence of evidence versus evidence of absence
**Why it is stated explicitly.** The replication's interval `[−0.0026, +0.0781]` includes zero while
95% of bootstrap draws are positive. That is an absence of evidence *for* an effect, not evidence of
*no* effect. At four independent units the distinction is real and must not be rounded in either
direction.
**Warrant.** Argument.

### Determinism, and why no seed sweep is possible
**Measured.** `max |pred(seed) − pred(42)|` across five seeds is exactly **0.000e+00** for both
arms. `linear` is a closed-form ridge solve; `boosted` has `early_stopping=False`, which removed its
only stochastic component, and `HistGradientBoostingRegressor` does not subsample for binning at
n ≈ 4,500.
**Why it matters.** Five seeds would return five identical numbers, and reporting that as
"replicated across five seeds" would imply five independent draws where there is one.
**Warrant.** Measurement.

---

# Part 8 — Model arms

Implemented in `src/argotech/lab/arms.py`. Every baseline is an arm rather than a special case,
because the argument turns on comparing against them honestly.

| Arm | What it is | Why it is here |
| --- | --- | --- |
| `zero` | predicts 0 | Under a correctly demeaned target this *is* climatology, by construction. |
| `persistence` | carries the current within-deviation forward | "Is this field weak *now*." Vegetation is strongly autocorrelated, so it is genuinely hard to beat. |
| `climatology` | each field's mean `ztilde` over training rows | "Is this field *usually* weak." A real fitted arm, not a constant — under `level_z` it is the baseline that beat the incumbent. |
| `linear` | ridge over standardised, median-imputed features | The cross-region literature finds simpler models transfer better under shift. A ceiling check, not a candidate. |
| `boosted` | `HistGradientBoostingRegressor`, `early_stopping=False` | One tree, not a stack: on tabular data this size a stack buys a fraction of a point and costs interpretability, latency and four times the retraining surface. |

**Why `early_stopping=False`.** sklearn's internal validation split is IID-random within the training
frame, so it can place the same site's temporally adjacent rows on both sides and select the stopping
iteration on leaked, autocorrelated signal — inside a protocol whose entire purpose is blocking. It
also makes the arm deterministic and the linear/boosted comparison like-for-like. The cost is no
iteration-level regularisation, which any comparison of the two must state.
| `linear_huber` | Huber loss in the same pipeline shape as `linear` | Robust to the heavy tails that survive the validity fix — those tails are signal and must not be filtered, so the question is whether a robust *loss* fits them better. |
| `boosted_abs` | `HistGradientBoostingRegressor(loss="absolute_error")` | The same question for the boosted arm. Absolute error weights an 11-sigma residual linearly rather than quadratically. |

Both were added **alongside** `linear` and `boosted`, never replacing them: whether a robust loss
helps is an empirical question, and replacing the arms would have destroyed the comparison that
answers it.

**Why NaN is routed and not filled** (boosted) **but imputed** (linear). Substituting a value claims
the field was observed. The boosted arm routes NaN down its own branch; the linear arm cannot, so it
imputes and is thereby *handicapped* — which makes a linear win more meaningful, not less.

### Conformal prediction (specified, not yet implemented)
**Why weighted quantiles rather than plain split conformal.** Blocked spatial folds and
forward-chaining temporal folds both violate exchangeability by construction, so standard split
conformal has no coverage guarantee under this protocol.
**Warrant.** Barber, Candès, Ramdas & Tibshirani (2023), *Conformal prediction beyond
exchangeability*, Annals of Statistics 51(2):816–845, doi:10.1214/23-AOS2276. **[verified]**.
Earth-observation application: arXiv:2401.06421. **[verified]**.

---

---

# Part 9 — Why the panel was not rebuilt after the validity fix

The NDVI validity fix changes how the label is *built*, so realising it requires regenerating
`data/training_set.parquet`. That rebuild is deferred, deliberately, and the reason is a caching
defect worth recording:

- The **Sentinel and SAR caches** are keyed `{site_id}-{days}` — date-independent, so a rebuild
  reuses the same observations.
- The **meteo cache** is keyed `{lat},{lon},{start},{end}`, with both dates derived from
  `date.today()`. The committed cache holds **at least five distinct windows**
  (2022-05-03/04/08/14 → 2026-08-04/05/09/15), which means the committed panel was assembled across
  several days and **sites do not share a common weather window** — this matters because
  `_climatological_rain_30` uses each site's full range.
- A rebuild on 2026-08-27 requests 2022-05-20 … 2026-08-21, matching none of them: 100% meteo cache
  miss, ~122 network calls (a backfill previously exhausted Open-Meteo's daily quota), and a
  different weather window that would **confound the NDVI change with a data-window change**.
- 38 committed result files pin `content_hash 6c832dc2…`. A rebuild invalidates all of them — which
  is the provenance system working, not a reason to avoid the rebuild, but a cost to plan for.

`collect_site` now takes an optional `end_date` so a future rebuild can be pinned and repeated. The
consequence to state plainly in any write-up: **every result in `experiments/` was computed on a
panel that still contains the invalid NDVI observations.** The validity fix applies to future builds.

# References

**Verified during this work** (checked against publisher or authoritative record):

- Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998). *Crop Evapotranspiration — Guidelines for Computing Crop Water Requirements*. FAO Irrigation and Drainage Paper 56.
- Barber, R.F., Candès, E.J., Ramdas, A., Tibshirani, R.J. (2023). Conformal prediction beyond exchangeability. *Annals of Statistics* 51(2):816–845. doi:10.1214/23-AOS2276
- Huete, A.R. (1988). A soil-adjusted vegetation index (SAVI). *Remote Sensing of Environment* 25:295–309.
- Huete, A. et al. (2002). Overview of the radiometric and biophysical performance of the MODIS vegetation indices. *Remote Sensing of Environment* 83:195–213.
- Kogan, F.N. (1990). Remote sensing of weather impacts on vegetation in non-homogeneous areas. *International Journal of Remote Sensing* 11:1405–1419. doi:10.1080/01431169008955102
- Meyer, H., Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods in Ecology and Evolution* 12:1620–1633.
- Ploton, P. et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications* 11.
- Vickers, A.J., Elkin, E.B. (2006). Decision curve analysis: a novel method for evaluating prediction models. *Medical Decision Making* 26:565–574.
- Wallin, J.R. (1962). Summary of recent progress in predicting late blight epidemics. Severity values from RH ≥ 90% duration and mean temperature; 18–20 SV spray threshold. BLITECAST: Krause, Massie & Hyre (1975), combining Wallin (1962) with Hyre (1955).
- *Forecasting Crop Yield Anomalies on Panel Data via Spatially Demeaned Ensembles*, J. Agric. Biol. Environ. Stat. (2026). doi:10.1007/s13253-026-00743-8 — **cited by title and DOI; authorship [UNVERIFIED]**.
- Tseng, G. et al. Presto: Lightweight, pre-trained transformers for remote sensing timeseries. arXiv:2304.14065.
- Uncertainty quantification for probabilistic ML in Earth observation using conformal prediction. arXiv:2401.06421.
- Bringing cross-validation into the real world to evaluate transferability of satellite-based vegetation models. *Scientific Reports* (2026).
- Comprehensive review of detrending methods for crop yields. *Field Crops Research* (2026).

**[UNVERIFIED] — asserted from memory or secondary sources, to be confirmed before submission:**

- Rouse et al. (1974) — NDVI.
- Gao, B.-C. (1996) — NDMI.
- Barnes et al. (2000) / Gitelson & Merzlyak — NDRE.
- Kim & van Zyl — Radar Vegetation Index.
- McMaster & Wilhelm (1997) — GDD computation variants.
- Frisch & Waugh (1933) *Econometrica* 1(4):387–401; Lovell (1963) *JASA* 58(304):993–1010 — FWL.
- Efron (1979) — the bootstrap.
- Searle, Casella & McCulloch — variance components.
- López de Prado (2018) — purged and embargoed cross-validation.
- Fall armyworm ~390 degree-days per generation, base 10 °C.
- Crop-specific phenology GDD thresholds; pollen-viability heat thresholds.
- IPCC AR5/AR6 — verified as a body of work; no specific chapter or page cited.
