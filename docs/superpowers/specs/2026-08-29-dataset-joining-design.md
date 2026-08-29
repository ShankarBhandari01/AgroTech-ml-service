# Joining the datasets: what may be combined, at which key, for which estimand

**Status:** design. Supersedes nothing; formalises rules that E01–E06 established by measurement.

## What this decides

The project now has three data sources that do **not** share a key or a spatial resolution. This
spec fixes (a) which tier each source belongs to, (b) which estimands each tier is *admissible*
for, and (c) the order operations must run in so no join reintroduces one of the three leaks
already found and fixed.

The governing claim: **combining these is not a feature concatenation.** A single wide table is the
failure mode, not the goal.

---

## 1. The three tiers

| Tier | Key | Cardinality | Contents | File |
| --- | --- | --- | --- | --- |
| **T1** field × time | `site_id` × `obs_date` | 4,596 rows, 122 sites, 4 clusters, 2022-09-10 → 2026-06-21 | 28 time-varying: weather, FAO-56 water balance, phenology, Sentinel-2 optical, Sentinel-1 radar | `data/training_set.parquet` |
| **T2** field, static | `site_id` | 122 rows | `clay sand silt soc phh2o cec bdod` (SoilGrids v2.0), `slope_percent`; `elevation`/`latitude`/`longitude` already inline in T1 | `data/site_covariates.parquet` |
| **T3** EA × season | EA id × season | not yet acquired | LSMS-ISA household and plot modules: assets, plot area, realised yields | World Bank microdata (see the E07/E08 plan) |

T1 carries `label_date = obs_date + 30 days`. That 30-day lead is the forecast horizon and it sets
the embargo width in §5.

---

## 2. The rule: the estimand decides admissibility

A column is admissible for an estimand only if the estimand has not already annihilated the
variation that column could explain. This is not a modelling preference; for T2 it is arithmetic.

Let `x_i` be a T2 column: constant within field `i` by construction (soil and slope do not change
over the panel window).

| Estimand | What happens to a T2 column | Admissible? |
| --- | --- | --- |
| `level_z` | untouched; explains between-field variation | **Yes — this is T2's only home** |
| `within_y` | survives in `X`, but `y` has had its field mean removed, so the only component `x_i` could explain is now identically zero | **No** — it can fit noise and nothing else |
| `within_xy` | demeaned by field → exactly zero → dropped | **No** — mechanically impossible |
| `delta_z` | differences across time cancel a constant | **No** |

`within_xy` makes this visible; the runner prints it on every run:

```
[targets] within_xy: dropped time-invariant column(s) (demean to zero): elevation
```

**Consequence.** Soil and terrain can only ever address the **between**-field component — the field
effect, measured at 34.5% [0.2357, 0.4353] of variance and found entirely within-cluster
(between-cluster ICC 0.0000). That is exactly E06's question, and it is the only question static
data is capable of answering. No amount of soil data will improve within-field ranking.

**The `within_y` row is the trap.** Nothing errors, nothing is dropped, the model trains, and a
metric comes back. A T2 column under `within_y` is a pure overfitting surface. Excluding it must be
explicit, not left to the demeaning to catch — because under `within_y` the demeaning does not
catch it.

---

## 3. T1 ⋈ T2 — the only field-level join

```
panel = T1.merge(T2, on="site_id", how="left", validate="many_to_one")
```

Required, in this order:

1. `validate="many_to_one"` — a duplicated `site_id` in T2 silently multiplies panel rows. Assert it.
2. **Reconcile `cluster`.** It appears in both. Assert equality and drop T2's copy; a mismatch means
   the two files were built from different site lists and the join is meaningless.
3. **Assert no row loss and record the null rate.** SoilGrids has genuine gaps (water bodies,
   incomplete depth stacks). A left join turns a missing site into silent NaNs; the count belongs in
   the run's provenance block, not in a log line.
4. **Tag the joined columns as static** so §2's admissibility gate can act on them by provenance
   rather than by name matching.

T2 must **not** be merged into `data/training_set.parquet` on disk. The 38 committed experiment
provenance hashes are content hashes of that file; rewriting it invalidates every one. Join at use
time, in memory. This mirrors the rule the E07/E08 plan already states.

---

## 4. T3 — never joins at field resolution

Public LSMS-ISA releases enumeration-area centroids, displaced 0–5 km (rural, 99%) or 0–10 km (1%).
T1's satellite features use a **±500 m AOI half-width** (`d = 0.005` degrees, `data/sentinel.py:204`)
— a ~1.1 km box. Rural displacement reaches 5 km: **ten times that half-width, and roughly 64× the
box in area** (a 5 km disc is 78.5 km2 against 1.2 km2). The pixel at a published coordinate is in
general not that household's pixel.

Therefore:

- **Forbidden:** joining a T3 household to a T1 site, or to any pixel, by nearest neighbour or radius.
  A row produced this way is noise wearing a farm's name.
- **Required:** aggregate T1 upstreams to the EA scale *first*, then join T3 to that. The unit of
  analysis becomes the EA-season, not the field-day.

The EA-scale panel is a separate build with its own key (`ea_id × season`), not a variant of T1. It
re-fetches the same weather and Sentinel upstreams over the EA's plausible support rather than a
500 m box. Read *Privacy Protection, Measurement Error, and the Integration of Remote Sensing and
Socioeconomic Survey Data* (arXiv:2202.05220) before implementing it; the error is quantified there.

**Wave 5 caveat:** GHS-Panel Wave 5 (post-planting Jul–Sep 2023, post-harvest Jan–Mar 2024) overlaps
T1's window, but its geovariables are listed as *forthcoming*. Without them Wave 5 supports the
non-spatial half (household attributes → exposure and vulnerability) and nothing spatial.

### 4a. The EA aggregation, specified

**Displacement is not uniform across upstreams.** It degrades each in proportion to that upstream's
native resolution, so the EA panel is not uniformly damaged and the surviving features must be
labelled as such:

| Upstream | Native resolution | 0–5 km displacement is | Consequence |
| --- | --- | --- | --- |
| ERA5 (Open-Meteo archive) | 0.25° ≈ 28 km (ERA5-Land 0.1° ≈ 9 km) | **sub-pixel** | displaced point usually falls in the same grid cell; weather features survive nearly intact |
| SoilGrids v2.0 | 250 m | 20× | aggregate over the support; soil varies smoothly, so the loss is modest |
| Sentinel-2 L2A | 10–20 m, AOI ±500 m | 10× the half-width | destroyed at pixel scale |
| Sentinel-1 GRD | 20 m | same | destroyed at pixel scale |

The practical reading: the weather half of `assess_hazard` — which is what drives the drought,
heat and disease sub-hazards — largely survives the move to EA scale. The canopy half does not.

**Step 1 — the support.** Displacement is a uniform random direction with distance uniform in
[0, R] (R = 5 km rural, 10 km for 1%; 2 km urban), **constrained to remain inside the survey's
administrative unit**. So the support is not a disc:

```
support = disc(published_centroid, R) ∩ admin2_polygon
```

The intersection is load-bearing near a boundary, where it removes a large share of the naive disc.

**Step 2 — the weighting is not uniform over area.** With `d ~ U(0,R)` and angle uniform, the joint
density in polar coordinates is `1/(2πR)`, and the area element is `r dr dθ`, so the density **per
unit area** is `1/(2πRr) ∝ 1/r`. Pixels must be weighted by `1/r` (capped as `r → 0`), not equally.
A naive buffer mean over-weights the periphery, which is where most of the area lies. This step is
what separates a defensible aggregate from a buffer average.

**Step 3 — mask to cropland before averaging.** A 5 km disc contains towns, roads, water, bare
ground and forest. A mean NDVI over it measures **land cover, not crop condition**. Mask to cropland
(ESA WorldCover class 40 or equivalent) first. Omitting this is the most likely route to a
confidently meaningless EA panel.

**Step 4 — compute per pixel, then aggregate; never the reverse.** NDVI is a nonlinear ratio, so
`mean(NDVI) ≠ NDVI(mean NIR, mean Red)` by Jensen's inequality, and the bias does not vanish with
more pixels. The same holds for `water_satisfaction`. **Run-length statistics cannot be averaged at
all:** `dry_spell_30` is the longest run below a threshold and is path-dependent — averaging the
series first and then computing runs yields a systematically shorter spell than computing runs per
cell and averaging. Moot for weather (sub-pixel), but binding for any downscaled product.

**Step 5 — the temporal key changes meaning.** T1 is a 30-day-lead nowcast
(`label_date = obs_date + 30d`). LSMS yields refer to a whole season. The EA panel is therefore not
a re-slice of T1: features become season-integrated (cumulative rain, GDD sum, NDVI integral and
peak, dry-spell count). For Wave 5 the season is bracketed by the post-planting visit (Jul–Sep 2023)
and the yield recall at post-harvest (Jan–Mar 2024).

**Step 6 — expect a power collapse.** ~5,000 households at roughly 10 per EA gives **n ≈ 500
EA-seasons per wave**, against T1's 4,596 field-months — an order of magnitude fewer rows, and
without the within-field variation that carries all measured skill. Block by EA and by state.
Given that only 8 of 165 spatial cells cleared zero against ~4.1 expected by chance, this panel
should be assumed underpowered for anything subtle: E08 must answer **one pre-registered question**,
not sweep a matrix.

### 4b. E09 has reported — the per-upstream predictions above were half wrong

The table in 4a predicted weather would survive because displacement is "sub-pixel", and that
optical and radar would both be "destroyed at pixel scale". Measured, at R = 5 km:

| Upstream | Footprint | within-site retention | verdict |
| --- | --- | --- | --- |
| ERA5-Land weather | 68.5 km2 | 0.93–0.99 (heat_stress_days **0.34**) | survives, except the onset chain |
| Sentinel-1 radar `vh` | 1.23 km2 | **0.843** | survives far better than predicted |
| Sentinel-1 `vv` / `rvi` | 1.23 km2 | 0.796 / 0.677 | survives |
| Sentinel-2 `ndvi` | 1.23 km2 | **0.435** | badly degraded |
| Sentinel-2 `ndwi` | 1.23 km2 | 0.340 | badly degraded |

**Correction 1 — weather does not survive because displacement is sub-pixel.** The grid is
7.83 x 8.75 km, not the 0.25 deg / 28 km assumed, and **38.3% of draws change cell** (measured over
4,880 draws). Weather survives because most of its features are smooth aggregates. The exception
proves it: `heat_stress_days` collapses to **0.34**, because `season_onset_index` is a THRESHOLD
detection whose flip cascades through phenology stage into the flowering window it gates.

**Correction 2 — footprint does not govern the damage; spatial correlation length does.** Optical and
radar share an identical 1.23 km2 AOI and an identical 75.1% zero-overlap geometry, yet retain 0.435
and 0.843. Footprint cannot explain a gap that large. NDVI at 10 m reflects field-level MANAGEMENT --
planting date, crop choice, inputs -- which decorrelates within a few hundred metres; C-band
backscatter is driven by terrain, canopy structure and soil moisture, which are landscape-scale.
This was verified against the obvious confound: `gap_days` is 0 for every pair in both modalities, so
the two are matched on exact dates.

**Correction 3 — 4a's acceptance gate is void.** It required "weather-only retention ~1.0 or reject
the harness". That was written on the 28 km premise; `heat_stress_days` fails it and everything else
passes. Replace it with the per-feature table above.

**What this means for E08, stated before any LSMS file is acquired:**

* **Viable at EA scale:** the drought sub-hazard (`water_satisfaction_30` 0.979, `dry_spell_30`
  0.975 among moved draws) and the Sentinel-1 radar block. Build the EA panel on weather + radar.
* **Not viable:** anything phenology-gated (`stage_kc`, `water_deficit_30`, heat-during-flowering),
  and the optical canopy block at 0.435 with nRMSE ~1.0.
* **The structural limit, which no feature choice fixes:** this project's measured skill rests on the
  **field effect** -- 34.5% of variance, entirely WITHIN cluster, and E05 showed removing it removes
  the temporal skill with it. A field effect is a per-field constant. A displaced coordinate points
  at a DIFFERENT field, so it cannot carry that field's effect at all. E08 can therefore test
  landscape-scale relationships; it cannot test the thing this repository has actually measured.

**Step 7 — run the placebo first (E09).** The ceiling on all of the above is measurable *before*
acquiring any LSMS file, by displacing the 122 known-coordinate sites with the same distribution and
rebuilding features. See `docs/superpowers/plans/2026-08-29-E09-displacement-placebo.md`. **E09 gates
E08:** if the known signal does not survive simulated displacement, E08 cannot work as designed.

---

## 5. Ordering, so joins do not reintroduce a leak

All three leaks found in this project came from a reference statistic computed over rows the model
should not have seen. A join is an opportunity to repeat that. The order is fixed:

1. **Split first.** Spatially blocked (leave-one-cluster-out) and/or purged forward chaining:
   train on `label_date < boundary`, test on `obs_date >= boundary`, with a 30-day embargo equal to
   the label lead.
2. **Fit every reference on the training side only** — `fit_peer_stats`, `alpha_hat`, and any
   standardisation of T2 or T3 columns. Apply to test; never refit.
3. **Join T2** (static, no fitted quantity — safe at any point, but do it after the split so any
   later standardisation of it inherits the fold).
4. **Derive the estimand**, which applies §2's gate.

A T2 column standardised over the whole frame is the same defect as the peer-standardisation leak,
in a new place: the test fold's soil values would inform the training fold's mean.

---

## 6. Which tier serves which risk term

```
risk = Hazard × Exposure × Vulnerability
         │         │            │
      T1 (+T2      T3 + backend farmer payload
      between-     ────────────┬───────────────
      field        the two terms with no data today
      baseline)
```

This is the actual integration design: **each source feeds the term it can address at the resolution
it has.** T3's value is not that it improves hazard — it cannot reach field resolution — but that it
is the only available evidence for exposure and vulnerability, which today are unvalidated.

Weighted by measured impact, that is also where the value is: `expected_loss` swings ~$3,531 across a
plausible exposure range against $450 for the full observed hazard range and $225 for vulnerability
(notebook 05, §4). And §9.3 of `model-design.md` showed hazard pinned at exactly 1.0 for 59.6% of
the panel, so for most fields the ranking is already driven by exposure regardless.

---

## 7. Acceptance checks

Each must fail if the rule it names is broken. A test that passes against broken code is worse than
no test — four did so earlier in this project.

- [ ] T2 column present in a `within_xy` design matrix → the run fails, not warns.
- [ ] T2 column present under `within_y` → the run fails. **Assert this with an arm that actually
      reads `test[features]`**; a `zero` baseline never evaluates its features and will pass
      regardless.
- [ ] Duplicate `site_id` in T2 → the merge raises, not silently fans out rows.
- [ ] `cluster` disagreement between T1 and T2 → raises.
- [ ] Peer/reference statistics fitted after a T2 join produce **identical** values to fitting before
      it — proving the join introduced no cross-fold information.
- [ ] Any nearest-neighbour join from T3 to a field or pixel → rejected in review by §4.

## 8. Open

- The EA-scale rebuild (§4) is unspecified beyond its key and its prohibition; it needs its own
  design once Wave 5 geovariable availability is known.
- Whether `level_z` remains a reported estimand at all. E05 found that removing the field effect
  removes the temporal skill with it, so `level_z` results are largely a restatement of the field
  effect — which is the thing T2 is being asked to explain. If E06 explains it, `level_z` becomes
  interpretable; if not, `level_z` skill should be read as "the model memorised which fields are
  bad", not as agronomic prediction.
