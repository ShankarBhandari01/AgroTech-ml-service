# E06 — Does static soil explain the field effect?

**The question.** E01 measured a per-field effect at **34.5%** of `forward_z` variance
(CI [0.236, 0.435]). E05 showed, from the opposite direction, that removing it removes essentially
all the predictable signal: under `level_z` (effect retained) temporal boosted takes
+0.0194 [+0.0182, +0.0216]; under `within_xy` (effect removed) every learned arm sits at or below
zero. So the field effect is not one term among many — **at this resolution it is the signal**, and
it is currently an unexplained per-site constant.

This experiment asks what it *is*.

## Why soil, and why this is well-posed

Measured for this plan, on `data/training_set.parquet`:

| | value |
| --- | --- |
| sites with an estimable field effect | 122 |
| sd of the field effect across sites | **0.5086** |
| **between-cluster share (ICC)** | **0.0000** |
| within-cluster sd | 0.5135 |
| cluster means | +0.015, +0.028, +0.063, −0.040 |

**The field effect is entirely within-cluster.** It is not regional or climatic — it is what
separates *neighbouring* sites inside the same 120–180 km bounding box. That rules out a whole class
of explanations and points at one:

| candidate | varies within a cluster? | time-invariant? | matches? |
| --- | --- | --- | --- |
| weather / climate | barely — and the target already cancels what the cohort shares | no | ✗ |
| **static soil** (texture, organic carbon, pH, drainage) | **strongly, at field scale** | **yes** | **✓** |
| soil *moisture* (SMAP) | yes | **no — it is dynamic** | ✗ see below |
| management, variety, planting date | yes | partly | ✓ but unobserved here |

**Soil moisture is the wrong instrument for this question**, and the distinction is the crux of the
design: the field effect is *time-invariant by construction*, so only a time-invariant covariate can
explain it. A dynamic series like SMAP addresses the residual `ε_it`, which is a different
experiment (and one E02–E05 suggest is not learnable at four spatial units).

**What the panel already explains: nothing.** The only static covariates present are elevation and
coordinates. Within cluster:

```
elevation : r = +0.129  (r² = 0.017)
latitude  : r = −0.141  (r² = 0.020)
longitude : r = −0.184  (r² = 0.034)
```

So ~97% of the field effect is unexplained by anything in the panel today.

## Design

**Unit of analysis is the SITE, not the row.** n = 122. Monthly observations of one site are not
independent, and this is a question about a per-site constant. Anyone reporting n = 4,596 here has
made the error this repository has already been bitten by.

**Target:** `α̂_i`, the site's estimated field effect (`lab.targets.alpha_hat`, min_history ≥ 5).

**Covariates:** SoilGrids v2.0 at 250 m — clay, sand, silt, SOC, pH, CEC, bulk density, and depth to
bedrock, at 0–5 cm and 5–15 cm. Static, global, free, and finer than anything else in the panel.

**Estimation:** regress `α̂_i` on soil **with cluster fixed effects**, since the effect is entirely
within-cluster and a pooled regression would let between-cluster variation that is already ~0 do
imaginary work.

**Report:** within-cluster R², per-covariate contribution, and a bootstrap interval over **sites**
— never rows.

### Tasks

- [ ] **1. Fetch.** SoilGrids REST API per site, cached under `.cache/soilgrids/` keyed
      `{site_id}` (static data, so no date in the key — unlike the meteo cache, whose `date.today()`
      key is what currently makes a panel rebuild non-deterministic). Refuse to memoise an empty
      response, exactly as `_cached_fetch` already does.
- [ ] **2. Join and describe.** One row per site. Report coverage — a site with no SoilGrids
      response is a hole, not a zero.
- [ ] **3. Regress**, with cluster fixed effects, and report within-cluster R² with a site-level
      bootstrap interval.
- [ ] **4. Compare against the null**: the r² ≈ 0.017–0.034 that elevation and coordinates already
      give. Soil must beat that, or it adds nothing.
- [ ] **5. Write it up in `experiments/E06-soil/`** with a manifest hash, git SHA and seed, like
      every other experiment here.

## What each outcome means — decided in advance

| result | reading | what follows |
| --- | --- | --- |
| Soil explains a **substantial** share, interval excluding zero | the field effect is largely soil | add soil to the feature set; re-run E02. The estimand becomes learnable because the dominant term is now observed rather than absorbed into a constant. |
| Soil explains **little**, interval crossing zero | the field effect is *not* soil | it is management, variety, planting date, or irrigation — **unobservable from remote sensing**, and only outcome capture (`POST /outcomes`) can reach it. That would be a strong negative result and it redirects effort away from more covariates. |
| Soil explains a **moderate** share | partial | report the share honestly; do not oversell a partial explanation as a mechanism. |

**Pre-registering these readings matters here.** E02–E05 tested 165 spatial cells and found 8
significant-positive against ~4.1 expected by chance. This experiment is one hypothesis, one target,
one estimator, and the interpretation is fixed before the number is seen.

## Threats to validity, stated now

- **n = 122 sites in 4 clusters.** With cluster fixed effects the effective degrees of freedom are
  lower still. A null result may be low power rather than no effect, and the interval must be
  reported so a reader can tell.
- **Resolution.** SoilGrids at 250 m is finer than the 500 m Sentinel box and far finer than the
  reanalysis cell, but a smallholder plot is still smaller. This tests soil-as-measured, not
  soil-as-experienced-by-the-plant.
- **The sites are lattice points, not farms.** A soil–field-effect relationship at Halton-sampled
  points need not hold on real plots, where management is chosen partly *because* of the soil.
- **`α̂` is itself estimated**, and noisily at short histories. Use `min_history ≥ 5` and report
  sensitivity, as E03 did.
- **Reverse causation is possible and unresolvable here.** Farmers choose crops and inputs partly by
  soil, so a correlation does not separate "soil drives canopy" from "soil drives management drives
  canopy." Say so rather than implying a mechanism.

## Explicitly not in scope

- SMAP / dynamic soil moisture — wrong instrument, see above.
- CHIRPS rainfall — also dynamic, and the target already cancels what a cohort shares.
- Any change to the serving path. This is a research question; nothing ships from it.
