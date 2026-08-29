# Data dictionary

Every column in `data/training_set.parquet` (35 columns, 4,596 rows, 122 sites) and
`data/site_covariates.parquet` (one row per `site_id`, 122 sites). Formulas and the exact
file:line each column is computed at are in `docs/FACTS.md` §2.1 (panel) and are **not**
repeated here — this table is the per-column reference `FACTS.md` doesn't have: name, type,
units, source, how derived, gotchas.

Both files key on `site_id`. `site_covariates.parquet` is deliberately **not** merged into
`training_set.parquet` — see "Why a separate file" below.

## Why a separate file

`data/training_set.parquet`'s `content_hash` (`lab/panel/panel.py:manifest`) is pinned in the
provenance block of 38 committed experiment result files. Appending soil/slope columns to that
parquet would change the hash, and every one of those 38 files would then cite a dataset that no
longer exists on disk under the hash they recorded — exactly the failure the provenance system
was built to catch. `site_covariates.parquet` is a separate, site-keyed file that an experiment
joins in explicitly (`pd.merge(panel, covariates, on="site_id")`), so existing provenance blocks
stay valid and new experiments opt in.

---

## 1. `data/training_set.parquet`

### 1.1 Thermal

| Column | Type | Units | Source | Gotchas |
| --- | --- | --- | --- | --- |
| `gdd_90` | float64 | °C·day, summed over 90 days | Open-Meteo ERA5 archive (`Tmax`, `Tmin`) | Base 10 °C, cap 30 °C (maize). FACTS §2.1. |
| `gdd_since_onset` | float64 | °C·day | derived from `gdd_90`'s daily series | 0 before season onset. |
| `tmax_mean_30`, `tmin_mean_30` | float64 | °C | ERA5 `temperature_2m_max`/`min` | 30-day means. |
| `diurnal_range_30` | float64 | °C | ERA5 | `(ΣTmax − ΣTmin)/n` over 30 days, not a per-day mean of daily ranges. |
| `heat_stress_days` | int64 | count (0–30) | ERA5 `Tmax` | Counted **only** during the `Flowering` phenology stage; 0 at every other stage regardless of actual heat. |

### 1.2 Water

| Column | Type | Units | Source | Gotchas |
| --- | --- | --- | --- | --- |
| `rain_30`, `rain_90` | float64 | mm, summed | ERA5 `precipitation_sum` | |
| `et0_90` | float64 | mm, summed | ERA5 `et0_fao_evapotranspiration` | If the ET0 series is wholly absent from a response, `data/meteo.py:has_all_variables` rejects the site rather than substituting zeros — a missing ET0 read as 0 previously moved `water_satisfaction_30` from 0.221 to 1.000 and flipped the dominant hazard from drought to disease under a 200 OK. |
| `water_satisfaction_30` | float64 | ratio, clipped to [0, 1] | derived | `min(1, Σrain / (Kc·Σet0))`; `Kc` from `stage_kc`. |
| `water_deficit_30` | float64 | mm | derived | `max(0, Kc·Σet0 − Σrain)`. |
| `dry_spell_30`, `dry_spell_90` | int64 | days | ERA5 `precipitation_sum` | Longest run with rain < 1.0 mm/day. |
| `rain_anomaly_30` | float64 | mm | ERA5 + site's own climatology | `Σrain_30 − clim_rain_30`; climatology excludes the sample's own year. |

### 1.3 Atmosphere

| Column | Type | Units | Source |
| --- | --- | --- | --- |
| `rh_mean_30` | float64 | % | ERA5 `relative_humidity_2m_mean`, 30-day mean |
| `radiation_90` | float64 | MJ/m², summed | ERA5 `shortwave_radiation_sum` |

### 1.4 Phenology

| Column | Type | Units | Source | Gotchas |
| --- | --- | --- | --- | --- |
| `days_since_onset` | int64 | days | derived from rainfall | Sivakumar (1988) onset rule. `0` both for "onset today" and "no onset found in the window" — the two are not distinguishable from this column alone. |
| `stage_kc` | float64 | unitless (FAO-56 crop coefficient) | derived from `gdd_since_onset` | Discrete lookup over 6 stages (0.35 … 1.20 … 0.30), not a continuous function. |

### 1.5 Canopy state — optical (Sentinel-2 L2A, CDSE Statistical API)

| Column | Type | Units | Source | Gotchas |
| --- | --- | --- | --- | --- |
| `ndvi` | float64 | index, theoretical range [−1, 1] | `(B08−B04)/(B08+B04)`, server-side aggregate over a P30D window, `d=0.005°` AOI | **NDVI ≤ 0 is not vegetation** — water, cloud, shadow, or a no-data sentinel, never a low-vigour crop. 233 of 4,596 rows (5.07%) are < 0, two exactly −1.0 (NIR = 0). Cited: `docs/FACTS.md` §7.8. |
| `ndmi` | float64 | index, [−1, 1] | `(B08−B11)/(B08+B11)` | **This is NDMI (canopy moisture), not McFeeters' NDWI** (a water-body index). The repo's Sentinel client still calls the raw field `ndwi` server-side and renames it on ingest (`features/agronomic.py`, `domain/indices.py:26-28`) — the column name here is already corrected. |
| `evi` | float64 | index | `2.5(B08−B04)/(B08+6B04−7.5B02+1)` | Soil/aerosol-corrected; preferred over NDVI on dense canopy. |
| `vci` | float64 | %, nominally [0, 100] but not clamped | Kogan (1990), `(NDVI−min)/(max−min)·100` over **this site's own prior observations** | Uses only observations strictly before `t` — no lookahead. Can exceed [0,100] if the current value breaks the site's own historical range. |
| `ndvi_z_peer` | float64 | z-score, unbounded | `peer_z` against `{cluster × calendar month}` fitted (μ, σ) | See "Two peer mechanisms" below — this is **not** the same reference `forward_z` uses. |

### 1.6 Canopy structure — radar (Sentinel-1 GRD)

| Column | Type | Units | Source | Gotchas |
| --- | --- | --- | --- | --- |
| `rvi` | float64 | index | Radar Vegetation Index from VV/VH | NaN when no S1 observation falls within 20 days of the optical sensing date. |
| `vh_vv_ratio` | float64 | ratio | `VH/VV` | NaN when `VV == 0`. **Averaged in linear power, not dB** — CDSE returns linear backscatter deliberately, because averaging dB values averages logarithms, not power. `docs/FACTS.md` line 41; contrast with the Presto embedding path (`lab/presto/embeddings.py:92-96`), which converts to dB only there, for the pretrained encoder's own normalisation. | 
| `rvi_z_peer` | float64 | z-score, unbounded | `peer_z` against `{cluster × month}` | Same caveat as `ndvi_z_peer`. |

### 1.7 Site / identity

| Column | Type | Units | Source | Gotchas |
| --- | --- | --- | --- | --- |
| `latitude`, `longitude` | float64 | degrees (WGS84) | site sampling grid | Coverage: lat −0.178…12.566, lon 7.315…36.052 (`docs/FACTS.md` §1.5). |
| `elevation` | float64 | meters | Open-Meteo response's own `elevation` field (the archive/forecast grid-cell elevation, not a dedicated DEM query) | Range 60–2837 m across the panel. |
| `site_id` | object (string) | — | sampling grid identity, e.g. `Kaduna_Grain_Belt-000` | Join key to `site_covariates.parquet`. 122 distinct values over 4,596 rows. |
| `cluster` | object (string) | — | hand-drawn bounding box name, **not** an administrative unit | 4 clusters present in this file: `Kaduna_Grain_Belt`, `Kano_Sudan_Savannah`, `Kenya_Rift_Valley`, `Benue_River_Basin` (32/32/31/27 sites — see §1.5 below and `docs/FACTS.md` §1.5 for the 3 more clusters defined in code but absent here). |
| `obs_date` | object (string, ISO date) | — | feature/prediction date | Range 2022-09-10 … 2026-06-21. |
| `label_date` | object (string, ISO date) | — | outcome observation date | **Exactly 30 days after `obs_date` for all 4,596 rows** (std 0.0) — verified by direct computation on the committed file. |

### 1.8 Label / target

| Column | Type | Units | Source | Gotchas |
| --- | --- | --- | --- | --- |
| `forward_z` | float64 | z-score by construction | `(NDVI_field(t+30) − peer_mean) / peer_sd`, peers = same cluster, same `label_date`, ≥5 peers, `sd ≥ 1e-6` | **Ranges −11.35 to +7.08**, sd 1.144, despite being a z-score. Cause is not degenerate cohorts (sizes ~32, sd 0.07–0.33) — it's invalid NDVI entering the cohort/label (see `ndvi` gotcha above). Rows with `\|z\| > 3` are **8× enriched** for a negative label-date NDVI (39.7% vs. a 5.1% base rate). `docs/FACTS.md` §7.8. This is the training target for the production regressor. |
| `label` | int64 | {0, 1, 2} | thresholded from `forward_z`: `2` if `z ≤ −1.0`, `1` if `z ≤ −0.35`, else `0` | Retained for the classifier arm and classification metrics only; production regresses `forward_z` directly. |

**Two distinct peer mechanisms** (`docs/FACTS.md` §3.3) — do not conflate:
(a) the **label** cohort for `forward_z` is cluster × *exact sensing date*, cross-sectional, min 5 peers;
(b) the **feature** peer reference for `ndvi_z_peer`/`rvi_z_peer` is cluster × *calendar month*, from fitted (μ, σ) constants pooled over the whole training frame.

---

## 2. `data/site_covariates.parquet`

One row per `site_id` (122 sites, from `data/training_set.parquet`'s own `site_id`/`latitude`/
`longitude`/`cluster` — not re-derived from `sample_sites`). A missing covariate is an absent
column value for that row (NaN on read via `pd.read_parquet`), never a fabricated zero.

### 2.1 Identity

| Column | Type | Units | Source |
| --- | --- | --- | --- |
| `site_id` | object (string) | — | copied from the panel; join key |
| `cluster` | object (string) | — | copied from the panel, carried through only for coverage reporting |

### 2.2 Soil — SoilGrids v2.0 (ISRIC), 250 m, static

Endpoint `https://rest.isric.org/soilgrids/v2.0/properties/query`, `value=mean`, at depths
`0-5cm` and `5-15cm`. 14 columns, named `{property}_{depth}`:

| Property | Column examples | Natural unit | Raw `mapped_units` | `d_factor` (this run) | Meaning |
| --- | --- | --- | --- | --- | --- |
| `clay` | `clay_0-5cm`, `clay_5-15cm` | % | g/kg | 10 | Clay fraction |
| `sand` | `sand_0-5cm`, `sand_5-15cm` | % | g/kg | 10 | Sand fraction |
| `silt` | `silt_0-5cm`, `silt_5-15cm` | % | g/kg | 10 | Silt fraction |
| `soc` | `soc_0-5cm`, `soc_5-15cm` | g/kg | dg/kg | 10 | Soil organic carbon |
| `phh2o` | `phh2o_0-5cm`, `phh2o_5-15cm` | pH (unitless) | pH×10 | 10 | pH in water |
| `cec` | `cec_0-5cm`, `cec_5-15cm` | cmol(c)/kg | mmol(c)/kg | 10 | Cation exchange capacity |
| `bdod` | `bdod_0-5cm`, `bdod_5-15cm` | kg/dm³ | cg/cm³ | **100** | Bulk density |

**Gotcha — SoilGrids returns scaled integers, and the scale is not the same for every property.**
A probe against this API returned `clay=159` for a real Kaduna site, which is 15.9%, not 159%. The
builder (`src/argotech/lab/covariates/site_covariates.py:layer_values`) converts with
`value / unit_measure.d_factor`, read from each property's own response — **never a hardcoded
`/10`** — because `bdod`'s `d_factor` is 100 while every other requested property here uses 10; a
hardcoded divisor would have silently under-converted bulk density by 10×. Confirmed against the
live API response for all 7 properties (table above) at the time this file was built.

### 2.3 Terrain — slope

| Column | Type | Units | Source | Gotchas |
| --- | --- | --- | --- | --- |
| `slope_percent` | float64 | **percent grade** (rise/run × 100) | Open-Meteo's dedicated elevation endpoint (`api.open-meteo.com/v1/elevation`), sampled at 4 points (N/S/E/W) ±0.005° around the site | Gradient magnitude via central difference: `dz/dlat` and `dz/dlon` converted from degrees to meters (1° lat = 111,320 m; 1° lon = 111,320 m × cos(lat)), combined as `hypot(dz/dlat, dz/dlon) × 100`. ±0.005° is the same half-width as the Sentinel-2 AOI bbox (`src/argotech/data/sentinel.py:204`), so slope is measured over the same footprint the canopy indices are aggregated over. **This is a real, computed value — every prior Presto embedding was told every site is flat**: `lab/presto/embeddings.py:170` sets `x[:, IDX_SLOPE] = 0.0` unconditionally ("no slope upstream; 0 == flat after /50"). This file does not change that — `embeddings.py` is deliberately not modified here; wiring slope into Presto is a separate change. |

The `elevation` field already used elsewhere in this repo (`elevation = payload["elevation"]`,
`lab/panel/panel.py:230`) is the archive/forecast **grid-cell** elevation returned incidentally
by the weather endpoints, not a dedicated terrain query — that is a different quantity from the
4-point DEM sample used here for slope, though both ultimately come from Open-Meteo.

### 2.4 Caching and coverage

Cached under `.cache/soilgrids/`, keyed **`{site_id}-{property}.json`** for soil (one file per
property, both depths) and **`{site_id}-slope.json`** for the elevation grid — no date in either
key, since both are static. This directory is shared with a concurrent SoilGrids fetch running in
this repository at the same time (experiment E06): reading through the same per-property key
means either process's successful fetch is available to the other, without either one
overwriting or deleting the other's entries. Writes go through a temp-file-then-rename so a
half-written file is never observed mid-read.

An empty or failed response is never cached — same rule as `lab/panel/panel.py:_cached_fetch`:
caching an indistinguishable transient failure is what turned a burst of CDSE 429s into 69
permanently band-less sites in the panel's own Sentinel fetch. A miss costs a retry on the next
run; a false cache hit costs a silently corrupted dataset.

**Coverage is reported by the builder at run time** (`site_covariates.print_coverage`), including
a per-cluster breakdown of misses and an explicit flag if a gap is cluster-structured — i.e. an
entire cluster has zero coverage for a covariate, which is the one gap shape that would make
"held-out cluster" and "missing covariate" the same event under leave-one-cluster-out. See the
commit/PR notes for the actual run's coverage numbers; they are not restated here because they
are a property of one run against a live, rate-limited API, not a fact about the code.
