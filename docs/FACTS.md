# FACTS — what is actually in this repository

Phase 1 inventory for the research summary. Every line below is either a citation to a file and
line range, a value read out of a committed data or metrics file, or an explicit `[UNVERIFIED]`.

Repository state: branch `redesign/model-and-structure`, HEAD `58a9305` (2026-08-22).
Claims in `README.md` and `docs/model-design.md` are treated as *evidence of what was written*, not
as evidence of fact; where they conflict with committed artifacts, the conflict is recorded below.

---

## 1. Data sources

### 1.1 Satellite — Sentinel-2 L2A (optical)

| Fact | Value | Where |
| --- | --- | --- |
| Provider / API | Copernicus Data Space Ecosystem Sentinel Hub **Statistical API** (aggregated statistics, not rasters) | `src/argotech/data/sentinel.py:1-14`, `src/argotech/config.py:43-44` |
| Collection string | `sentinel-2-l2a` | `src/argotech/data/sentinel.py:191` |
| Processing level | L2A surface reflectance, values in [0, 1] | `src/argotech/data/sentinel.py:55-57` |
| Bands requested — index path | `B02`, `B04`, `B08`, `B11`, `dataMask` | `src/argotech/data/sentinel.py:33` |
| Bands requested — Presto path | `B02 B03 B04 B05 B06 B07 B08 B8A B11 B12`, `dataMask` | `src/argotech/data/sentinel.py:62` |
| Derived on the server | `ndvi = (B08−B04)/(B08+B04)`; `ndwi = (B08−B11)/(B08+B11)`; `evi = 2.5(B08−B04)/(B08+6·B04−7.5·B02+1)` | `src/argotech/data/sentinel.py:43-45` |
| Cloud filter | `maxCloudCoverage: 40` (percent), plus `dataMask` exclusion of no-data pixels | `src/argotech/data/sentinel.py:208`, `:27-28` |
| Temporal aggregation | `P30D` buckets; intervals with no valid statistics are dropped entirely | `src/argotech/data/sentinel.py:164`, `:243-249` |
| Spatial footprint | square bbox of half-width `d = 0.005°` ≈ 500 m per side around the point | `src/argotech/data/sentinel.py:204`, `:211-212` |
| Requested pixel resolution | `res_m=60` for **training** (both index and band paths), `res_m=10` default for serving `fetch_history` | `src/argotech/training/dataset.py:127-128`, `:169-170`; `src/argotech/data/sentinel.py:157` |
| Look-back at training | `days = years × 365`, default `--years 4` | `src/argotech/training/dataset.py:175-177`, `:332` |
| Look-back at serving | 365 days | `src/argotech/serving/pipeline.py:135` |

Note the naming: the column the repo calls `ndwi` is the B08/B11 formula, i.e. **NDMI**, and it is
renamed to `ndmi` inside the feature builder (`src/argotech/features/agronomic.py:371`,
`src/argotech/domain/indices.py:26-28`).

### 1.2 Satellite — Sentinel-1 GRD (radar)

| Fact | Value | Where |
| --- | --- | --- |
| Collection string | `sentinel-1-grd` | `src/argotech/data/sentinel.py:177` |
| Bands | `VV`, `VH`, `dataMask` | `src/argotech/data/sentinel.py:104` |
| Units | linear power, **not** dB (deliberate — averaging dB averages logarithms) | `src/argotech/data/sentinel.py:98-99` |
| Derived on the server | `rvi = 4·VH/(VV+VH)` | `src/argotech/data/sentinel.py:114-116` |
| Data filter | empty dict — no cloud filter applies to radar | `src/argotech/data/sentinel.py:178` |
| Resolution | `res_m=20` default; **`res_m=60` in the training builder** | `src/argotech/data/sentinel.py:166`; `src/argotech/training/dataset.py:137-138` |
| Aggregation | `P30D`, same as optical | `src/argotech/data/sentinel.py:176` |

### 1.3 Reanalysis / weather — Open-Meteo

| Fact | Value | Where |
| --- | --- | --- |
| Training endpoint | `https://archive-api.open-meteo.com/v1/archive`, described in the module docstring as ERA5 | `src/argotech/data/meteo.py:1, 32, 42-43` |
| Serving endpoint | `https://api.open-meteo.com/v1/forecast`, `past_days` capped at 92 | `src/argotech/data/meteo.py:33, 77-87` |
| Temporal resolution | **daily**, both endpoints, same variable list | `src/argotech/data/meteo.py:1-10, 22-30` |
| Variables (6) | `temperature_2m_max`, `temperature_2m_min`, `precipitation_sum`, `et0_fao_evapotranspiration`, `relative_humidity_2m_mean`, `shortwave_radiation_sum` | `src/argotech/data/meteo.py:23-30` |
| Timezone | UTC | `src/argotech/data/meteo.py:52, 87` |
| Latency allowance | archive window ends `today − 6 days` ("ERA5 lags ~5 days") | `src/argotech/data/meteo.py:108`, `src/argotech/training/dataset.py:178` |
| Rainfall climatology | mean 30-day total ending on the same MM-DD over the last `years=4`, **excluding the sample's own year** at training | `src/argotech/data/meteo.py:99-115`; `src/argotech/training/dataset.py:203-213` |
| Gap handling | forward-fill within a series, back-fill a leading gap; a **wholly absent** variable returns NaN and the site/request is refused | `src/argotech/data/meteo.py:118-158` |

**Spatial resolution of the reanalysis is not stated anywhere in the code.** No `models=` parameter
is passed to the archive endpoint, so which reanalysis product Open-Meteo actually serves for a given
coordinate is `[UNVERIFIED]` from this repository. `docs/model-design.md:522` asserts "ERA5 is a
~9-25 km grid" — that is prose in a design document, not something the code establishes.

An additional **hourly** call exists on the serving path only (temperature and relative humidity,
`past_days=14`) to derive leaf-wetness hours for the disease model:
`src/argotech/serving/pipeline.py:92-122`.

### 1.4 Other data sources

- **Terrain**: elevation only, taken from the Open-Meteo response's `elevation` field
  (`src/argotech/training/dataset.py:194`, `src/argotech/serving/pipeline.py:184`). No slope; the
  Presto packer writes slope as literal 0 (`src/argotech/models/embeddings.py:170`).
- **Soil**: none. No soil product is requested anywhere.
- **Administrative**: none. "Cluster" is a hand-drawn bounding box, not an administrative unit
  (`src/argotech/training/dataset.py:60-76`).
- **Holding-level / household**: read from the Kotlin backend's Postgres schema at *serving* time
  only — `farmer_profiles`, `farms`, `farm_crops`, `crops`, `farmers_ml_profiles`
  (`src/argotech/data/backend_schema.py:68-90`). **None of it enters the trained model.** It feeds
  the exposure and vulnerability terms of the risk composition only
  (`src/argotech/serving/pipeline.py:318-323`).

### 1.5 Geographic extent actually covered

Seven clusters are *defined* in code (`src/argotech/training/dataset.py:60-76`):

| Cluster | lat | lon | in committed training set? | in shipped artifact's `cluster_bounds`? |
| --- | --- | --- | --- | --- |
| Kaduna_Grain_Belt | 10.2 – 11.5 | 7.3 – 8.5 | yes | yes |
| Kano_Sudan_Savannah | 11.5 – 12.6 | 8.0 – 9.2 | yes | yes |
| Benue_River_Basin | 7.0 – 8.1 | 8.2 – 9.4 | yes | yes |
| Kenya_Rift_Valley | −0.2 – 1.2 | 34.8 – 36.1 | yes | yes |
| Ethiopian_Highlands | 8.3 – 9.9 | 38.0 – 39.6 | **no** | yes |
| Tanzania_Morogoro | −8.0 – −6.5 | 36.4 – 37.8 | **no** | yes |
| Niger_Delta | 4.3 – 5.6 | 5.6 – 6.9 | **no** | **no** |

Read from `data/training_set.parquet`: 4 clusters only; latitude −0.178 … 12.566, longitude
7.315 … 36.052, elevation 60 … 2837 m. The shipped artifact `artifacts/agronomic_risk.joblib`
carries `cluster_bounds` for 6 clusters and `cluster_stats` for only the 4 that are in the parquet.

Niger_Delta was added in commit `65306b4` ("Sample the Niger Delta, where the registered farmers
actually are") but **no Niger Delta data was ever fetched**: `.cache/sentinel/`,
`.cache/sentinel_sar/` and `.cache/sentinel_bands/` contain files for six cluster names and none for
Niger_Delta. The code comment at `src/argotech/training/dataset.py:67-70` states that both farms in
the production database sit near 4.93 N 6.34 E, which `assign_cluster` maps to `None`. So for the
only registered coordinates the code mentions, the cluster-relative twins and the peer anomaly are
NaN (`src/argotech/features/agronomic.py:187-206`, `src/argotech/serving/pipeline.py:273-277`).

Why Ethiopian_Highlands and Tanzania_Morogoro are absent from the current parquet despite having
populated caches is `[UNVERIFIED]` — no log, note or commit message in the repository explains it.

### 1.6 Date range actually covered

From `data/training_set.parquet`:

- `obs_date` (feature/prediction date): **2022-09-10 … 2026-06-21**
- `label_date` (outcome observation): **2022-10-10 … 2026-07-21**
- samples by year of `obs_date`: 2022 → 406, 2023 → 1105, 2024 → 1177, 2025 → 1248, 2026 → 660

The shipped artifact records `trained_on: 2022-09-10..2026-06-21`.

Because `dataset.collect_site` anchors its window on `date.today()`
(`src/argotech/training/dataset.py:175-178`), the covered range is a function of when the build was
run, and two builds on different days are not comparable — stated at
`src/argotech/features/agronomic.py:77-79`.

### 1.7 Sites and samples — where each number comes from

| Number | Value | Source |
| --- | --- | --- |
| Sites sampled per cluster (default) | `--sites 30` in code, `--sites 32` in README | `src/argotech/training/dataset.py:331`; `README.md` Training section |
| Sites in the committed training set | **122** (27 / 32 / 32 / 31 across four clusters) | `data/training_set.parquet` |
| Samples in the committed training set | **4,596** | `data/training_set.parquet`; matches `artifacts/metrics.json` `n_samples` and the artifact bundle's `n_samples` |
| Class balance | 0 → 0.6828, 1 → 0.1897, 2 → 0.1275 | `data/training_set.parquet`, `artifacts/metrics.json` |
| Rows with missing radar | 51 of 4,596 (`rvi`, `vh_vv_ratio`, `rvi_z_peer` all NaN) | `data/training_set.parquet` |
| Rows with missing label | 0 | `data/training_set.parquet` (`forward_z` count = 4596) |
| Earlier, archived dataset | 7,527 samples, 185 sites, 6 clusters | `data/training_set_pre_sar.parquet` |
| Presto embedding files | 6,716 rows × 128 dims, 165 sites, three variants | `data/presto_embeddings*.parquet` |

Site coordinates are generated deterministically, not drawn from a farm register: a Halton-style
radical-inverse lattice inside each bounding box, seed 7 (`src/argotech/training/dataset.py:92-119`).
**No site in the training set corresponds to a real farm.**

Conflicting counts recorded in the repository's own prose, all currently wrong against the committed
parquet and metrics:

- `README.md:421` — "6,957 samples, 185 sites"
- `docs/model-design.md:466` — "7,527 samples, 185 sites, 6 clusters"
- `src/argotech/training/train.py:132` — "35 of 6,957 rows have no Sentinel-1 pass"
  (actual: 51 of 4,596)

---

## 2. Features

### 2.1 The stored feature schema

`FEATURE_COLUMNS`, 29 columns, `src/argotech/features/agronomic.py:23-42`. Column order is declared
to be the model contract; appending is safe, reordering requires a retrain (`:22`).

Look-back window `WINDOW_DAYS = 90`, chosen because 92 days is the Open-Meteo forecast endpoint's
maximum look-back so serving can reproduce it (`src/argotech/features/agronomic.py:18-20`).

All arithmetic below is from `build()`, `src/argotech/features/agronomic.py:268-337`, delegating to
`src/argotech/domain/agronomy.py`.

**Thermal**
- `gdd_90` — Σ over 90 days of `max(0, (min(Tmax,30) + clamp(Tmin,10,30))/2 − 10)`. Base 10 °C, cap 30 °C for maize. `agronomy.py:29-37`, `:21-25`
- `gdd_since_onset` — same sum from the season-onset index forward. `agronomic.py:293-295`
- `tmax_mean_30`, `tmin_mean_30` — 30-day means. `agronomic.py:304-305`
- `diurnal_range_30` — `(ΣTmax − ΣTmin)/n` over 30 days. `agronomic.py:306`
- `heat_stress_days` — count of days with Tmax ≥ 35 °C in the 30-day window, **counted only when the phenology stage is `Flowering`**, otherwise 0. `agronomy.py:130-135`

**Water**
- `rain_30`, `rain_90` — precipitation sums. `agronomic.py:309-310`
- `et0_90` — sum of `et0_fao_evapotranspiration`. `agronomic.py:311`
- `water_satisfaction_30` — `min(1, Σrain / (Kc · Σet0))` over 30 days, Kc from the stage. `agronomy.py:101-127`
- `water_deficit_30` — `max(0, Kc·Σet0 − Σrain)`. same
- `dry_spell_30`, `dry_spell_90` — longest run of days with rain < 1.0 mm. `agronomy.py:116-119`
- `rain_anomaly_30` — `Σrain_30 − clim_rain_30`, where the climatology is this site's own mean 30-day total for the same MM-DD in *other* years. `agronomic.py:316-317`, `dataset.py:203-213`

**Atmosphere**
- `rh_mean_30` — mean `relative_humidity_2m_mean`. `agronomic.py:319`
- `radiation_90` — sum of `shortwave_radiation_sum`. `agronomic.py:320`

**Phenology**
- `days_since_onset` — days from the Sivakumar (1988) rainy-season onset rule: first day of a 3-day window accumulating ≥ 20 mm not followed by a ≥ 10-day dry spell within the next 30 days; 0 if no onset in the window. `agronomy.py:40-61`, `agronomic.py:289-292`
- `stage_kc` — FAO-56 single crop coefficient for the stage implied by `gdd_since_onset`: Emergence 0.35, Vegetative 0.75, Flowering 1.20, Grain Fill 1.05, Maturity 0.60, Post-Harvest 0.30. `agronomy.py:64-85`

**Canopy state (optical)**
- `ndvi`, `ndmi`, `evi` — the aggregated means returned by the Statistical API for the P30D bucket. `agronomic.py:369-373`
- `vci` — Kogan (1990) Vegetation Condition Index, `(NDVI − min)/(max − min) · 100`, where min/max are over **this site's own observations strictly before t**. `indices.py:61-68`, `agronomic.py:372`
- `ndvi_z_peer` — see §3.3

**Site**
- `latitude`, `longitude`, `elevation`

**Canopy structure (radar)**
- `rvi` — as returned by the S1 evalscript
- `vh_vv_ratio` — `VH/VV`, NaN when `VV == 0`. `agronomic.py:399`
- `rvi_z_peer` — see §3.3
- All three are NaN when no S1 observation falls within 20 days of the optical sensing date. `agronomic.py:244-265`, `:378-401`

### 2.2 Derived twins — `_cz`

`CLUSTER_RELATIVE`, 17 columns (`src/argotech/features/agronomic.py:65-72`): `gdd_90`,
`tmax_mean_30`, `diurnal_range_30`, `heat_stress_days`, `rain_30`, `rain_90`, `et0_90`,
`water_satisfaction_30`, `dry_spell_30`, `dry_spell_90`, `rh_mean_30`, `radiation_90`, `ndvi`,
`ndmi`, `evi`, `rvi`, `vh_vv_ratio`.

Transform (`src/argotech/features/agronomic.py:89-127`), per cluster, pooled over the whole frame
(one (μ, σ) pair per cluster × column, **not** per date):

```
x_cz = (x − μ_c) / σ_c    if σ_c > 1e-9 and x observed
     = 0                  if the column is constant within the cluster
     = NaN                if x is missing, or the field is outside every known cluster
```

σ uses pandas `std` (ddof = 1). The constants are snapshotted into the artifact as `cluster_stats`
(`train.py:646`) and replayed one row at a time at serving by `cluster_relative_row`
(`agronomic.py:187-206`). The three-way split (value / 0 / NaN) is deliberate and documented at
`agronomic.py:110-117`.

### 2.3 What the model actually consumes

`MODEL_FEATURES = [c for c in FEATURE_COLUMNS if c not in UNINFORMATIVE] + [c + "_cz" for c in
CLUSTER_RELATIVE]` — `src/argotech/features/agronomic.py:83-86`. **40 columns**, matching
`artifacts/metrics.json` `model_features` and the artifact bundle's `feature_columns`.

### 2.4 Features explicitly excluded

`UNINFORMATIVE = {vci, water_deficit_30, rain_anomaly_30, longitude, gdd_since_onset,
tmin_mean_30}` — `src/argotech/features/agronomic.py:57-59`. The exclusion is enforced by the list
comprehension at `:84`. Stated reasons: permutation importance at or below zero on a held-out
cluster, plus algebraic redundancy (`vci` and `rain_anomaly_30` are normalisations of `ndvi` and
`rain_30`; `water_deficit_30` is the complement of `water_satisfaction_30`; `gdd_since_onset` tracks
`gdd_90`; `longitude` is cluster identity).

Protected attributes are excluded from the **vulnerability** term, not from the model (the model
never sees household data at all): `PROTECTED_ATTRIBUTES = {head_gender, household_max_education,
religion, ethnicity}`, enforced by a raise at `src/argotech/domain/risk.py:235-237`. The serving
caller omits them at `src/argotech/serving/pipeline.py:430-446`.

Radar can be ablated as a set: `RADAR_FEATURES`, `src/argotech/features/agronomic.py:80-81`, applied
by `--no-radar` at `src/argotech/training/train.py:543-544`.

---

## 3. Label construction

### 3.1 Definition

`forward_z` — the field's NDVI at the **next P30D bucket**, standardised against the concurrently
observed NDVI of the *other* sites in the same cluster on that same bucket date.

Code, `src/argotech/training/dataset.py:276-292`:

```
peers = [NDVI of every other site in the same cluster with the same label sensing_date]
require len(peers) >= 5
mean = fmean(peers);  sd = pstdev(peers);  require sd >= 1e-6
z = (NDVI_field − mean) / sd            # -> forward_z
label = 2 if z <= −1.0 else 1 if z <= −0.35 else 0
```

Thresholds `SEVERE_Z = −1.0`, `ELEVATED_Z = −0.35` — `src/argotech/training/dataset.py:79`.
A non-finite `z` drops the sample rather than being scored as healthy (`:286-289`).

The **production model regresses `forward_z` directly**; the 3-class `label` is retained for the
control classifier arm and for every classification metric (`train.py:174-177`).

### 3.2 Lead time

`LABEL_HORIZON_INTERVALS = 1` — one P30D bucket ahead (`src/argotech/training/dataset.py:78`).

Because the Statistical API omits cloud-free-less intervals, "the next bucket" is the next
*observed* one, which may be much further out. A guard rejects any pair more than
`MAX_LABEL_GAP_DAYS = 45` days apart (`:81-85`, applied `:243-246`). The stated motivation is that
7 % of samples in the cached histories were 60–240 days ahead and were being pooled into something
downstream called a 30-day forecast.

**Verified in the committed data**: the gap between `obs_date` and `label_date` in
`data/training_set.parquet` is exactly 30 days for all 4,596 rows (std 0.0). So the horizon in the
shipped dataset is a clean 30 days, not merely bounded at 45.

### 3.3 Peer cohort and reference groups — two distinct mechanisms

These are easy to conflate and the code keeps them apart deliberately.

**(a) The label cohort — cluster × exact sensing date, cross-sectional.**
Built at `src/argotech/training/dataset.py:222-227` as `{(cluster, sensing_date): [ndvi, …]}` over
all sites, then used at `:277-285`. Minimum 5 peers, self excluded. It exists only in the training
builder; `forward_z` is never computed at serving, so it carries no train/serve skew
(`:218-221`).

**(b) The feature-side peer reference — cluster × calendar month, from fitted constants.**
`peer_bucket(cluster, date) = "<cluster>|<MM>"` — `src/argotech/features/agronomic.py:135-142`.
`peer_stats(df)` returns `{bucket: {column: [μ, σ]}}` for `PEER_RELATIVE = ["ndvi", "rvi"]`
(`:145-159`). `ndvi_z_peer` and `rvi_z_peer` are computed by `peer_z` against that reference
(`:340-351`), yielding NaN when the bucket was never measured (`:162-171`).

Calendar month rather than exact date, because "the cohort a field is compared against has to be
knowable at serving time, and a specific date's cohort is not" (`:137-141`). The shipped artifact
carries 45 such buckets.

The dataset builder runs this as a **second pass** over exactly the rows it just built
(`src/argotech/training/dataset.py:305-314`) so that training features and the constants shipped in
the artifact are the same numbers. The commit message for `c54f643` is "Give the peer anomaly one
definition"; the docstring at `agronomic.py:361-366` records that before this change, training used
the concurrent cross-site cohort and serving used the site's own 12-month history under one column
name — correlated 0.626 with 25.5 % sign flips.

For evaluation, `peer_stats` is computed **per fold**, but for the shipped artifact it is computed
over the whole frame; the reasoning is given at `train.py:649-657` and at `agronomic.py:152-154`.
Note the asymmetry this creates and the argument offered for it: computing it per fold would leave a
held-out cluster with no bucket at all, which would blind both baselines while the model keeps 39
other features.

### 3.4 Standardisation summary

| Quantity | Reference | ddof | Missing → |
| --- | --- | --- | --- |
| `forward_z` (label) | concurrent peers, cluster × exact date, ≥ 5 peers, self excluded | population (`pstdev`) | sample dropped |
| `ndvi_z_peer`, `rvi_z_peer` | cluster × calendar month, fitted constants in the artifact | sample (`.std()`, ddof=1) | NaN |
| `*_cz` twins | cluster, pooled over the frame | sample (ddof=1) | NaN (or 0.0 if σ ≈ 0) |

### 3.5 Where the feature/label separation is enforced

1. Feature window is `daily[end_idx − 90 : end_idx]`, exclusive of the prediction date itself —
   `src/argotech/training/dataset.py:256-259`; a sample with fewer than 90 prior days is dropped.
2. Label comes from `history[k+1]`, a strictly later observation —
   `src/argotech/training/dataset.py:235-239`.
3. Horizon guard rejects gaps > 45 days — `:243-246`.
4. VCI uses `history[:k]`, strictly earlier observations of the same site —
   `:260`, `src/argotech/features/agronomic.py:354-359`.
5. Rainfall climatology excludes the sample's own year —
   `src/argotech/training/dataset.py:203-213` (`ts[:4] == exclude_year` → skip).
6. The site-climatology *baseline* uses an expanding mean shifted by one, i.e. strictly earlier
   observations — `src/argotech/training/train.py:272-277`.
7. `label_date` is stored on every row so the horizon is auditable and a temporal split can key on
   when the outcome became known — `src/argotech/training/dataset.py:296-298`.

**What is *not* leakage-controlled, and is documented as such**: `add_cluster_relative` and
`peer_stats` for the shipped artifact are fitted over the full frame including the held-out cluster.
The argument given (`agronomic.py:91-95`, `train.py:649-657`) is that both are label-free transforms.
That argument is about *label* leakage; it does not address transduction over the feature
distribution. Recorded here as a judgement made in the code, not as a settled fact.

---

## 4. Model

| Fact | Value | Where |
| --- | --- | --- |
| Production estimator | `HistGradientBoostingRegressor` on `forward_z` | `src/argotech/training/train.py:146-168`; confirmed by loading `artifacts/agronomic_risk.joblib` |
| Hyperparameters | `max_iter=300`, `learning_rate=0.06`, `max_depth=5`, `min_samples_leaf=25`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.15`, `random_state=42` | `train.py:159-168`; confirmed in the loaded artifact |
| Calibration | **none on the production arm** — a regressor has no posterior. Explicit at `train.py:150-151` and `:571` | |
| Control arm 1 | `CalibratedClassifierCV(HistGradientBoostingClassifier(same hyperparameters), method="sigmoid", cv=5)` on `label` | `train.py:93-120` |
| Control arm 2 | `SimpleImputer(median) → StandardScaler → LogisticRegression(max_iter=2000, C=0.5)` on `label` | `train.py:123-143` |
| Class weighting | deliberately **not** used; reasoning at `train.py:114-119` | |
| Missing values | routed natively by the boosted arms; the linear arm is median-imputed, described as its handicap | `train.py:132-137` |
| Shipped artifact version | `agro-20260821T121016Z`, 265,871 bytes | file listing; bundle field `version` |
| Artifact contents | `model`, `version`, `feature_columns` (40), `cluster_relative` (17), `cluster_stats` (4 clusters), `peer_stats` (45 buckets), `cluster_bounds` (6 clusters), `classes`, `label`, `target="forward_z"`, `thresholds`, `n_samples=4596`, `trained_on` | `train.py:637-669`; read from the bundle |
| Load-time contract | serving refuses any artifact whose `target != "forward_z"` or that carries no `peer_stats` | `src/argotech/models/registry.py:42-56` |
| Final fit | on **all** rows with a finite `forward_z`, after the fold estimates are recorded | `train.py:613-618` |
| Class balance in training | 0 → 68.28 %, 1 → 18.97 %, 2 → 12.75 % | `data/training_set.parquet`, `artifacts/metrics.json` |

### 4.1 Decision rule

The regressor's output is negated into a risk score (`risk = −ŷ`, `train.py:398`) and cut at the
**training prior's quantiles**, not at argmax (`decide`, `train.py:198-216`):

```
π_k = n_k / n           q1 = 1 − π_1 − π_2      q2 = 1 − π_2
t_j = Quantile(risk, q_j)
ŷ  = 2 if risk ≥ t2 else 1 if risk ≥ t1 else 0
```

### 4.2 How the model enters a served prediction

The model is **one hazard term**, not the risk score. `src/argotech/serving/pipeline.py:279-324`:

1. `ẑ = model.predict(row)` where `row` is the 40-column feature vector.
2. `h_veg = 1 / (1 + exp((ẑ + 1.0) / 0.5))` — logistic centred on `SEVERE_ANOMALY_Z = −1.0`, width
   `ANOMALY_SOFTNESS = 0.5` (`src/argotech/domain/risk.py:38, 44, 47-79`).
3. `h_veg` joins drought, disease and heat — all derived from published agronomy with no fitted
   parameters — under a noisy-OR: `combined = 1 − Π(1 − h_i)` (`risk.py:82-91, 112-149`).
4. `loss_rate = combined × (0.5 + 0.5·vulnerability) × MAX_LOSS_FRACTION`, with
   `MAX_LOSS_FRACTION = 0.6` a stated constant, explicitly not calibrated against outcomes
   (`risk.py:281-295`).
5. `expected_loss = value_at_risk × loss_rate`, where `value_at_risk = area × expected_yield ×
   price` and price is a static four-entry USD/tonne table (`pipeline.py:86-89`).

`VEGETATION_HAZARD_SOURCE` selects between the model (default since 2026-08-21) and persistence —
the field's own observed `ndvi_z_peer` carried forward through the identical logistic
(`src/argotech/config.py:9-34`, `pipeline.py:284-305`).

---

## 5. Evaluation

### 5.1 Splitting strategy

**Spatial — leave-one-cluster-out**, `src/argotech/training/train.py:424-432`.
One fold per distinct value of `cluster`; a fold is skipped if the test set has < 30 rows or the
train set lacks all three classes. **Block size = the whole cluster bounding box** (roughly 1.1–1.6°
of latitude, i.e. ~120–180 km per side). **Buffer distance: none.** There is no buffer parameter
anywhere in the code — blocking is by cluster membership, and clusters are ≥ 300 km apart, so no
buffer is needed between them; there is likewise no within-cluster blocking. **Fold count = number
of clusters present in the parquet = 4** in the committed run.

**Temporal — forward chaining**, `src/argotech/training/train.py:435-450`.
3 folds. Cut date = the date at index `int(n_dates × (0.55 + 0.12·i))` for i = 0,1,2. Train =
`obs_date < cut`; test = the **first two distinct observation dates on or after the cut** only
(`:445-446`), so each fold tests one step ahead. Same < 30 rows / < 3 classes skip rule.
Committed cuts: 2024-09-29 (n=183), 2025-03-28 (n=242), 2025-09-24 (n=171).

Note: the temporal split cuts on `obs_date`, not on `label_date`. Since the label is observed 30 days
after `obs_date`, the last training rows' outcomes become known *after* the first test rows'
prediction dates. This is not corrected anywhere in the code.

### 5.2 Baselines implemented

All three are recomputed **per fold**, because two of them depend on the training prior
(`train.py:383-385`).

| Baseline | What it does | Where |
| --- | --- | --- |
| Majority class | predicts `argmax` of the training class counts for every test row | `train.py:223-224` |
| Persistence | carries `ndvi_z_peer` forward unchanged and applies the label's own thresholds: `ŷ = 2 if z ≤ −1.0 else 1 if z ≤ −0.35 else 0`; ranking score `−ndvi_z_peer` | `train.py:227-244` |
| Site climatology | expanding mean of the site's `ndvi_z_peer` over strictly earlier observations (`c_j = (1/j)·Σ_{m<j} z_m`, `c_0 = 0`), thresholded identically; ranking score `−clim_z_prior` | `train.py:247-287` |

A **linear arm** (`macro_f1_linear`, `spearman_linear`, …) is reported beside the boosted arms as a
shift-robustness check, not as a candidate (`train.py:123-143`). The **classifier arm** (`_hgb`
suffix) is the previous production model retained as a control (`train.py:171-177`).

### 5.3 Metrics computed

| Metric | Definition | Where |
| --- | --- | --- |
| `macro_f1` | sklearn `f1_score(average="macro", zero_division=0)` over the 3 classes | `train.py:359` |
| `balanced_accuracy` | sklearn `balanced_accuracy_score` | `train.py:360` |
| `ece` | `Σ_B (\|B\|/n)·\|acc(B) − conf(B)\|` over 10 equal-width bins of `max_k P(Y=k\|x)`. **Classifier arms only** | `train.py:290-311` |
| `precision_at_k`, k ∈ {10, 25, 50} | share of the k highest-risk test rows whose true label is ≥ 1 (elevated **or** severe) | `train.py:314-328` |
| `spearman` | Spearman ρ between the risk score and `−forward_z`, over every finite pair in the fold; positive = correctly ordered | `train.py:331-351` |
| Decision-rule sweep | alert-rate multiplier m ∈ {0.5, 0.75, 1, 1.5, 2, 3} applied to the prior before the quantile cut; reports `flagged_share`, `macro_f1`, `recall_severe`, `recall_elevated_or_worse` on pooled out-of-fold spatial predictions | `train.py:453-488` |
| Permutation importance | mean drop in macro F1 over 3 permutations per column, measured on the single held-out cluster `max(df.cluster.unique())` (alphabetically last — `Kenya_Rift_Valley` in the committed run) | `train.py:491-520` |

Not implemented anywhere despite being specified in `docs/model-design.md:384-390`: the fairness
slice report and the CI release gate. `.github/workflows/deploy.yml` builds and deploys; it does not
gate on any metric.

### 5.4 Numbers actually in the repository

**`artifacts/metrics.json`** (the file the shipped artifact was written beside, commit `c54f643`)
— 4,596 samples, 40 features, **4 spatial folds**, 3 temporal folds. Fold means computed from that
file:

| Leave-one-cluster-out (4 folds) | P@25 | ρ | macro F1 |
| --- | --- | --- | --- |
| Regressor (`hgbr`, production) | **0.600** | 0.398 | 0.450 |
| Site climatology | **0.640** | 0.415 | 0.443 |
| Persistence | 0.490 | **0.423** | 0.478 |
| Classifier arm (`hgb`, control) | 0.630 | 0.333 | 0.436 |
| Linear arm | 0.480 | 0.281 | 0.431 |
| Majority class | — | — | 0.270 |

Per-fold P@25 for the regressor: Benue 0.32, Kaduna 0.80, Kano 0.80, Kenya 0.48.
Per-fold ρ: 0.233, 0.564, 0.532, 0.260.

| Forward-chaining temporal (3 folds) | P@25 | ρ | macro F1 |
| --- | --- | --- | --- |
| Regressor | 0.600 | 0.464 | 0.463 |
| Site climatology | **0.720** | **0.536** | — |
| Persistence | 0.573 | 0.402 | — |

Decision-rule sweep (pooled out-of-fold, spatial): at m = 1.0, flagged share 0.317, macro F1 0.443,
recall(severe) 0.271, recall(≥ elevated) 0.498. Macro F1 peaks at m = 1.0; recall(severe) rises
monotonically to 0.555 at m = 3.0 where 95 % of fields are flagged.

Permutation importance on the held-out cluster: `ndvi_z_peer` **+0.0642**, then `et0_90_cz` +0.0051,
`dry_spell_90` +0.0035, `rh_mean_30` +0.0021, `rain_30` +0.0014. **23 of 40 columns score
negative.** The top feature is the same quantity the persistence baseline uses on its own.

Distribution facts read from `data/training_set.parquet`: `forward_z` std **1.145** (mean −0.0005,
min −11.35, max +7.08); `ndvi_z_peer` std **0.995**; the persistence hazard exceeds 0.5 on **11.6 %**
of rows.

**Other committed metrics files:**

- `artifacts/metrics.agro-20260811T124108Z.json` — 7,527 samples, 6 folds, classifier-era (no
  `spearman` key). Spatial P@25 mean 0.693 vs persistence 0.460. This is the run reproduced in
  `docs/model-design.md:471-489`, and those tables match it exactly.
- `artifacts/experimental/agronomic_risk_clusterrel_metrics.json` — 6,716 samples, **168 features**
  (i.e. with the 128 Presto dims), 6 folds, still classifier-era. Spatial P@25 mean **0.760** vs
  climatology 0.707, persistence 0.507. No `spearman` key.

### 5.5 Numbers cited in the repository that cannot be reproduced from it

`README.md:421-441`, `src/argotech/config.py:18-33` and `src/argotech/domain/risk.py:63-67` all cite
the same table and attribute it to `artifacts/metrics.json`, seed 42, **six** leave-one-cluster-out
folds, artifact `agro-20260821T065051Z`, 6,957 samples, 185 sites:

| Cited | P@25 | ρ |
| --- | --- | --- |
| Regressor | 0.760 | 0.377 |
| Site climatology | 0.680 | 0.363 |
| Persistence | 0.467 | 0.368 |
| Classifier arm | 0.647 | 0.235 |

**`[UNVERIFIED]` — none of this is in the repository.** `artifacts/metrics.json` has four folds,
4,596 samples, and P@25 0.600 / ρ 0.398 for the regressor. The value 0.760 appears only in the
168-feature *experimental* file, which has no `spearman` key at all. Grepping all four committed
metrics files, the strings `0.377`, `0.368`, `0.363`, `0.235`, `0.647`, `0.467` and `0.680` occur
exactly once between them, as an unrelated `balanced_accuracy` of 0.3778. The cited artifact version
`agro-20260821T065051Z` is not the shipped one (`agro-20260821T121016Z`).

Likewise `README.md:436-438`: "predicted std 0.508, observed `forward_z` std 1.165, persistence's
`ndvi_z_peer` 1.517 … exceeding 0.5 on 2.2 % of training fields against persistence's 14.2 %". The
committed parquet gives 1.145 and 0.995 and 11.6 %. `[UNVERIFIED]` for 0.508 — reproducing it needs
a model prediction over the training frame, which is not committed.

Likewise `README.md:467-473`: the permutation-importance figures (`ndvi_z_peer` +0.0322, `ndmi`
+0.0172, `evi` +0.0157, `rvi_z_peer` +0.0153, twelve negative columns) do not match
`artifacts/metrics.json` (+0.0642 for `ndvi_z_peer`; 23 negative columns).

The most likely explanation is that commit `c54f643` retrained on a smaller rebuilt dataset and
overwrote `artifacts/metrics.json` without updating the prose. That is inference, not fact, and is
recorded as such.

---

## 6. Known gaps

### 6.1 Marked in code

| Marker | What | Where |
| --- | --- | --- |
| `ponytail:` | Farm-gate prices are a static 4-crop USD/tonne table | `src/argotech/serving/pipeline.py:86-89` |
| `ponytail:` | Three crops hard-coded in `CROP_GDD` | `src/argotech/domain/agronomy.py:20-25` |
| "placeholder" | `_expected_yield` falls back to an NDVI-anchored linear guess when no reported yield exists | `src/argotech/serving/pipeline.py:415-428` |
| "placeholder" | "a farmer's most recent farm" stands in for naming an actual field | `src/argotech/data/backend_schema.py:36-39` |
| Stated constant | `MAX_LOSS_FRACTION = 0.6`, explicitly "not a number pretending to be learned" | `src/argotech/domain/risk.py:281-284` |
| Always zero | `has_irrigation` is not in the backend schema, so one of seven coping factors is constant 0 | `src/argotech/serving/pipeline.py:439` |
| Silent default | `asset_score` defaults to 45.0 and `market_access_score` to 50.0 when the `farmers_ml_profiles` row is absent | `src/argotech/serving/pipeline.py:444-445`; `README.md:49` states nothing in this repo writes that table |
| Silent default | An uncaptured crop becomes "Maize" | `src/argotech/data/backend_schema.py:98-113` |
| Not built | Fairness slice report, CI release gate | specified `docs/model-design.md:384-390`; absent from `.github/workflows/deploy.yml` |
| Not consumed | `predictions` and `field_outcomes` are written but nothing reads them | `README.md:50`; `src/argotech/serving/api/outcomes.py` |

### 6.2 Assumptions that break outside the current region or season

1. **Cluster membership is a bounding-box lookup with no fallback.** A field outside all boxes gets
   `cluster = None` → all 17 `_cz` twins NaN and `ndvi_z_peer` NaN
   (`src/argotech/features/agronomic.py:209-220`, `:187-206`;
   `src/argotech/serving/pipeline.py:273-277`). The shipped artifact knows 6 boxes, all in the Sahel,
   the Rift Valley and the Ethiopian highlands. Both farms the code names as being in the production
   database (≈ 4.93 N 6.34 E, Niger Delta) fall outside all of them.
2. **The peer reference is `cluster|MM` and the artifact carries 45 buckets.** A month never sampled
   in a cluster yields NaN (`agronomic.py:162-171`). Benue, for example, has no bucket for 07, 08 or
   09 — the peak of its rainy season.
3. **Single-season phenology.** `season_onset_index` returns the *first* qualifying onset in the
   90-day window (`agronomy.py:40-61`). The code's own note on the Niger Delta
   (`dataset.py:71-74`) states that region has two cropping seasons; the onset rule has no concept
   of a second one.
4. **Crop defaults to maize**, with base 10 °C / cap 30 °C thermal time and FAO-56 maize Kc
   (`agronomy.py:21-26`, `:78-85`; `backend_schema.py:32`). Only maize, sorghum and rice are
   parameterised; cassava has a price entry but no GDD entry.
5. **Heat stress is only counted during `Flowering`** (`agronomy.py:130-135`), so a stage
   misclassification zeroes the entire heat hazard.
6. **The disease model is Wallin (1962) late blight**, a potato/tomato model, applied on any crop
   (`agronomy.py:142-170`); leaf wetness is proxied by hours at RH ≥ 90 %
   (`pipeline.py:116-117`).
7. **Leaf wetness comes from a 14-day hourly forecast call that is serving-only.** It is not in the
   training features at all, so `cumulative_dsv` — which feeds the disease hazard — was never
   evaluated against anything.
8. **`_stats` returns `[]` on any failure**, indistinguishable from "no imagery here"
   (`sentinel.py:251-253`). The training cache refuses to memoise an empty result for exactly this
   reason (`dataset.py:142-157`), after a burst of CDSE 429s produced 69 permanently band-less
   sites; the serving path has no such protection and degrades silently to physics-only.
9. **`date.today()` anchors the dataset window** (`dataset.py:175-178`), so a rebuild is not
   reproducible and two builds are not comparable — stated at `agronomic.py:77-79`.
10. **Temporal split cuts on `obs_date`, not `label_date`** (`train.py:443`), so the last month of
    training outcomes overlaps the first test predictions in wall-clock time.

### 6.3 Provenance caveats on everything above

- No site in the training set is a real farm; all 122 are lattice points inside 4 bounding boxes
  (`dataset.py:92-119`).
- The label is a satellite proxy. No harvest record, agronomist diagnosis or field visit appears
  anywhere in the repository. `field_outcomes` exists as an empty table definition and an endpoint
  (`src/argotech/data/store.py:10-13`, `src/argotech/serving/api/outcomes.py`); no outcome data is
  committed.
- Claims in `README.md` about deployment (GCE VM, compose stack, Kotlin backend) are configuration
  and prose. Nothing in this repository evidences a live user, a prediction served to a farmer, or a
  realised outcome. Treat all such statements as `[UNVERIFIED]`.

---

## 7. The lab redesign — peer-standardisation leak, temporal cut, NDVI validity

Panel throughout this section: `data/training_set.parquet`, 4,596 rows, 122 sites, 4 clusters,
2022-09-10 .. 2026-06-21, `content_hash` `6c832dc2741766af8a74b5fd2dd39cab4d4557fdba811ecc728a8a6b8bf28aeb`
(hashed and recorded by `build_dataset`, `src/argotech/lab/panel.py:400`). Concepts and formulas are
defined once, in `docs/CONCEPTS.md` Part 4; this section is the citation trail behind that document's
claims, not a restatement of them.

### 7.1 The diagnosis — a field effect survives standardisation

`forward_z` decomposes as a time-invariant field effect plus a cohort-date effect plus residual;
stage-1 peer standardisation removes the date effect, nothing removed the field effect. Measured
(`experiments/E01_variance_decomposition.out`, reproducible via `decompose()`,
`src/argotech/lab/variance.py:39`, using the unbiased one-way random-effects `icc()` at
`variance.py:16`): field-effect ICC **0.345**, 95% CI **[0.2357, 0.4353]** over 1,000 site
bootstraps; cluster ICC 0.0000; cohort-date ICC 0.0000 (mean +0.0014), confirming stage 1 works.
Naive eta-squared reports 0.360 on the same data — it credits groups with their own sampling noise,
which the unbiased estimator does not. `ndvi_z_peer` itself has a field-effect ICC of 0.279. A
zero-parameter expanding-window field mean (the `climatology` arm) reaches Spearman +0.437 and
out-of-sample R² +0.199, against the fitted 40-feature model's +0.398 on blocked folds
(`artifacts/metrics.json`) — the discriminating test used in place of the spec's original "field
effect is the majority share" criterion, which failed (34.5% is not a majority).

### 7.2 Three leaks, found and fixed

**(a) Same-row leak in a test, not the code.** `alpha_hat`'s own guard test spiked only each site's
*last* row, so deleting `.shift(1)` at `src/argotech/lab/targets.py:54` — a genuine same-row leak —
passed all 11 existing tests. Fixed by `test_alpha_hat_excludes_the_current_rows_own_value`
(`tests/test_targets.py:40`), verified to fail if `.shift(1)` is reverted to `.shift(0)`.

**(b) Inverted temporal cut.** An earlier version of this repository's evaluation cut training rows
on `obs_date` (the prediction date) rather than `label_date` (the date the outcome became known) and
called that the defect to be fixed; a later commit shipped that same defect again under a message
claiming to fix it. Measured on the committed panel: the `obs_date`/`label_date` gap is exactly 30
days on all 4,596 rows; cutting `train = df[df.obs_date < boundary]` gives 2,931 rows of which
**121** have `label_date >= boundary` — an outcome that postdates the first test prediction; cutting
`train = df[df.label_date < boundary]` gives 2,810 rows with **zero**. Fixed in
`forward_chaining()`, `src/argotech/lab/splits.py:31-49`: train takes `label_date < boundary`, test
takes `obs_date >= boundary`, rows satisfying neither are embargoed (52/31/69 rows per fold across
the three boundaries) — purged/embargoed forward chaining.

**(c) Peer reference fitted over the whole frame.** `peer_stats`'s own docstring
(`src/argotech/features/agronomic.py:152-154`, see §3.3 above) said it should be "computed per fold
during evaluation"; no caller in `argotech.training` ever did. Measured: the bucket key is
`cluster|MM`, calendar month pooled across all years — **42 of 45 buckets span multiple years**
(`Benue_River_Basin|01` runs 2023-01-08 .. 2026-01-22). Across the three forward-chaining
boundaries, **74.0% / 49.9% / 24.3%** of a training row's peer cohort lay at or after the boundary
(max 94.2%). Under leave-one-cluster-out the held-out cluster's bucket drew from exactly one
cluster: itself. Fixed by `src/argotech/lab/peers.py`: `peer_key()` (`:30`) computes the bucket for
three kinds — `leaky` (the panel's whole-frame baked column, kept as a control), `cluster_month`
(`<cluster>|<MM>`), and `geo_month` (`<lat_band>_<elev_band>|<MM>`, fixed cut points, never
quantiles fit over the frame); `fit_peer_stats()` (`:47`) fits on training rows only;
`apply_peer_z()` (`:72`) applies those fitted constants to any frame. Measured coverage of a
held-out cluster after the fix: `cluster_month` **[0.0, 0.0, 0.0, 0.0]** — an honest cluster-keyed
fit leaves an unseen region with no reference at all. `geo_month` **[1.0, 1.0, 1.0, 0.472]** — Kenya
partial, the panel's one highland cluster (1760 m mean elevation), borrowing a reference only where
its sites fall below the 1000 m band.

### 7.3 Two further honesty defects, both fixed

**(a) A fabricated `alpha_hat`.** With zero peer coverage, `ndvi_z_peer` is 100% NaN for a fold, but
the prior implementation's `prior_mean.fillna(0.0)` combined with a `prior_n` that counted rows
regardless of NaN yielded exactly `alpha_hat = 0.0`, so `ztilde = forward_z - 0.0 = forward_z`
exactly. Measured on the real Benue fold: peer coverage 0.000, `alpha_hat` unique value `[0.]`,
`ztilde == forward_z` exactly. Every `within_*` × `cluster_month` cell was therefore a `level_z`
result reported under a `within` label. Fixed at `src/argotech/lab/targets.py:48`: only non-NaN
priors are counted, the `fillna` is removed, and a NaN `prior_mean` propagates to a NaN `alpha_hat`
rather than being fabricated as zero.

**(b) Partially-skipped runs reported as complete.** Measured: `delta_z`/`geo_month` at the 5°/500 m
band skipped the Benue and Kenya folds, exited 0, and printed a "linear NB +0.1644" headline computed
over 2 of 4 clusters as if it were a full leave-one-cluster-out result. Fixed: fold counts are now
recorded in the run's output and surfaced rather than silently dropped
(`src/argotech/lab/run.py`).

### 7.4 Results — the authoritative matrix (`experiments/E02/`, 12 cells)

Only `level_z` supports comparison across `peer_key`; `within_y`, `within_xy` and `delta_z` are
defined *through* `ndvi_z_peer`, so changing the key changes the target, not just the estimator —
`run.py` warns at runtime if this is attempted. `level_z`, net benefit with 95% CI:

| Protocol | Peer key | Net benefit [95% CI] | Crosses zero? |
| --- | --- | --- | --- |
| spatial | `leaky` | +0.0064 [-0.0014, +0.0142] | yes |
| spatial | `geo_month` | +0.0066 [-0.0020, +0.0157] | yes |
| spatial | `cluster_month` | — all 4 folds skipped, no reference exists | n/a |
| temporal | `leaky` | +0.0194 [+0.0182, +0.0216] | no |
| temporal | `cluster_month` | +0.0179 [+0.0152, +0.0202] | no |
| temporal | `geo_month` | +0.0172 [+0.0162, +0.0182] | no |

Best single cell in the 12-cell matrix: spatial / `geo_month` / `within_xy` / `linear`, net benefit
**+0.0404**, 95% CI **[-0.0072, +0.0940]** — crosses zero.

### 7.5 The replication (`experiments/E03-replication/`)

Per-fold decomposition of the +0.0404 headline:

| Fold | Net benefit | n | Event rate | Peer coverage |
| --- | --- | --- | --- | --- |
| Benue | +0.1268 | 566 | 0.488 | 1.000 |
| Kaduna | -0.0103 | 1043 | 0.079 | 1.000 |
| Kano | -0.0042 | 1257 | 0.049 | 1.000 |
| Kenya | +0.0491 | 748 | 0.287 | 0.472 |

**Two of four folds are negative.** The mean is carried by Benue at roughly 3× the headline effect;
the fold spread (-0.0103 .. +0.1268, range 0.137) is more than 3× the reported effect. Maximum
attainable net benefit *is* the event rate (`net_benefit()`, `src/argotech/lab/evaluate.py:44`), so
folds are not on a common scale — Benue's ceiling (0.488) is ten times Kano's (0.049), and the fold
carrying the mean is the fold with the most room to score.

No seed sweep is possible: measured `max |pred(seed) - pred(42)|` across seeds `{42, 7, 2024, 1, 99}`
is exactly `0.000e+00` for both the `linear` and `boosted` arms. `linear` is a closed-form ridge
solve; `boosted` runs with `early_stopping=False` (set to stop sklearn's internal IID validation
split from leaking across the spatial blocking), which removed its only stochastic component — five
seeds would return five identical numbers.

Site-level bootstrap (`bootstrap_ci()`, `src/argotech/lab/evaluate.py:104`; script
`experiments/E03-replication/bootstrap_sites.py`), 200/200 replicates, `boot_seed = 20260826`:
`linear` +0.0404, CI **[-0.0026, +0.0781]**, P(NB>0)=0.950; `boosted` +0.0218, CI
[-0.0016, +0.0424], P(NB>0)=0.950. That interval (width 0.0808) is *narrower* than the fold-level one
(width 0.1012), against a prediction made before the run — resampling sites holds cluster identity
fixed, so it measures within-cluster variance, while transferability depends on between-cluster
variance. The interval is correct and answers an easier question; four clusters remain the
independent units for the harder one.

### 7.6 The band sweep (`experiments/E04-band-sweep/`, 24 cells)

| Band | Donor rows [Benue, Kaduna, Kano, Kenya] |
| --- | --- |
| 5° / 500 m | [0, 282, 1098, 0] |
| 10° / 1000 m | [36, 1353, 1098, 647] |
| 20° / 1000 m | [2175, 2799, 1792, 3139] |

Cluster means: Benue 7.6°N/118 m, Kaduna 10.8°N/676 m, Kano 12.0°N/455 m, Kenya 0.5°N/1760 m. Beyond
20° latitude stops discriminating entirely (clusters span 0.5–12°N), so 20° and 90° rows are
identical and the key is effectively elevation-and-month, pooling humid Benue with Sudan-savannah
Kano. **0 of 12 `within_xy` cells and 0 of 12 `level_z` cells have a CI excluding zero.** Coverage
responds to band width (0.29 → 0.89); net benefit does not separate from zero at any band. Two
confounds make the net-benefit column unreadable as skill-versus-band on its own: low-coverage cells
score only 2 of 4 folds, and the event rate ranges 0.124–0.300 while the maximum attainable net
benefit *is* the event rate (§7.5).

### 7.7 The multiple-comparisons check that governs 7.4–7.6

Spatial: **165 cells tested, 8 significant-positive, ~4.1 expected by chance at a 95% two-sided
interval with no correction — a ratio of 1.9×.** Temporal: **65 cells tested, 13
significant-positive, ~1.6 expected — 8.0×.** The spatial hits are also internally incoherent:
`level_z`/`geo_month`/`boosted` clears zero at the 500 m elevation band and at 2000 m but *not* at
the 1000 m band between them, which is not the shape a real effect produces.

**Defensible claim:** temporal skill is established; spatial (out-of-region) skill is not
distinguishable from multiple-testing noise at four independent spatial units. No formal correction
(Bonferroni, Benjamini–Hochberg) has been applied to any of the above; the raw ratio is reported
instead, and a pre-registered comparison would be the correct fix.

**Correction of record:** an earlier claim that "no configuration shows out-of-region skill" was
wrong — 8 of 165 spatial cells do clear zero. That earlier claim was a generalisation from
`within_xy`, the one target the E03 replication tested; the multiple-comparisons ratio above is the
correct rebuttal to those 8 cells, not a denial that they exist.

### 7.8 NDVI validity (diagnosed this session; the fix is at the source, the panel was not rebuilt)

`forward_z` runs **-11.35 to +7.08** with sd 1.144 despite being a z-score by construction; the
arithmetic was verified correct by independent recomputation (r = 0.998). The cause is not small or
tight cohorts (sizes ~32, sds 0.07–0.33); it is physically invalid NDVI. NDVI = (NIR−Red)/(NIR+Red);
NDVI ≤ 0 means NIR ≤ Red — water, cloud, shadow or snow, never vegetation. **233 rows (5.07%) of the
committed panel have NDVI < 0**, two of them exactly -1.0 (NIR = 0, a no-data sentinel). Rows with
`|z| > 3` are **8× enriched** for a negative label-date NDVI (39.7% against a 5.1% base rate).

Filtering is a validity fix, not a tail fix: simulated, excluding NDVI < 0 from both cohort and
label moves the minimum -11.35 → -4.56 but *pushes the maximum* +7.08 → +7.59, because removing low
cohort members shrinks the cohort sd; overall sd barely moves (1.144 → 1.142). At a threshold of
0.15 the max reaches +8.83. Remaining tails after filtering are legitimate signal, not noise to be
cut further: a field at NDVI 0.9 in a cohort averaging 0.28 with sd 0.11 really is z = +5.6, which is
what a triage system exists to find.

A second, independent bug at the same site: peers were excluded by *value*
(`v != label_obs["ndvi"]`), so any peer whose NDVI happened to equal the field's own was silently
dropped, shrinking and biasing the cohort. Fixed to exclude by site identity —
`src/argotech/lab/panel.py:326-327` now reads `if site_id != s["site_id"]`.

**What was changed** (`src/argotech/lab/panel.py`, commit `2d4a595`):

| Constant | Value | Was | Where | Why |
| --- | --- | --- | --- | --- |
| `MIN_VALID_NDVI` | `0.0` | not enforced | `panel.py:88`, applied `:264` (cohort) and `:291` (label observation) | Physics, not a tuning knob: NDVI ≤ 0 means NIR ≤ Red, which no canopy produces. |
| `MIN_COHORT_SD` | `0.005` | `1e-6` | `panel.py:96`, applied `:332` | Measured cohort sds run 0.07–0.33, so 0.005 sits over an order of magnitude below any real cohort while being ~5,000× stricter than the previous guard, which guarded against equal floats rather than a degenerate spread. |

**The panel was deliberately not rebuilt.** See §7.9.

### 7.9 Why the panel was not rebuilt

The Sentinel and SAR caches are keyed `{site_id}-{days}` — date-independent, reusable across a
rebuild (`src/argotech/lab/panel.py:143`, `:153`). The meteo cache is keyed
`{lat},{lon},{start},{end}`, both dates derived from `date.today()`. The committed cache holds at
least **five** distinct windows: 2022-05-03/04/08/14 → 2026-08-04/05/09/15 — so the committed panel
was assembled across several days, and sites do not share a common weather window, which matters
because `_climatological_rain_30` (`panel.py:239-246`) uses each site's full range. A rebuild run on
2026-08-27 would request 2022-05-20 .. 2026-08-21, matching none of the cached windows: 100% meteo
cache miss, ~122 network calls (a prior backfill previously exhausted Open-Meteo's daily quota), and
a weather window that would confound the NDVI validity change with a data-window change. **38
committed result files pin `content_hash 6c832dc2…`.** A rebuild invalidates all of them — correct
provenance behaviour, not a bug, but a cost that has to be planned for rather than paid
incidentally. The fix therefore lands at the source and the rebuild is deferred, deliberately.
`collect_site` now takes an optional `end_date` so a future rebuild can be pinned and repeated.

**Consequence, stated plainly: every result in §7.4–7.7 was computed on a panel that still contains
the invalid NDVI observations described in §7.8.** The validity fix applies to future builds only.

### 7.10 What was retired

`argotech.training` is gone (commit `e8f7f4d`, "Export the artifact from the lab, then retire
argotech.training"): `train.py` (684 lines) and `__init__.py` were deleted; `embed.py` moved to
`argotech.lab.embed` and `dataset.py` moved to `argotech.lab.panel` by `git mv` (commit `d80bb40`,
"Move the panel builder into the lab and fingerprint what it emits"; history preserved across 9
pre-move commits).

`argotech.training.train` was the sole producer of `artifacts/agronomic_risk.joblib`.
`src/argotech/lab/export.py` (`export_artifact()`, `:62`) replaces it, and was proven end-to-end
before the deletion: exported to a scratch path, loaded through `src/argotech/models/registry.py`'s
`ModelManager`, passed the contract check, and `model.predict` succeeded on the exact DataFrame shape
`serving/pipeline.py` builds. The production artifact itself was not modified — an empty diff across
the whole session.

`export.py` fits peer stats on the *full* panel deliberately (`export.py:52-62` docstring) — the one
place in `lab/` where that is correct, because a shipped artifact runs at inference time, where there
is no future to leak from. Every other module in `lab/` does the opposite, fitting on training rows
only (§7.2c).

`rank_correlation` and `spearman` were found not to have matching semantics: the former negates the
target and rounds to 4dp on degenerate input, the latter does neither. The test comparing them was
rewritten to reflect that, not repointed to hide it.

### 7.11 A methodological pattern worth recording

Four separate times in this session a passing test was blind to exactly the thing it was named to
guard: the same-row `alpha_hat` leak (§7.2a); the temporal cut whose fixture had no `label_date`
column (§7.2b); a leaky-reproduces-pre-change invariant that only ever ran the spatial split; and a
`within_xy` regression test whose only arm (`zero`) never evaluates `test[features]`. Three of the
four were caught only because deliberate-failure evidence was required for each fix — break the
code, watch the guarding test fail, restore the code, watch it pass. Recording this as a
recommendation: require that evidence for any test guarding a leakage or honesty property, not only
for the four caught here.
