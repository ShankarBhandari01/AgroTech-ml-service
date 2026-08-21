# AgroTech ML Service (`argotech-ai`)

FastAPI inference service for the FarmerXential early warning system. It turns a farmer record — or
a bare pair of GPS coordinates — into a decomposed agronomic risk assessment: hazard from physics
and epidemiology, exposure in currency, vulnerability from coping capacity, plus a Crop Health Index
from Sentinel-2.

Repository: `ShankarBhandari01/AgroTech-ml-service`. Deployed as the `ml` service in the Kotlin
backend's `docker-compose.prod.yml` on a single GCE VM.

> **Design rationale lives in [`docs/model-design.md`](docs/model-design.md).** It covers why the
> model is decomposed the way it is, the measured retraining results, and the roadmap. Read it
> before changing the model or the feature set.

---

## Contents

- [What this service is](#what-this-service-is)
- [How a prediction is produced](#how-a-prediction-is-produced)
- [Architecture](#architecture)
- [API reference](#api-reference)
- [The nightly job](#the-nightly-job)
- [Configuration](#configuration)
- [Local development](#local-development)
- [Docker build and deploy](#docker-build-and-deploy)
- [The model](#the-model)
- [Training](#training)
- [Troubleshooting](#troubleshooting)
- [Known issues](#known-issues)
- [Repository layout](#repository-layout)

---

## What this service is

Not a public-facing API. No authentication, no rate limiting, no published ports in production. Its
only caller is the Kotlin backend (`agri-saas-kotlin-backend`), which reaches it over the internal
compose network at `http://ml:8000` via `FASTAPI_ML_URL`.

| Concern | Owner |
| --- | --- |
| Auth, tenancy, rate limiting | Kotlin backend |
| Circuit breaking / retry / bulkhead around ML calls | Kotlin backend (`FastApiMlClientImpl`, Resilience4j) |
| Feature assembly, agronomy, model inference, risk composition | This service |
| Prediction audit trail and outcome capture | This service (`predictions`, `field_outcomes`) |
| Map-tile / raster visualisation | Kotlin backend (this service uses only the CDSE **Statistical** API) |

The backend calls it from `FastApiMlClientImpl.kt` and `PredictionQueueConsumer`, and degrades to a
static `FALLBACK` payload when unreachable — which is why `ml` is deliberately **not** in the `app`
service's `depends_on`.

---

## How a prediction is produced

Risk is not a single model output. It is composed, and each term is separately computable and
separately checkable:

```
Risk = Hazard × Exposure × Vulnerability
```

| Term | Source | Needs training data? |
| --- | --- | --- |
| **Hazard** — drought, disease, heat | FAO-56 water balance, Wallin/BLITECAST severity values, flowering heat-stress days | No. Physics and epidemiology. |
| **Hazard** — vegetation | By default the field's own peer anomaly carried forward, entering as one more independent hazard via noisy-OR. The trained model is the `VEGETATION_HAZARD_SOURCE=model` alternative | No by default |
| **Exposure** | `area × expected yield × farm-gate price`, in USD | No |
| **Vulnerability** | Weighted coping capacity: irrigation, extension access, credit, inputs, diversification, assets, market access | No |

Two numbers come out, answering different questions:

- `risk_score_percent` (0–100) — expected **loss rate**, comparable across farms of any size. The
  number a farmer sees.
- `expected_loss_usd` — what a triage queue ranks by, and what makes the return on an extension
  visit measurable.

`head_gender` and `household_max_education` are **rejected** by `risk.assess_vulnerability`, not
merely unused — a system that allocates extension visits and credit must not route them by a
protected attribute. They stay available for measuring disparate impact.

---

## Architecture

```mermaid
flowchart LR
    client[Mobile / Web client] --> caddy[Caddy]
    caddy --> app["Kotlin backend<br/>agri-app"]

    subgraph vm["GCE VM — docker compose network"]
        app -->|"POST http://ml:8000/predict/*"| ml["FastAPI inference service<br/>agri-ml"]
        app --> pg[("Postgres")]
        ml -->|"reads farmer_profiles,<br/>writes field_features,<br/>predictions, field_outcomes"| pg
        cron["nightly precompute job"] --> pg
    end

    cron -->|"ERA5 daily + Sentinel-2 + hourly RH"| upstream["Open-Meteo<br/>Copernicus CDSE"]
    ml -.->|"only when no fresh<br/>precomputed row exists"| upstream
    ml -->|"joblib, cached in-process"| art["artifacts/agronomic_risk.joblib"]
    ml -.->|"only if OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is set"| otlp["OTLP collector"]
```

The important line is the dotted one from `ml` to the upstreams. **Predictions normally make no
external calls**: the nightly job precomputes the expensive part into `field_features` and a request
is a database read plus a model call. Live computation remains as the fallback for a field with no
fresh row — a farm registered this morning still gets a prediction today, it just pays the latency
once.

Both paths call the same `pipeline.gather_upstream`, so a precomputed prediction and a live one are
the same computation.

### Prediction request flow

```mermaid
sequenceDiagram
    autonumber
    participant BE as Kotlin backend
    participant API as FastAPI router
    participant PS as PredictionsService
    participant PG as Postgres

    BE->>API: POST /predict/farmer {farmer_id}
    API->>PS: predict()
    PS->>PG: farmer_profiles ⋈ farmers_ml_profiles
    alt no row
        PG-->>PS: empty
        PS-->>BE: 404 Farmer not found
    else row found
        PS->>PG: SELECT field_features (latest, < 48h old)
        alt fresh row
            PG-->>PS: features + context
        else missing or stale
            PS->>PS: gather_upstream() — ERA5, Sentinel-2, hourly RH, climatology
        end
        PS->>PS: model → vegetation hazard, domain.risk → H × E × V
        PS->>PG: INSERT predictions (features, version, score) → prediction_id
        PS-->>BE: PredictionResponse
    end
```

---

## API reference

Base URL in production: `http://ml:8000` (compose-internal). No auth on any route.

| Method | Path | Body | Notes |
| --- | --- | --- | --- |
| `GET` | `/health` | — | `{"status":"ok"}`. Consumed by the compose healthcheck. **Does not verify the model loads** — see Known issues |
| `POST` | `/predict/farmer` | `FarmerPredictionRequest` | Loads the farmer from Postgres. 404 if absent, 422 without coordinates, 503 if upstreams are down on a live-path prediction |
| `POST` | `/predict/coldstart` | `CoordinatesColdStartPredictionRequest` | No farmer lookup. `field_id` is `coldstart-<lat:.4f>-<lon:.4f>` |
| `GET` | `/predict/crop-health` | `?latitude=&longitude=` | Crop Health Index from Sentinel-2 against the field's own 12-month history. Deterministic, no model |
| `POST` | `/outcomes` | `OutcomeRequest` | Record what an agent found / was diagnosed / was harvested. **The label stream** |
| `GET` | `/outcomes/label-count` | `?horizon_days=30` | How many prediction↔outcome pairs exist |
| `GET` | `/docs`, `/openapi.json` | — | FastAPI defaults, not disabled |

`model_name` / `model_alias` are gone from the request schemas. They named an MLflow registry entry
that no loader has read since the registry rewrite, and having them on the wire suggested a
per-request model choice that does not exist. Pydantic ignores unknown fields, so a Kotlin client
still sending them keeps working unchanged. What answered a given request is reported back in
`model_version`; which source is configured service-wide is `VEGETATION_HAZARD_SOURCE`.

### Response — `PredictionResponse`

| Field | Notes |
| --- | --- |
| `field_id` | Farmer id, or the synthesised coldstart id |
| `prediction_id` | Row id in `predictions`. **Pass this back on `POST /outcomes`** — it is the join that produces training labels |
| `model_version` | Artifact version behind this prediction, or `"none"` when no satellite scene was available |
| `feature_source` | `"precomputed"` or `"live"` |
| `features_computed_at` | When the precomputed row was built; null on the live path |
| `crop_type`, `phenology_stage` | Stage is derived from GDD accumulated since the rainy-season onset |
| `prediction`, `priority_label` | `0`/`1`/`2`, Low/Medium/High Priority |
| `risk_score_percent` | Expected loss rate, 0–100 |
| `probabilities` | Calibrated model probabilities. All in `low` when the model did not contribute |
| `top_risk_factors` | Drivers, most specific first |
| `inference` | `{risk_level, dominant_hazard, probability, primary_drivers, model_contributed}` |
| `risk_assessment` | `{risk_score, expected_loss_usd, value_at_risk_usd, hazard{...}, vulnerability{...}}` |
| `crop_health` | `{score, vigour, moisture, anomaly, status}`. Null with no cloud-free scene |
| `spatiotemporal_indices` | `{ndvi, ndwi, evi, vci, canopy_stress_status, source, sensing_date}`. `ndwi` is the B08/B11 formula, i.e. NDMI — kept under this name for API compatibility |
| `microclimate_metrics` | Water satisfaction and deficit, dry spell, real rainfall anomaly, heat days, cumulative DSV, spray threshold, GDD since onset |
| `recommended_action` | Derived from the dominant hazard and the phenology stage |

Example — `POST /predict/coldstart`:

```json
{ "latitude": 10.85, "longitude": 7.66, "crop_type": "Maize", "farm_size": 2.0 }
```

```json
{
  "field_id": "coldstart-10.8500-7.6600",
  "prediction_id": 10482,
  "model_version": "agro-20260811T101300Z",
  "feature_source": "precomputed",
  "crop_type": "Maize",
  "phenology_stage": "Grain Fill",
  "prediction": 1,
  "priority_label": "Medium Priority",
  "risk_score_percent": 48.9,
  "probabilities": { "low": 0.672, "medium": 0.192, "high": 0.137 },
  "inference": {
    "risk_level": "ELEVATED",
    "dominant_hazard": "disease",
    "probability": 0.574,
    "model_contributed": true
  },
  "risk_assessment": {
    "risk_score": 48.9,
    "expected_loss_usd": 217.55,
    "value_at_risk_usd": 737.5,
    "hazard": { "drought": 0.0, "disease": 0.444, "heat": 0.0, "vegetation": 0.233,
                "combined": 0.574, "dominant": "disease" },
    "vulnerability": { "score": 0.713, "coping_capacity": 0.287,
                       "gaps": ["has_irrigation", "has_extension_access", "received_credit"] }
  },
  "crop_health": { "score": 36.2, "vigour": 0.335, "moisture": 0.206,
                   "anomaly": 0.556, "status": "Stressed" },
  "spatiotemporal_indices": { "ndvi": 0.31, "ndwi": -0.077, "evi": 0.219, "vci": 33.5,
                              "canopy_stress_status": "Stressed",
                              "source": "sentinel-2", "sensing_date": "2026-08-06" },
  "microclimate_metrics": { "water_satisfaction_30d": 1.0, "water_deficit_30d_mm": 0.0,
                            "longest_dry_spell_days": 1, "rainfall_anomaly_30d_mm": 67.4,
                            "heat_stress_days": 0, "cumulative_dsv": 8,
                            "spray_threshold_reached": false, "gdd_since_onset": 1088.8 },
  "recommended_action": "Blight severity is accumulating on Maize at Grain Fill. Scout the lower canopy for lesions now and have fungicide staged before the threshold is reached."
}
```

### Recording an outcome

```bash
curl -X POST localhost:8000/outcomes -H 'content-type: application/json' -d '{
  "field_id": "farmer-123",
  "prediction_id": 10482,
  "observed_at": "2026-08-20T09:00:00Z",
  "outcome_type": "agent_visit",
  "stress_confirmed": true,
  "diagnosis": "Northern corn leaf blight, lower canopy",
  "reported_by": "agent-44"
}'
```

Every advisory should be reconcilable against what an agent actually found. Nothing in the
supervised-model roadmap is possible until these accumulate.

---

## The nightly job

```bash
python -m argotech.jobs.precompute            # all fields
python -m argotech.jobs.precompute --limit 1  # smoke test; also creates the schema
```

```
0 2 * * *  python -m argotech.jobs.precompute
```

Single instance — it owns the DDL (`store.ensure_schema`) and API startup deliberately does not, so
N replicas cannot race. Sequential with a 1.5 s pause between fields: both upstreams are free and
rate-limited, and the job has all night. Parallelism here is what triggered 429s during training.

It prints a summary worth alerting on:

```
Done: {'fields': 412, 'ok': 405, 'degraded': 63, 'failed': 7, 'pruned': 380, 'degraded_rate': 0.156}
```

`degraded_rate` is the share of fields with no cloud-free Sentinel-2 scene — no canopy signal, no
model contribution, physics-only assessment. It is a genuine quality regression that no latency or
error dashboard will show you.

Feature rows are pruned after 90 days. They are kept that long because a prediction under
investigation must be reproducible from the row that produced it.

---

## Configuration

All settings live in `src/argotech/config.py` (`pydantic-settings`, `env_file=".env"`,
`extra="ignore"`). In Docker there is no `.env` file — `.dockerignore` excludes it — so every value
comes from the container environment or the default.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://postgres:password@localhost:5432/agrotech` | Reads the backend's farmer tables; owns `field_features`, `predictions`, `field_outcomes`. A `postgres://` prefix is rewritten |
| `AGRONOMIC_MODEL_PATH` | `artifacts/agronomic_risk.joblib` | Model artifact, loaded by relative path from the working directory. Only read when `VEGETATION_HAZARD_SOURCE=model` |
| `VEGETATION_HAZARD_SOURCE` | `model` | Where the peer anomaly fed to the hazard map comes from. `model` predicts it 30 days ahead (`HistGradientBoostingRegressor` on `forward_z`); `persistence` carries the field's own observed value forward. Same mapping either way. See [The model](#the-model) |
| `SENTINEL_CLIENT_ID` | `""` | CDSE OAuth client id. Blank disables the satellite path entirely |
| `SENTINEL_CLIENT_SECRET` | `""` | CDSE OAuth client secret |
| `SENTINEL_TOKEN_URL` | CDSE Keycloak token endpoint | Override for commercial Sentinel Hub |
| `SENTINEL_STATS_URL` | `https://sh.dataspace.copernicus.eu/api/v1/statistics` | Statistical API endpoint |

Read from the environment directly, not through `Settings`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | unset | When set, adds a `BatchSpanProcessor` with an OTLP/HTTP exporter. Left unset in production so the exporter does not retry against a dead endpoint forever |
| `LOG_LEVEL` | `INFO` | Root log level. `INFO` gives one arrival and one completion line per request (with status and latency), the parsed request body, a response summary, and which source answered the vegetation hazard |

`USE_LOCAL_MODEL`, `MLFLOW_TRACKING_URI`, `MODEL_NAME` and `MODEL_ALIAS` are gone with the MLflow
path, and have now been dropped from `.env` as well — they had outlived the loader that read them by
several commits. `extra="ignore"` means a compose file still setting them is harmless.

Without Sentinel credentials the service still works: `crop_health` is null, `probabilities` collapse
to `low: 1.0`, and the drought/disease/heat hazards carry the assessment on their own.

---

## Local development

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[train,dev]'

uvicorn argotech.serving.main:app --host 0.0.0.0 --port 8000 --reload
```

Run from the **repository root** — the model artifact is loaded by relative path.

```bash
python tests/test_domain.py     # agronomy, indices, risk composition
python tests/test_store.py      # freshness rule, JSONB round-trip fidelity
```

Smoke test:

```bash
curl -s localhost:8000/health

curl -s "localhost:8000/predict/crop-health?latitude=10.85&longitude=7.66" | python -m json.tool

curl -s -X POST localhost:8000/predict/coldstart -H 'content-type: application/json' \
  -d '{"latitude":10.85,"longitude":7.66}' | python -m json.tool
```

`/predict/coldstart` needs no farmer row, but does need `DATABASE_URL` to reach a Postgres for the
feature and audit tables. It degrades rather than failing if they are absent.

---

## Docker build and deploy

```bash
gcloud builds submit \
  --tag europe-west2-docker.pkg.dev/farmerxential-backend/agri/ml:$(git rev-parse --short HEAD) \
  --region=europe-west2 .
```

```bash
# /opt/agri/.env
ML_IMAGE=europe-west2-docker.pkg.dev/farmerxential-backend/agri/ml:<tag>
```

```bash
sudo docker compose -f docker-compose.prod.yml pull ml
sudo docker compose -f docker-compose.prod.yml up -d ml
```

### Image layout

| Layer | Why |
| --- | --- |
| `python:3.11-slim` | Base |
| `apt-get install libgomp1` | scikit-learn's `HistGradientBoosting` links against libgomp at runtime; nothing else needs a system package |
| `COPY pyproject.toml` then `pip install .` | Dependency layer rebuilds only when the pins change |
| `COPY artifacts/agronomic_risk.joblib` | Loaded by relative path from `WORKDIR` |
| `COPY src src` then `pip install --no-deps .` | Application code |
| `useradd --system ml` + `USER ml` | Non-root runtime |
| `CMD uvicorn argotech.serving.main:app --port 8000` | Fixed port; compose talks to it internally |

`.gcloudignore` is written explicitly because gcloud otherwise derives one from `.gitignore`, which
lists `venv/` but not `.venv/` — a 1 GB upload on every build.

### Compose service

```yaml
ml:
  image: ${ML_IMAGE}
  container_name: agri-ml
  restart: always
  environment:
    DATABASE_URL: postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/${DB_NAME}
    SENTINEL_CLIENT_ID: ${SENTINEL_CLIENT_ID:-}
    SENTINEL_CLIENT_SECRET: ${SENTINEL_CLIENT_SECRET:-}
  healthcheck:
    test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
    interval: 15s
    timeout: 5s
    retries: 5
    start_period: 60s
  depends_on:
    postgres: { condition: service_healthy }
```

The healthcheck uses `python`, not `curl` or `wget` — the slim base image ships neither.

---

## The model

One artifact: `artifacts/agronomic_risk.joblib` (~1.5 MB). A `HistGradientBoostingClassifier` under
a cross-fit `CalibratedClassifierCV(method="sigmoid", cv=5)`. The bundle carries its own
`feature_columns` and `version`, so the column contract is explicit at load time and every persisted
prediction is traceable to the artifact behind it.

**It is one hazard term, not the risk score**, and as of `agro-20260816` it is **off by default** —
`VEGETATION_HAZARD_SOURCE=persistence`.

The earlier claim that it beat persistence rested on precision@25, which reads only the top 25 of a
~1,100-field fold and carries a standard error of about 0.098. Scored on the whole ranking with
Spearman's rho against the continuous `forward_z`, leave-one-cluster-out:

| | model | persistence | climatology |
| --- | --- | --- | --- |
| mean rho, spatially blocked | 0.191 | **0.351** | 0.341 |
| mean rho, forward-chaining temporal | 0.242 | **0.426** | 0.465 |
| folds won (of 9) | 0 | 9 | — |

rho has a standard error near 0.030 on a fold that size, so the 0.161 gap is ~5 standard errors,
where the precision@25 gaps are one or less. The model loses to a rule that carries the field's own
peer anomaly forward, in every fold.

The cause is structural rather than a tuning failure. The label is a z-score *within* a
(cluster, date) cohort, so it cancels whatever the cohort shares — and the weather block is exactly
that. Weather features retain 19.7% of their variance inside a cohort and correlate 0.027 with the
target; canopy features retain 74.9% and correlate 0.127. The strongest single feature,
`ndvi_z_peer`, reaches 0.256, and it is the same quantity persistence uses on its own. An
unregularised fit reaches macro F1 1.000 in-sample, so capacity was never the constraint.

Set `VEGETATION_HAZARD_SOURCE=model` to A/B it. Making the model worth its place needs features that
vary *between neighbouring fields* — soil, irrigation, planting date, management — not more weather
and not a different estimator.

Permutation importance on held-out ground is led by `ndmi` and `evi`, with real contributions from
the agronomy (`et0_90`, `dry_spell_30`, `stage_kc`) — at half this sample size the model was
effectively a smoothed persistence model, and [`docs/model-design.md` §9](docs/model-design.md)
records both results and why they differ.

Do not promote it to the primary signal without new evidence on the same protocol.

---

## Training

```bash
pip install -e '.[train]'

python -m argotech.training.dataset --sites 32 --years 4   # → data/training_set.parquet
python -m argotech.training.train                          # → artifacts/agronomic_risk.joblib
```

The dataset is built from real measurements only: ERA5 daily reanalysis over the 90 days *before*
each prediction date, passed through `argotech.domain`, plus the Sentinel-2 canopy state. The label
is the peer-standardised NDVI anomaly one 30-day interval *ahead* — a future satellite observation,
so no feature can determine its own target. Leakage controls are documented at the top of
`argotech/training/dataset.py`.

The builder also fetches **Sentinel-1 backscatter** (`.cache/sentinel_sar/`, keyed separately so it
can be added without invalidating the optical cache). Radar sees through cloud, which is the point:
optical gaps cluster in the rainy season. It is additive — a site with no S1 coverage still yields
samples, with the radar block as NaN. To measure what it buys, ablate it on the same parquet:

```bash
python -m argotech.training.train --data data/training_set.parquet             # radar on
python -m argotech.training.train --data data/training_set.parquet --no-radar  # control
```

Compare on the *same* file. The builder keys its window off `date.today()`, so two builds made on
different days are not a controlled comparison.

### Frozen Presto embeddings (experimental)

```bash
pip install -e '.[train]'                     # adds torch + einops, training-only
python -m argotech.training.embed --data data/training_set.parquet \
                                  --out  data/presto_embeddings.parquet
python -m argotech.training.train --data data/training_set.parquet \
                                  --embeddings data/presto_embeddings.parquet
```

[Presto](https://arxiv.org/abs/2304.14065) is a 402K-parameter transformer pre-trained on
remote-sensing pixel timeseries — 12 monthly steps × 17 channels. It is **vendored**
(`models/_presto_vendored.py`, MIT) rather than installed: the published package pins `torch==2.0`
and imports `earthengine-api` at init, so it cannot be installed alongside this project. Weights go
in `.cache/presto/`; fetch them once with

```bash
curl -L -o .cache/presto/default_model.pt \
  https://raw.githubusercontent.com/nasaharvest/presto/main/data/default_model.pt
```

`--no-bands` masks the ten optical reflectance channels, which is the cheaper variant that needs no
raw-band fetch. Results and the unit-conversion traps are in
[`docs/model-design.md` §9.1 F](docs/model-design.md).

Both upstreams are cached on disk under `.cache/`. A full build takes roughly 40 minutes cold and
seconds warm. **Open-Meteo's archive endpoint has a daily quota** that one full build can exhaust;
if it starts returning 429 across the board, resume tomorrow — the cache preserves progress.

`train.py` prints the whole evaluation, not a headline number: leave-one-cluster-out, forward
chaining, **three** baselines (majority, persistence, site climatology), a linear shift-robustness
arm beside the boosted trees, a decision-rule sweep, permutation importance on held-out ground,
expected calibration error, and precision@k. Results are written to `artifacts/metrics.json`.

Two transforms are applied to the parquet at train time and need no dataset rebuild, because both
derive from columns already in it:

- **Cluster-relative features** — a within-cluster z-score twin for each regionally-signatured
  feature. The label is already standardised against the peer cohort; these stop the inputs handing
  the model raw cluster identity. Label-free, so a held-out cluster normalising against its own
  statistics is not leakage — it is the mechanism.
- **Site climatology** — each field's mean prior peer anomaly, as a third baseline. It asks "is this
  field *usually* weak", where persistence asks "is it weak *right now*".

A retrain that does not beat the incumbent and all three baselines on blocked CV should not be
promoted — and "beat" has to name a metric, because macro F1 and precision@k currently disagree.
See [`docs/model-design.md` §6](docs/model-design.md).

---

## Troubleshooting

### `FileNotFoundError: Agronomic model 'artifacts/agronomic_risk.joblib' not found`

The artifact is not in the working directory. In Docker it is copied to `/app/artifacts`; locally,
run `uvicorn` from the repository root, or build it with the two training commands above.

### `crop_health` is null and `probabilities` are `{low: 1.0, ...}`

No cloud-free Sentinel-2 scene, so the model did not contribute and the assessment is physics-only.
`inference.model_contributed` is `false` and `spatiotemporal_indices.source` is `"unavailable"`.
Check in order:

1. `SENTINEL_CLIENT_ID` / `SENTINEL_CLIENT_SECRET` are set on the container.
2. Logs for `[SentinelClient] statistics query failed: ...`.
3. Whether any scene under 40 % cloud exists for that AOI in the last 365 days. In the rainy season
   this is common and expected — watch `degraded_rate` from the nightly job rather than individual
   requests.

### Predictions are slow (seconds, not milliseconds)

`feature_source` is `"live"`, meaning no fresh `field_features` row. Either the nightly job has not
run, it failed for that field, or the row is over 48 h old. Check the job's last summary.

### `[store] prediction not persisted: relation "predictions" does not exist`

The nightly job has never run, so the schema was never created. Predictions still serve — the audit
write is best-effort by design. Run `python -m argotech.jobs.precompute --limit 1` once.

### `404 Farmer '<id>' not found in database`

`/predict/farmer` requires a row in `farmer_profiles` for that `user_id`. `farmers_ml_profiles` is a
`LEFT JOIN`, so a farmer with no ML profile still works — vulnerability falls back to defaults.

### `422 Prediction requires latitude and longitude`

The farmer row exists but has no coordinates. Everything in this service is geospatial; there is no
meaningful prediction without a location.

### Build context upload is enormous / slow

`.venv/` is roughly 1 GB. Confirm `.gcloudignore` exists and lists `.venv/`, `.cache/` and `data/`.

### Resolved production incidents, kept for context

- **`ModuleNotFoundError: No module named '_loss'`** — unpinned `requirements.txt` resolved
  scikit-learn 1.9.0 against pickles trained under 1.6.1. Pins now live in `pyproject.toml`. Still
  latent: the artifact is a joblib pickle, so bump scikit-learn and retrain in one change. Exporting
  to ONNX removes the class of failure.
- **`ModuleNotFoundError: No module named 'torch'` crash loop** — `torch` was imported at module
  scope by a PyTorch fusion layer that was never instantiated. Both the class and the dependency are
  gone.

---

## Known issues

| # | Issue | Impact | Fix |
| --- | --- | --- | --- |
| 1 | `/health` returns `ok` without checking that the model loads | A container with a missing artifact passes its healthcheck and fails on the first prediction | Load the artifact in the readiness check; split liveness from readiness |
| 2 | No circuit breaker on *our* upstream calls | The backend has Resilience4j around calls to us; we have nothing around Open-Meteo and CDSE. The nightly job absorbs most of this, the live fallback does not | Token bucket plus breaker per upstream |
| 3 | `.cache/` is container-local | Lost on restart, not shared between replicas. Affects the training build and the climatology fetch on the live path | Move the climatology cache into Postgres |
| 4 | The CDSE OAuth token is cached per process | N replicas make N token calls against a rate-limited endpoint | Shared cache, or accept it at current replica count |
| 5 | Direct read access to the backend's schema | A backend migration can break inference silently | Have the backend pass the farmer payload in the request body |
| 6 | No auth on any route | Safe only because the container publishes no ports | Keep it network-only; add a shared secret if ever exposed |
| 7 | Model artifact is baked into the image | A model update requires an image rebuild | Pull from GCS at startup, keyed by version |
| 8 | Farm-gate prices are a static per-crop table | `expected_loss_usd` drifts from reality | Wire to a market-price feed |
| 9 | `has_irrigation` is not in the schema | Always 0, so vulnerability overstates every irrigated farm | Add to `farmers_ml_profiles` |
| 10 | Images tagged by date | A tag does not identify the code that produced it | Tag with `$(git rev-parse --short HEAD)` |
| 11 | No serving contract test | The Kotlin client's contract can drift silently | Golden-response test with mocked upstreams |

---

## Repository layout

```
.
├── Dockerfile / Procfile
├── pyproject.toml                  # single dependency source
├── docs/model-design.md            # analysis, design, measured results, roadmap
├── artifacts/
│   ├── agronomic_risk.joblib       # the served model (in image)
│   └── metrics.json                # full evaluation of the shipped artifact
├── tests/
│   ├── test_domain.py
│   └── test_store.py
└── src/argotech/
    ├── config.py
    ├── domain/                     # PURE: no I/O, no network, no DB. The agronomy.
    │   ├── indices.py              #   spectral indices, VCI, anomaly, Crop Health Index
    │   ├── agronomy.py             #   GDD, season onset, phenology, FAO-56, DSV, FAW
    │   └── risk.py                 #   Hazard × Exposure × Vulnerability
    ├── data/                       # I/O adapters, one per upstream
    │   ├── meteo.py                #   Open-Meteo: ERA5 archive + forecast, disk-cached
    │   ├── sentinel.py             #   CDSE Statistical API
    │   ├── db.py                   #   engine + session
    │   └── store.py                #   field_features, predictions, field_outcomes
    ├── features/agronomic.py       # THE feature builder — training and serving both call it
    ├── models/registry.py          # artifact loading
    ├── jobs/precompute.py          # nightly feature job
    ├── serving/                    # FastAPI app, routers, schemas, pipeline
    └── training/                   # dataset construction, training + evaluation
```

The load-bearing boundary is `domain/`: pure functions over plain values, no network, no database,
no model artifact. That is what makes the agronomy testable in milliseconds and reusable from both
the training pipeline and the serving path without duplication. Everything else is an adapter
around it.
