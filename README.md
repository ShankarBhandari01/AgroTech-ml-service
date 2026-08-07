# AgroTech ML Service (`argotech-ai`)

FastAPI inference service for the FarmerXential Early Warning System. It turns a farmer record — or
a bare pair of GPS coordinates — into a 3-class agronomic risk prediction (`0` normal, `1` elevated,
`2` critical), enriched with Sentinel-2 spectral indices, Open-Meteo microclimate lag features, and a
cross-attention fusion score.

Repository: `ShankarBhandari01/AgroTech-ml-service`. Deployed as the `ml` service in the Kotlin
backend's `docker-compose.prod.yml` on a single GCE VM.

---

## Contents

- [What this service is](#what-this-service-is)
- [Architecture](#architecture)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Local development](#local-development)
- [Docker build and deploy](#docker-build-and-deploy)
- [Models](#models)
- [Training](#training)
- [Troubleshooting](#troubleshooting)
- [Known issues](#known-issues)

---

## What this service is

This is not a public-facing API. It has no authentication, no rate limiting, and no published ports
in production. Its only caller is the Kotlin backend (`agri-saas-kotlin-backend`), which reaches it
over the internal compose network at `http://ml:8000` via `FASTAPI_ML_URL`.

| Concern | Owner |
| --- | --- |
| Auth, tenancy, rate limiting, persistence of predictions | Kotlin backend |
| Circuit breaking / retry / bulkhead / timeout around ML calls | Kotlin backend (`FastApiMlClientImpl`, Resilience4j) |
| Feature assembly, model loading, inference, risk narrative | This service |
| Map-tile / raster visualisation | Kotlin backend (this service only uses the CDSE **Statistical** API) |

The backend calls it from `FastApiMlClientImpl.kt` and from `PredictionQueueConsumer`, and degrades
to a static `FALLBACK` payload when this service is unreachable — which is why `ml` is deliberately
**not** in the `app` service's `depends_on`.

---

## Architecture

```mermaid
flowchart LR
    client[Mobile / Web client] --> caddy[Caddy]
    caddy --> app["Kotlin backend<br/>agri-app"]

    subgraph vm["GCE VM — docker compose network"]
        app -->|"POST http://ml:8000/predict/*"| ml["FastAPI inference service<br/>agri-ml"]
        app --> pg[("Postgres")]
        ml -->|"farmer_profiles<br/>farmers_ml_profiles<br/>farmers_crops"| pg
    end

    ml -->|"joblib.load, cached in-process"| pkl["farmerxential_model.pkl<br/>farmerxential_powerful_model.pkl"]
    ml -->|"CDSE Statistical API<br/>NDVI / NDWI / EVI"| sentinel["Copernicus Data Space<br/>Sentinel-2 L2A"]
    ml -->|"7-day hourly forecast + past_days"| meteo["Open-Meteo"]
    ml -.->|"only if OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is set"| otlp["OTLP collector"]
```

Notes on the diagram:

- The `ml` container has **no published ports**. Nothing outside the compose network can reach it.
- Model loading is local-file `joblib.load` by default (`USE_LOCAL_MODEL=true`). The MLflow registry
  path exists in `ModelManager` but is not used in production.
- Sentinel is optional. With no credentials, `SentinelClient.enabled` is `False` and the pipeline
  falls back to deterministic synthetic indices tagged `source: "modelled"`.
- OpenTelemetry tracing is always instrumented; **span export** is opt-in via
  `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`. No collector runs in the deployed stack, so nothing is
  exported there.

### Prediction request flow

```mermaid
sequenceDiagram
    autonumber
    participant BE as Kotlin backend
    participant API as FastAPI router
    participant PS as PredictionsService
    participant PG as Postgres
    participant OM as Open-Meteo
    participant S2 as CDSE Sentinel-2
    participant MM as ModelManager

    BE->>API: POST /predict/farmer {farmer_id, model_name, model_alias}
    API->>PS: predict()
    PS->>PG: SELECT farmer_profiles JOIN farmers_ml_profiles
    alt no row
        PG-->>PS: empty
        PS-->>BE: 404 Farmer not found
    else row found
        PG-->>PS: profile + ML profile + crop_diversity_score
        PS->>OM: 7-day hourly temp / RH / soil moisture / rain
        OM-->>PS: hourly arrays (168x4 sequence)
        PS->>S2: Statistical API, 30-day window, maxCloudCoverage 40
        S2-->>PS: mean NDVI / NDWI / EVI (or none, then synthetic)
        PS->>PS: CrossAttentionFusionLayer.forward(spatial 4-vec, temporal seq)
        PS->>MM: predict + predict_proba on 27-column frame
        MM->>MM: get_model (cached) then _align_features to feature_names_in_
        MM-->>PS: class label + class probabilities
        PS-->>BE: PredictionResponse
    end
```

`/predict/coldstart` follows the same path but skips the Postgres step: it builds a
`SimpleNamespace` stand-in from the request body and calls `predict_from_farmer_data` directly.

### Where the numbers come from

| Output field | Source |
| --- | --- |
| `prediction`, `probabilities`, `risk_score_percent` | `HybridSpatiotemporalEnsemble` in the `.pkl` (XGBoost + HistGB + RF + ExtraTrees stacked under a `LogisticRegression`, sigmoid-calibrated) |
| `spatiotemporal_indices` | CDSE Sentinel-2 Statistical API, else deterministic `sin/cos` synthetic model |
| `microclimate_metrics` | Open-Meteo hourly, aggregated in `_fetch_microclimate_weather_data` |
| `cross_attention_fusion` | NumPy `CrossAttentionFusionLayer` (untrained fixed random projections, seed 42) |
| `top_risk_factors`, `inference`, `recommended_action` | Rule thresholds in `_generate_risk_drivers` and `predict_from_farmer_data`, **not** the model |

`risk_level` / `disease_type` / `recommended_action` are deterministic if/else branches over
`prediction` and `risk_score`. They are a narrative layer, not a second model.

---

## API reference

Base URL in production: `http://ml:8000` (compose-internal). No auth on any route.

| Method | Path | Request body | Response | Notes |
| --- | --- | --- | --- | --- |
| `GET` | `/health` | — | `{"status": "ok"}` | Consumed by the compose healthcheck and the backend's `getHealth()` |
| `POST` | `/predict` | `PredictionRequest` | — | **Always returns 400.** Placeholder that tells you to use `/predict/farmer` or `/predict/coldstart` |
| `POST` | `/predict/farmer` | `FarmerPredictionRequest` | `PredictionResponse` | Loads the farmer from Postgres. 404 if not found, 400 if the pipeline throws, 500 on unexpected errors |
| `POST` | `/predict/coldstart` | `CoordinatesColdStartPredictionRequest` | `PredictionResponse` | No DB lookup. `field_id` is synthesised as `coldstart-<lat:.4f>-<lon:.4f>` |
| `GET` | `/predict/crop-health` | — | `{average_ndvi, pest_risk_level, disease_probability, yield_forecast_mt_ha}` | **Mock.** Returns `random` values on every call; no model, no DB, no satellite |
| `GET` | `/docs`, `/openapi.json` | — | — | FastAPI defaults, not disabled |

### Request schemas

`FarmerPredictionRequest` — `POST /predict/farmer`

| Field | Type | Default |
| --- | --- | --- |
| `farmer_id` | `str` | required |
| `model_name` | `str?` | `"farmerXential_powerful_model"` |
| `model_alias` | `str?` | `"prod"` |

`CoordinatesColdStartPredictionRequest` — `POST /predict/coldstart`

| Field | Type | Default |
| --- | --- | --- |
| `latitude` | `float` | required |
| `longitude` | `float` | required |
| `state` | `str?` | `"Kaduna"` |
| `crop_type` | `str?` | `"Maize"` |
| `farm_size` | `float?` | `1.5` |
| `model_name` | `str?` | `"farmerXential_powerful_model"` |
| `model_alias` | `str?` | `"prod"` |

`PredictionRequest` — the flat 24-field survey payload (`yield_value`, `has_extension_access`,
`household_max_education`, `shock_level`, `received_assistance`, `used_fertilizer`, `land_size`,
`household_size`, `zone`, `transport_cost`, `dependency_ratio`, `asset_score`,
`postharvest_activity_score`, `crop_loss_risk_score`, `crop_diversity_score`,
`digital_access_score`, `has_veterinary_access`, `market_access_score`, `is_rural`,
`rainfall_anomaly`, `drought_risk`, `cultivates_crops`, `received_credit`, `head_gender`). It is
declared and validated but never used for inference — `/predict` rejects everything.

### Response schema

`PredictionResponse`:

| Field | Type | Notes |
| --- | --- | --- |
| `field_id` | `str` | The `farmer_id` from the request, or the synthesised coldstart id |
| `crop_type` | `str` | First crop on the farmer record, `"Maize"` when absent |
| `phenology_stage` | `str` | Hardcoded `"Vegetative / Flowering"` |
| `prediction` | `int` | `0` / `1` / `2` |
| `priority_label` | `str` | `Low` / `Medium` / `High Priority` |
| `risk_score_percent` | `float` | `P(class 2) * 100`, one decimal |
| `probabilities` | `{low, medium, high}` | Calibrated class probabilities |
| `top_risk_factors` | `string[]` | All triggered rule drivers |
| `inference` | `{risk_level, disease_type, probability, primary_drivers}` | `primary_drivers` is the first 4 of `top_risk_factors` |
| `spatiotemporal_indices` | `{ndvi, ndwi, evi, canopy_stress_status, source}` | `source` is `"sentinel-2"` or `"modelled"` |
| `microclimate_metrics` | `{rh_85_consecutive_hrs, incubation_hours, soil_water_deficit_72h, rainfall_anomaly}` | |
| `cross_attention_fusion` | `{fusion_score, peak_incubation_hour, fusion_status}` | |
| `recommended_action` | `str` | Rule-derived |
| `created_at` | `datetime` | `datetime.utcnow()` at response construction |

Example — `POST /predict/coldstart`:

```json
{
  "latitude": 10.52,
  "longitude": 7.44,
  "state": "Kaduna",
  "crop_type": "Maize",
  "farm_size": 2.0
}
```

```json
{
  "field_id": "coldstart-10.5200-7.4400",
  "crop_type": "Maize",
  "phenology_stage": "Vegetative / Flowering",
  "prediction": 2,
  "priority_label": "High Priority",
  "risk_score_percent": 87.5,
  "probabilities": { "low": 0.05, "medium": 0.075, "high": 0.875 },
  "top_risk_factors": [
    "Relative humidity > 85% for 18 consecutive hours",
    "Fungal incubation window active for 22 hours (18–24°C)",
    "Leaf water stress detected (NDWI: 0.08)",
    "Low crop yield (< 1.5 tons/ha)"
  ],
  "inference": {
    "risk_level": "CRITICAL",
    "disease_type": "Late Blight / Fungal Leaf Rust in Maize",
    "probability": 0.875,
    "primary_drivers": [
      "Relative humidity > 85% for 18 consecutive hours",
      "Fungal incubation window active for 22 hours (18–24°C)",
      "Leaf water stress detected (NDWI: 0.08)",
      "Low crop yield (< 1.5 tons/ha)"
    ]
  },
  "spatiotemporal_indices": {
    "ndvi": 0.42,
    "ndwi": 0.08,
    "evi": 0.38,
    "canopy_stress_status": "High Water Stress",
    "source": "sentinel-2"
  },
  "microclimate_metrics": {
    "rh_85_consecutive_hrs": 18,
    "incubation_hours": 22,
    "soil_water_deficit_72h": 0.062,
    "rainfall_anomaly": 2.4
  },
  "cross_attention_fusion": {
    "fusion_score": 0.1873,
    "peak_incubation_hour": 91,
    "fusion_status": "Active Spatiotemporal Cross-Attention Align"
  },
  "recommended_action": "Apply protective copper-based fungicide spray within 24–48 hours and dispatch extension agent for immediate field inspection.",
  "created_at": "2026-08-07T09:41:02.113244"
}
```

---

## Configuration

All settings live in `src/services/inferenceService/app/core/config.py` (`pydantic-settings`,
`env_file=".env"`, `extra="ignore"`). In Docker there is no `.env` file — it is excluded by
`.dockerignore` — so every value comes from the container environment or the default.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | Registry URI. Only consulted when `USE_LOCAL_MODEL` is false or the model name is not one of the two local ones |
| `MODEL_NAME` | `farmerXential_model` | Default registry model name. Note the per-request bodies default to `farmerXential_powerful_model` instead |
| `MODEL_ALIAS` | `prod` | Default registry alias |
| `USE_LOCAL_MODEL` | `true` | Load `.pkl` from disk rather than MLflow. `true` in production |
| `DATABASE_URL` | `postgresql://postgres:password@localhost:5432/agrotech` | Read-only queries against the backend's Postgres. A `postgres://` prefix is rewritten to `postgresql://` |
| `SENTINEL_CLIENT_ID` | `""` | CDSE OAuth client id. Blank disables real Sentinel fetch |
| `SENTINEL_CLIENT_SECRET` | `""` | CDSE OAuth client secret. Blank disables real Sentinel fetch |
| `SENTINEL_TOKEN_URL` | CDSE Keycloak token endpoint | Override for commercial Sentinel Hub |
| `SENTINEL_STATS_URL` | `https://sh.dataspace.copernicus.eu/api/v1/statistics` | Statistical API endpoint |

Read directly from the environment, not through `Settings`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | unset | When set, adds a `BatchSpanProcessor` with an OTLP/HTTP exporter. Left unset deliberately in production so the exporter does not retry against a dead endpoint forever |

What the production compose actually sets on the `ml` service: `DATABASE_URL`, `USE_LOCAL_MODEL=true`,
`SENTINEL_CLIENT_ID`, `SENTINEL_CLIENT_SECRET`. Everything else runs on defaults.

---

## Local development

```bash
cd /path/to/argotech-ai
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# torch: install the CPU wheel first if you do not want the CUDA stack
pip install --index-url https://download.pytorch.org/whl/cpu torch

uvicorn src.services.inferenceService.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Run it from the **repository root**. `ModelManager` opens `farmerxential_model.pkl` by bare relative
path, so the process working directory must be the directory holding the `.pkl` files.

Smoke test:

```bash
curl -s localhost:8000/health
# {"status":"ok"}

curl -s -X POST localhost:8000/predict/coldstart \
  -H 'content-type: application/json' \
  -d '{"latitude":10.52,"longitude":7.44}' | python -m json.tool
```

`/predict/coldstart` needs no database. `/predict/farmer` needs a reachable Postgres with
`farmer_profiles`, `farmers_ml_profiles`, and `farmers_crops` — set `DATABASE_URL` in a local `.env`
(gitignored) and point it at the backend's database.

`Procfile` (`web: uvicorn ... --port $PORT`) is a leftover from PaaS hosting. The Docker image hard-codes
port 8000 instead.

---

## Docker build and deploy

The image is built with Cloud Build and pushed to Artifact Registry:

```bash
gcloud builds submit \
  --tag europe-west2-docker.pkg.dev/farmerxential-backend/agri/ml:<tag> \
  --region=europe-west2 .
```

Then on the VM, set the tag and roll the service:

```bash
# /opt/agri/.env
ML_IMAGE=europe-west2-docker.pkg.dev/farmerxential-backend/agri/ml:<tag>
```

```bash
sudo docker compose -f docker-compose.prod.yml pull ml
sudo docker compose -f docker-compose.prod.yml up -d ml
```

Currently deployed tag: `ml:2026-08-07c`.

### Image layout

| Layer | Why |
| --- | --- |
| `python:3.11-slim` | Base |
| `apt-get install libgomp1` | XGBoost links against libgomp at runtime; nothing else needs a system package |
| `pip install --index-url https://download.pytorch.org/whl/cpu torch` | Installed **before** `requirements.txt` so the bare `torch` line there resolves to the already-satisfied CPU build instead of pulling ~2.5 GB of `nvidia-*` CUDA wheels onto a GPU-less VM |
| `COPY requirements.txt` then `pip install -r` | Dependency layer only rebuilds when requirements change |
| `COPY farmerxential_model.pkl farmerxential_powerful_model.pkl ./` | Into `/app`, the `WORKDIR`, because `ModelManager` loads them by bare relative path |
| `COPY src src` | Application code |
| `useradd --system ml` + `USER ml` | Non-root runtime |
| `CMD uvicorn ... --port 8000` | Fixed port; compose talks to it on a known internal port |

`.dockerignore` and `.gcloudignore` exclude `.venv/` (≈1 GB), `mlruns/`, `mlflow.db`, `*.csv`,
`farmerxential_features.pkl`, and `farmerxential_shap_explainer.pkl`. `.gcloudignore` is written
explicitly because gcloud otherwise derives one from `.gitignore`, which lists `venv/` but not
`.venv/` — a 1 GB upload on every build.

### Compose service

```yaml
ml:
  image: ${ML_IMAGE}
  container_name: agri-ml
  restart: always
  environment:
    DATABASE_URL: postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/${DB_NAME}
    USE_LOCAL_MODEL: "true"
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

## Models

Two artifacts are baked into the image. Both are `joblib` dumps of a
`HybridSpatiotemporalEnsemble` instance (`app/models/powerful_model.py`), which wraps:

- `XGBClassifier` (300 trees, lr 0.03, depth 6)
- `HistGradientBoostingClassifier` (250 iters, lr 0.03, depth 7)
- `RandomForestClassifier` (200 trees, depth 10)
- `ExtraTreesClassifier` (200 trees, depth 10)
- stacked under a `LogisticRegression` meta-learner (`cv=5`), then wrapped in
  `CalibratedClassifierCV(method="sigmoid", cv="prefit")`

| File | In image | Selected by |
| --- | --- | --- |
| `farmerxential_model.pkl` | yes | `model_name == "farmerXential_model"` |
| `farmerxential_powerful_model.pkl` | yes | `model_name == "farmerXential_powerful_model"` (the request-body default) |
| `farmerxential_shap_explainer.pkl` | **no** | Training artifact only; nothing at inference time loads it |
| `farmerxential_features.pkl` | **no** | Training artifact only |
| `mlruns/`, `mlflow.db` | **no** | Local MLflow experiment history |

At the time of writing both `*_model.pkl` files are byte-identical (both training scripts call
`save_model` twice with the two names), so the choice of `model_name` is currently cosmetic.

### Selection and caching

`ModelManager.get_model(model_name, model_alias)` caches by `"{name}:{alias}"` in a plain dict for the
process lifetime. If `USE_LOCAL_MODEL` is true **and** the name is one of the two known local names,
it `joblib.load`s the file; anything else goes to MLflow (`models:/{name}/{alias}`) and raises a
`ValueError` if the registry does not have it.

After loading it runs `patch_model`, which walks pipelines / voting / stacking estimators and restores
a `multi_class` attribute on any `LogisticRegression` that lacks it — a compatibility shim for pickles
written by a different scikit-learn.

### Feature alignment

`FeatureStore` builds a **27**-column frame (`FEATURES` in `feature_store/features.py`: 24 survey /
climate columns plus `ndvi`, `ndwi`, `evi`). The currently shipped `.pkl` was trained on 24 columns.
`ModelManager._align_features` reindexes the frame to the estimator's own `feature_names_in_`, adding
missing columns as `0` and dropping extras, so:

- today the model silently ignores `ndvi` / `ndwi` / `evi`,
- after a retrain with `train_real_production_model.py` (which does emit them) they are used
  automatically,
- neither direction throws `X has N features but model expects M`.

That decoupling is deliberate: it lets the feature set and the model artifact move independently
instead of requiring a lockstep redeploy.

### The version-pinning constraint

A `joblib` pickle stores references to the *exact internal module layout* of the scikit-learn that
wrote it. `requirements.txt` is therefore pinned to the training-time versions:

| Package | Pin |
| --- | --- |
| `scikit-learn` | `1.6.1` |
| `numpy` | `2.0.2` |
| `joblib` | `1.5.3` |
| `xgboost` | `2.1.4` |

**Retrain and bump together, never one alone.** If you regenerate the `.pkl` files under a newer
scikit-learn, update these pins in the same commit.

---

## Training

The training code is in two places and is the less finished half of the repo.

| Path | State | What it does |
| --- | --- | --- |
| `train_real_production_model.py` | Current | Builds a 600-sample synthetic dataset over five Sub-Saharan farming clusters, enriched with **real** Open-Meteo ERA5 archive queries per coordinate. 5-fold stratified CV, then fits on 100% and writes both `.pkl` files plus a TreeSHAP explainer. Emits the 27-column feature set including NDVI/NDWI/EVI |
| `train_powerful_model.py` | Superseded | Fully synthetic 3000-sample dataset, 24 columns, no spectral indices, no external calls |
| `src/services/training-worker/train.py` | Stub | Trains a `RandomForestRegressor` on a 5-row dummy dataset and registers it in MLflow as `farmerXential_model` with alias `prod`. Not the model that serves traffic |
| `src/services/training-worker/data_loader.py` | Stub | Hardcoded `postgresql://user:pass@localhost/db`, `SELECT * FROM training_dataset` |
| `src/services/training-worker/evaluate.py` | Empty file | — |

```bash
# regenerate the served artifacts
python train_real_production_model.py
```

Both root-level training scripts `import shap`, which is **not** in `requirements.txt`. Install it
separately (`pip install shap`) before training. It is correctly absent from the runtime image —
nothing at inference time loads the explainer.

Labels are rule-derived from the climate/index signals, not observed outbreaks. Treat reported CV
F1 as a measure of the model's ability to reproduce those rules, not of real-world disease
prediction accuracy.

---

## Troubleshooting

### `ModuleNotFoundError: No module named '_loss'` on startup

**Incident, production.** `requirements.txt` was unpinned, so the image resolved scikit-learn 1.9.0
while the `.pkl` files had been trained under 1.6.1. `joblib.load` fails because the pickle
references internal scikit-learn modules that were moved or renamed between versions.

Fix: pin to the training versions (`scikit-learn==1.6.1`, `numpy==2.0.2`, `joblib==1.5.3`,
`xgboost==2.1.4`). If you retrain, bump the pins and the models together in one change.

### `ModuleNotFoundError: No module named 'torch'` — container crash loop

**Incident, production.** `app/models/cross_attention_fusion.py` imports `torch` at module scope and
`PredictionsService` imports that module, but `torch` was never listed in `requirements.txt` — it had
only ever been present in the developer's venv. The container restarted forever.

Fix: `torch` is now in `requirements.txt`, and the Dockerfile installs the **CPU-only** wheel from
`https://download.pytorch.org/whl/cpu` *before* the requirements step. Installing plain `torch` from
PyPI drags in the full `nvidia-*` CUDA dependency set — roughly 2.5 GB — onto a VM with no GPU.
Keep those two lines in that order.

### Image builds but inference 400s with `Local model file '...' not found`

The `.pkl` files must sit in the process working directory. The Dockerfile copies them to `/app`
(the `WORKDIR`) for exactly this reason. Locally, run `uvicorn` from the repository root.

### `404 Farmer '<id>' not found in database`

`/predict/farmer` requires a row in `farmer_profiles` for that `user_id`. The `farmers_ml_profiles`
join is a `LEFT JOIN`, so a farmer with no ML profile still works — the pipeline switches to
`GEOSPATIAL_COLDSTART_REMOTE_SENSING` mode and derives yield/shock/asset proxies from the satellite
and weather signals instead.

### `spatiotemporal_indices.source` is always `"modelled"`

Sentinel credentials are missing or the fetch failed. Check in order:

1. `SENTINEL_CLIENT_ID` / `SENTINEL_CLIENT_SECRET` are set on the container.
2. Container logs for `[SentinelClient] fetch_indices failed: ...`.
3. Whether any scene under 40% cloud cover exists for that AOI in the last 30 days — a fully
   cloud-masked window yields `sampleCount == 0` and falls back silently.

### Build context upload is enormous / slow

`.venv/` is roughly 1 GB. Confirm `.gcloudignore` exists and lists `.venv/`. Without it, gcloud
generates one from `.gitignore`, which only excludes `venv/`.

### Healthcheck fails but the app looks up

The compose healthcheck runs `python -c "import urllib.request; ..."` inside the container. `curl`
and `wget` are not installed in `python:3.11-slim` — a healthcheck rewritten to use them will always
fail regardless of service state.

---

## Known issues

| # | Issue | Impact | Suggested fix |
| --- | --- | --- | --- |
| 1 | Images are tagged **by date** (`ml:2026-08-07c`), not by commit SHA | A tag does not identify the code that produced it; the running image cannot be traced back to a revision | The repo was containerised while carrying uncommitted work, which is why SHA tags were not usable. Commit the working tree, then switch to `ml:$(git rev-parse --short HEAD)` |
| 2 | `requests.get(url, params={"timeout": 4})` in `_fetch_microclimate_weather_data` | This appends `?timeout=4` to the Open-Meteo URL and sets **no** request timeout. A hung Open-Meteo connection blocks a threadpool worker indefinitely | `requests.get(url, timeout=4)` |
| 3 | `torch` is a required dependency but the PyTorch layer is never used | The `PyTorchCrossAttentionFusion` class is defined but `PredictionsService` instantiates the NumPy `CrossAttentionFusionLayer`. The import alone costs image size and startup time | Either move the `torch` import behind the class that needs it, or drop `PyTorchCrossAttentionFusion` |
| 4 | `CrossAttentionFusionLayer` weights are fixed random draws (`np.random.seed(42)`), never trained | `fusion_score` and `peak_incubation_hour` are deterministic but carry no learned signal | Either train the layer or label the output as diagnostic |
| 5 | `GET /predict/crop-health` returns `random` values | The backend calls it via `PredictionQueueConsumer` for `MLTaskType.CROP_HEALTH`; consumers may treat mock output as real | Implement or remove |
| 6 | `POST /predict` always returns 400 | Dead route that still advertises a full 24-field schema in `/docs` | Remove the route and `PredictionRequest`, or implement it |
| 7 | `requests` is imported but not in `requirements.txt` | Works only because `mlflow` pulls it in transitively; an mlflow change could break the image | Add `requests` explicitly |
| 8 | `shap` is imported by both training scripts but not in `requirements.txt` | `python train_real_production_model.py` fails on a clean venv | Add a `requirements-train.txt` |
| 9 | `sentinel_client.py` comments target Python 3.9; the image is 3.11 | Only a stale comment, but the `from __future__ import annotations` it justifies is now unnecessary | Update the comment |
| 10 | `@app.on_event("startup")` / `("shutdown")` | Deprecated in current FastAPI; will warn and eventually break | Move to the `lifespan` context manager |
| 11 | `zone` is derived from a 3-entry hardcoded `state_map` (`Kaduna`/`Kano`/`Lagos`), everything else maps to `0` | Every other state collapses into one bucket the model was not trained to distinguish | Move the mapping to config or the database |
| 12 | The service holds direct read access to the backend's Postgres schema | Two services coupled to one table layout; a backend migration can silently break inference | Have the backend pass the farmer payload in the request body |
| 13 | No auth on any route | Safe only because the container publishes no ports. A future `ports:` entry would expose it | Keep it network-only; add a shared secret if it ever needs exposing |
| 14 | Both `.pkl` artifacts are byte-identical | `model_name` selection has no effect today | Either differentiate them or collapse to one |

---

## Repository layout

```
.
├── Dockerfile                       # production image
├── Procfile                         # legacy PaaS entrypoint, unused in Docker
├── requirements.txt                 # runtime deps, pinned to model training versions
├── .dockerignore / .gcloudignore    # keep .venv and training artifacts out of the build context
├── farmerxential_model.pkl          # served model (in image)
├── farmerxential_powerful_model.pkl # served model (in image)
├── farmerxential_shap_explainer.pkl # training artifact (not in image)
├── train_real_production_model.py   # current training pipeline
├── train_powerful_model.py          # superseded training pipeline
└── src/services/
    ├── inferenceService/app/
    │   ├── main.py                  # FastAPI app, routers, OTel setup, /health
    │   ├── api/predict.py           # /predict, /predict/farmer, /predict/coldstart
    │   ├── api/crop_health.py       # GET /predict/crop-health (mock)
    │   ├── core/config.py           # pydantic-settings
    │   ├── core/container.py        # module-level FeatureStore + ModelManager singletons
    │   ├── core/database.py         # SQLAlchemy engine + get_db dependency
    │   ├── core/model_manager.py    # model load, cache, feature alignment, predict
    │   ├── dependencies.py          # FastAPI Depends wrappers over the singletons
    │   ├── feature_store/           # FEATURES schema + DataFrame builder
    │   ├── models/                  # HybridSpatiotemporalEnsemble, cross-attention fusion
    │   ├── schemas/                 # request / response pydantic models
    │   └── service/                 # PredictionsService, SentinelClient
    └── training-worker/             # MLflow registration stub, not production
```
