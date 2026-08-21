# Inference service image, built for the same single GCE VM the Kotlin backend runs on
# (docker-compose.prod.yml in agri-saas-kotlin-backend, service `ml`).
FROM python:3.11-slim AS runner
WORKDIR /app

# scikit-learn's HistGradientBoosting links against libgomp at runtime; every other
# dependency ships a manylinux wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer first, so it only rebuilds when the pins actually change. The torch install that
# used to sit here is gone: the PyTorch fusion layer was never instantiated.
COPY pyproject.toml .
RUN mkdir -p src/argotech && touch src/argotech/__init__.py && pip install --no-cache-dir .

# The trained model, loaded by relative path from AGRONOMIC_MODEL_PATH.
# ponytail: baked into the image, so a model update needs a rebuild. Pull from GCS at startup once
# models ship more often than code.
COPY artifacts/agronomic_risk.joblib artifacts/
COPY src src
RUN pip install --no-cache-dir --no-deps .

# `meteo.CACHE_DIR` and `sentinel`'s caches are relative paths under the working directory, and the
# service runs unprivileged — so /app/.cache has to exist and be writable before the drop, or every
# upstream response is re-fetched and the failure surfaces only as a warning:
#   [meteo] archive 11.900,8.500 failed: [Errno 13] Permission denied: '.cache'
# That is silent in the response: the climatology falls back to the observed 30-day total, so
# `rain_anomaly_30` reads exactly 0.0 — "perfectly normal rainfall" — for a field in deficit.
RUN useradd --system --create-home ml \
    && mkdir -p /app/.cache \
    && chown -R ml:ml /app/.cache
USER ml

EXPOSE 8000

CMD ["uvicorn", "argotech.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
