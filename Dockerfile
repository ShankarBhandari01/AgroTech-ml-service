# Inference service image, built for the same single GCE VM the Kotlin backend runs on
# (docker-compose.prod.yml in agri-saas-kotlin-backend, service `ml`).
FROM python:3.11-slim AS runner
WORKDIR /app

# xgboost links against libgomp at runtime; every other dependency ships a manylinux wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer first, so it only rebuilds when the pins actually change. The torch install that
# used to sit here is gone: the PyTorch fusion layer was never instantiated.
COPY pyproject.toml .
RUN mkdir -p src/argotech && touch src/argotech/__init__.py && pip install --no-cache-dir .

# models/registry.py loads these by bare relative path, so they must land in the working directory.
# ponytail: baked into the image, which is why a model update needs a rebuild. Pull from GCS at
# startup once models ship more often than code.
COPY farmerxential_model.pkl farmerxential_powerful_model.pkl ./
COPY src src
RUN pip install --no-cache-dir --no-deps .

RUN useradd --system --create-home ml
USER ml

EXPOSE 8000

CMD ["uvicorn", "argotech.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
