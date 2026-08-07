# Inference service image, built for the same single GCE VM the Kotlin backend runs on
# (docker-compose.prod.yml in agri-saas-kotlin-backend, service `ml`).
FROM python:3.11-slim AS runner
WORKDIR /app

# xgboost links against libgomp at runtime; every other dependency ships a manylinux wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch, from PyTorch's index rather than PyPI: the default linux wheel depends on the
# whole nvidia-* CUDA stack (~2.5 GB) for a VM that has no GPU. Installed before requirements.txt
# so the plain `torch` line there resolves to this build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Requirements second, so the dependency layer only rebuilds when they actually change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# model_manager.py loads these by bare relative path ("farmerxential_model.pkl"), so they must
# land in the working directory, not alongside the source. Both names are copied because
# MODEL_NAME picks between them at runtime.
COPY farmerxential_model.pkl farmerxential_powerful_model.pkl ./
COPY src src

RUN useradd --system --create-home ml
USER ml

EXPOSE 8000

# Procfile's uvicorn line, with the port fixed rather than read from $PORT — compose talks to
# this over the internal network on a known port, there is no PaaS assigning one.
CMD ["uvicorn", "src.services.inferenceService.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
