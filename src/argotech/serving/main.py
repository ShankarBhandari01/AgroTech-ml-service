import logging
import os
import time

from fastapi import FastAPI, Request
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from argotech.config import settings
from argotech.data.sentinel import sentinel_client
from argotech.serving.api.crop_health import router as crop_health_router
from argotech.serving.api.outcomes import router as outcomes_router
from argotech.serving.api.precompute import router as precompute_router
from argotech.serving.api.predict import router as predict_router

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
access_log = logging.getLogger("argotech.access")

# Set up OpenTelemetry Tracing. The exporter is opt-in via OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:
# no collector runs alongside the deployed compose stack, and an unconditional exporter pointed at
# a dead localhost:4318 makes BatchSpanProcessor retry and log an export failure every few seconds
# forever. Tracing itself stays wired up either way; only the shipping of spans is gated.
resource = Resource(attributes={"service.name": "fastapi-ml"})
provider = TracerProvider(resource=resource)
otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
if otlp_endpoint:
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
trace.set_tracer_provider(provider)



app = FastAPI(title="AgroTech AI", version="0.2.0")
FastAPIInstrumentor.instrument_app(app)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One line per request, with status and latency. Bodies are logged by the handlers that
    already have them parsed — reading the stream here would mean replaying it for the route."""
    started = time.perf_counter()
    query = f"?{request.url.query}" if request.url.query else ""
    access_log.info("--> %s %s%s", request.method, request.url.path, query)
    try:
        response = await call_next(request)
    except Exception:
        access_log.exception("<-- %s %s ERROR %.0fms", request.method, request.url.path,
                             (time.perf_counter() - started) * 1000)
        raise
    access_log.info("<-- %s %s %s %.0fms", request.method, request.url.path,
                    response.status_code, (time.perf_counter() - started) * 1000)
    return response


app.include_router(predict_router)
app.include_router(crop_health_router)
app.include_router(outcomes_router)
app.include_router(precompute_router)

# The artifact itself is loaded lazily, on the first request that takes the model path; registry.py
# logs which file and version it got. This says up front whether that will ever happen.
logging.getLogger("argotech.serving").info(
    "AgroTech AI %s: vegetation hazard from %r, agronomic artifact %s, sentinel %s",
    app.version, settings.VEGETATION_HAZARD_SOURCE, settings.AGRONOMIC_MODEL_PATH,
    "enabled" if sentinel_client.enabled else "disabled")


@app.get("/health")
def health():
    return {"status": "ok"}
