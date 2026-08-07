import os

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.services.inferenceService.app.api.crop_health import router as crop_health_router
from src.services.inferenceService.app.api.predict import router as predict_router

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



app = FastAPI()
FastAPIInstrumentor.instrument_app(app)

app.include_router(predict_router)
app.include_router(crop_health_router)


@app.on_event("startup")
def startup_event():
    print("Server is starting...")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("shutdown")
def shutdown_event():
    print("Server is shutting down...")
