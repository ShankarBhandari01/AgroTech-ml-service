from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.services.inferenceService.app.api.crop_health import router as crop_health_router
from src.services.inferenceService.app.api.predict import router as predict_router

# Set up OpenTelemetry Tracing
resource = Resource(attributes={"service.name": "fastapi-ml"})
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces"))
provider.add_span_processor(processor)
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
