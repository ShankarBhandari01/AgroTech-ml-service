from fastapi import FastAPI
from src.services.inferenceService.app.api.predict import router
from .dependencies import model_manager

app = FastAPI()
app.include_router(router)


@app.on_event("startup")
def startup_event():
    print("Server is starting...")
    model_manager.load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("shutdown")
def shutdown_event():
    print("Server is shutting down...")
