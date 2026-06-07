import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd

from src.services.inferenceService.app.core.config import settings


class ModelManager:
    def __init__(self):
        self.model = None
        self.features = None
        self.explainer = None

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    def load_model(self):

        model_uri = f"models:/{settings.MODEL_NAME}/{settings.MODEL_ALIAS}"

        if settings.USE_LOCAL_MODEL:
            self.model = joblib.load("farmerxential_model.pkl")
            self.features = joblib.load("farmerxential_features.pkl")
            self.explainer = None  # safer default
        else:
            self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, features):
        if self.model is None:
            raise Exception("Model not loaded. Call load_model() first.")

        return self.model.predict(features)

    def predict_proba(self, df: pd.DataFrame):
        return self.model.predict_proba(df)