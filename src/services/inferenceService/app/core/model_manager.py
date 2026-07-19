import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
import os

from src.services.inferenceService.app.core.config import settings


class ModelManager:
    def __init__(self):
        self.models = {}
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    def get_model(self, model_name: str, model_alias: str = "prod"):
        key = f"{model_name}:{model_alias}"
        if key not in self.models:
            print(f"Loading model: {model_name} (alias: {model_alias})...")
            # Local fallback for the default farmerxential model
            if settings.USE_LOCAL_MODEL and model_name == "farmerXential_model":
                model_path = "farmerxential_model.pkl"
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    
                    # Monkey-patch missing attributes for older scikit-learn compatibility
                    def patch_model(est):
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.ensemble import VotingClassifier
                        
                        if isinstance(est, LogisticRegression):
                            if not hasattr(est, "multi_class"):
                                est.multi_class = "auto"
                        elif isinstance(est, VotingClassifier):
                            if hasattr(est, "estimators_"):
                                for item in est.estimators_:
                                    if isinstance(item, tuple) and len(item) == 2:
                                        patch_model(item[1])
                                    else:
                                        patch_model(item)
                        
                        # Also check if it's a pipeline
                        if hasattr(est, "steps"):
                            for _, step in est.steps:
                                patch_model(step)
                                
                    patch_model(model)
                    self.models[key] = model
                else:
                    raise FileNotFoundError(f"Local model file '{model_path}' not found.")
            else:
                try:
                    model_uri = f"models:/{model_name}/{model_alias}"
                    self.models[key] = mlflow.pyfunc.load_model(model_uri)
                except Exception as e:
                    from mlflow.exceptions import RestException
                    if isinstance(e, RestException) and "RESOURCE_DOES_NOT_EXIST" in str(e):
                        raise ValueError(f"Model '{model_name}' (alias: '{model_alias}') not found in MLflow registry.")
                    raise ValueError(f"Failed to load model from MLflow: {str(e)}")
        return self.models[key]

    def predict(self, features, model_name: str = "farmerXential_model", model_alias: str = "prod"):
        model = self.get_model(model_name, model_alias)
        return model.predict(features)

    def predict_proba(self, df: pd.DataFrame, model_name: str = "farmerXential_model", model_alias: str = "prod"):
        model = self.get_model(model_name, model_alias)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(df)
        elif hasattr(model, "_model_impl") and hasattr(model._model_impl, "predict_proba"):
            return model._model_impl.predict_proba(df)
        else:
            # Fallback if the pyfunc model doesn't expose predict_proba directly
            # For some scikit-learn models wrapped in pyfunc, we can extract the underlying model
            try:
                underlying = model.unwrap_python_model()
                if hasattr(underlying, "predict_proba"):
                    return underlying.predict_proba(df)
            except:
                pass
            raise AttributeError(f"Model '{model_name}' does not support predict_proba.")