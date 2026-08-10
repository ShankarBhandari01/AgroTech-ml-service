import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
import os

from argotech.config import settings


class ModelManager:
    def __init__(self):
        self.models = {}
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    def get_model(self, model_name: str, model_alias: str = "prod"):
        key = f"{model_name}:{model_alias}"
        if key not in self.models:
            print(f"Loading model: {model_name} (alias: {model_alias})...")
            # Local fallback for the default farmerxential models
            if settings.USE_LOCAL_MODEL and model_name in ["farmerXential_model", "farmerXential_powerful_model"]:
                model_path = "farmerxential_powerful_model.pkl" if model_name == "farmerXential_powerful_model" else "farmerxential_model.pkl"
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    
                    # Patch compatibility attributes if needed
                    def patch_model(est):
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.ensemble import VotingClassifier, StackingClassifier
                        
                        if isinstance(est, LogisticRegression):
                            if not hasattr(est, "multi_class"):
                                est.multi_class = "auto"
                        elif isinstance(est, (VotingClassifier, StackingClassifier)):
                            if hasattr(est, "estimators_"):
                                for item in est.estimators_:
                                    if isinstance(item, tuple) and len(item) == 2:
                                        patch_model(item[1])
                                    else:
                                        patch_model(item)
                        
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

    @staticmethod
    def _align_features(model, features):
        """
        Reindex the input to exactly the columns the loaded model was trained on. This keeps
        inference working across a feature-set change without a lockstep redeploy: the current
        24-feature model simply ignores the new ndvi/ndwi/evi columns, and once the model is
        retrained with them (FEATURES now lists them) they're used automatically — no 'X has N
        features but model expects M' crash either way.
        """
        if not isinstance(features, pd.DataFrame):
            return features
        inner = getattr(model, "stacking_model", None)
        expected = None
        if inner is not None and hasattr(inner, "feature_names_in_"):
            expected = list(inner.feature_names_in_)
        elif hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
        if not expected:
            return features
        for col in expected:
            if col not in features.columns:
                features[col] = 0
        return features[expected]

    def predict(self, features, model_name: str = "farmerXential_model", model_alias: str = "prod"):
        model = self.get_model(model_name, model_alias)
        return model.predict(self._align_features(model, features))

    def predict_proba(self, df: pd.DataFrame, model_name: str = "farmerXential_model", model_alias: str = "prod"):
        model = self.get_model(model_name, model_alias)
        df = self._align_features(model, df)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(df)
        elif hasattr(model, "_model_impl") and hasattr(model._model_impl, "predict_proba"):
            return model._model_impl.predict_proba(df)
        else:
            try:
                underlying = model.unwrap_python_model()
                if hasattr(underlying, "predict_proba"):
                    return underlying.predict_proba(df)
            except:
                pass
            raise AttributeError(f"Model '{model_name}' does not support predict_proba.")