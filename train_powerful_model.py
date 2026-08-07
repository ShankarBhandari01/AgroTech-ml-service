import numpy as np
import pandas as pd
import joblib
import shap
from src.services.inferenceService.app.feature_store.features import FEATURES
from src.services.inferenceService.app.models.powerful_model import HybridSpatiotemporalEnsemble

def generate_synthetic_agricultural_dataset(n_samples: int = 2500, random_state: int = 42):
    """
    Generates a realistic multi-spectral and microclimate agricultural dataset.
    """
    np.random.seed(random_state)
    
    feature_list = list(FEATURES.keys())

    # Map features with realistic distributions
    yield_val = np.random.normal(loc=1.8, scale=0.6, size=n_samples).clip(0.3, 5.0)
    has_ext = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    edu = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.3, 0.4, 0.2, 0.1])
    shock = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    assistance = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    fertilizer = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    land_size = np.random.exponential(scale=2.5, size=n_samples).clip(0.5, 20.0)
    household_size = np.random.poisson(lam=5, size=n_samples).clip(1, 15)
    zone = np.random.choice([0, 1, 2, 3], size=n_samples)
    transport_cost = np.random.uniform(500, 15000, size=n_samples)
    dependency_ratio = np.random.uniform(0.1, 2.5, size=n_samples)
    asset_score = np.random.normal(loc=50, scale=15, size=n_samples).clip(10, 100)
    postharvest = np.random.uniform(0, 10, size=n_samples)
    crop_loss_risk = np.random.uniform(0, 1, size=n_samples)
    crop_diversity = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    digital_access = np.random.uniform(0, 100, size=n_samples)
    veterinary = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    market_access = np.random.uniform(0, 100, size=n_samples)
    is_rural = np.random.choice([0, 1], size=n_samples, p=[0.1, 0.9])
    rainfall_anomaly = np.random.normal(loc=0.0, scale=15.0, size=n_samples)
    drought_risk = np.where(rainfall_anomaly < -10.0, 1, 0)
    cultivates_crops = np.ones(n_samples, dtype=int)
    credit = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    head_gender = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

    df = pd.DataFrame({
        "yield": yield_val,
        "has_extension_access": has_ext,
        "household_max_education": edu,
        "shock_level": shock,
        "received_assistance": assistance,
        "used_fertilizer": fertilizer,
        "land_size": land_size,
        "household_size": household_size,
        "zone": zone,
        "transport_cost": transport_cost,
        "dependency_ratio": dependency_ratio,
        "asset_score": asset_score,
        "postharvest_activity_score": postharvest,
        "crop_loss_risk_score": crop_loss_risk,
        "crop_diversity_score": crop_diversity,
        "digital_access_score": digital_access,
        "has_veterinary_access": veterinary,
        "market_access_score": market_access,
        "is_rural": is_rural,
        "rainfall_anomaly": rainfall_anomaly,
        "drought_risk": drought_risk,
        "cultivates_crops": cultivates_crops,
        "received_credit": credit,
        "head_gender": head_gender
    })

    # Ground Truth Target Logic: 0 = Low Risk, 1 = Medium Risk, 2 = High Risk (Outbreak / Critical)
    risk_signal = (
        (df["yield"] < 1.2).astype(int) * 2.5 +
        (df["has_extension_access"] == 0).astype(int) * 1.5 +
        df["shock_level"] * 1.2 +
        (df["used_fertilizer"] == 0).astype(int) * 1.5 +
        df["drought_risk"] * 3.0 +
        (df["received_credit"] == 0).astype(int) * 1.0 +
        (df["asset_score"] < 35).astype(int) * 1.5 +
        np.random.normal(0, 1.0, size=n_samples)
    )

    y = np.zeros(n_samples, dtype=int)
    y[risk_signal >= 4.0] = 1
    y[risk_signal >= 7.5] = 2

    return df[feature_list], y

def train_and_save_powerful_model():
    print("--- Starting Hybrid Spatiotemporal Model Training ---")
    X, y = generate_synthetic_agricultural_dataset(n_samples=3000, random_state=42)

    model = HybridSpatiotemporalEnsemble(random_state=42)
    model.fit(X, y)

    # Save trained ensemble model
    model_path = "farmerxential_model.pkl"
    powerful_path = "farmerxential_powerful_model.pkl"
    model.save_model(model_path)
    model.save_model(powerful_path)

    # Fit TreeSHAP Explainer on XGBoost Base Estimator
    print("Fitting TreeSHAP Explainer on XGBoost Base Estimator...")
    explainer = shap.TreeExplainer(model.xgb)
    shap_path = "farmerxential_shap_explainer.pkl"
    joblib.dump(explainer, shap_path)
    print(f"SHAP Explainer saved to {shap_path}")

    print("--- Powerful Model Pipeline Upgrade Successfully Completed! ---")

if __name__ == "__main__":
    train_and_save_powerful_model()
