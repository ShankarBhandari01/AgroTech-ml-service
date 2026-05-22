from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# ==============================
# LOAD MODEL AND FEATURES
# ==============================
model = joblib.load("agroreach_model.pkl")
features = joblib.load("agroreach_features.pkl")
explainer = joblib.load("agroreach_shap_explainer.pkl")

app = FastAPI(title="AgroReach API", version="1.0")

# ==============================
# INPUT SCHEMA
# What data the API expects
# ==============================
class FarmerInput(BaseModel):
    yield_value: float
    has_extension_access: int
    household_max_education: int
    shock_level: int
    received_assistance: int
    used_fertilizer: int
    land_size: float
    household_size: int
    zone: int
    transport_cost: float
    dependency_ratio: float
    asset_score: float
    postharvest_activity_score: float
    crop_loss_risk_score: float
    crop_diversity_score: float
    digital_access_score: float
    has_veterinary_access: int
    market_access_score: float
    is_rural: int
    rainfall_anomaly: float
    drought_risk: int
    cultivates_crops: int
    received_credit: int
    head_gender: int

# ==============================
# HEALTH CHECK ENDPOINT
# ==============================
@app.get("/")
def home():
    return {
        "message": "AgroReach API is running!",
        "version": "1.0",
        "status": "healthy"
    }

# ==============================
# PREDICT ENDPOINT
# ==============================
@app.post("/predict")
def predict(farmer: FarmerInput):

    # Build dataframe from input
    input_data = pd.DataFrame([{
        "yield": farmer.yield_value,
        "has_extension_access": farmer.has_extension_access,
        "household_max_education": farmer.household_max_education,
        "shock_level": farmer.shock_level,
        "received_assistance": farmer.received_assistance,
        "used_fertilizer": farmer.used_fertilizer,
        "land_size": farmer.land_size,
        "household_size": farmer.household_size,
        "zone": farmer.zone,
        "transport_cost": farmer.transport_cost,
        "dependency_ratio": farmer.dependency_ratio,
        "asset_score": farmer.asset_score,
        "postharvest_activity_score": farmer.postharvest_activity_score,
        "crop_loss_risk_score": farmer.crop_loss_risk_score,
        "crop_diversity_score": farmer.crop_diversity_score,
        "digital_access_score": farmer.digital_access_score,
        "has_veterinary_access": farmer.has_veterinary_access,
        "market_access_score": farmer.market_access_score,
        "is_rural": farmer.is_rural,
        "rainfall_anomaly": farmer.rainfall_anomaly,
        "drought_risk": farmer.drought_risk,
        "cultivates_crops": farmer.cultivates_crops,
        "received_credit": farmer.received_credit,
        "head_gender": farmer.head_gender
    }])

    input_data = input_data[features]

    # Get prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    risk_score = round(float(probabilities[2]) * 100, 1)

    # Map prediction to label
    priority_map = {
        0: "Low Priority",
        1: "Medium Priority",
        2: "High Priority"
    }

    priority_label = priority_map[int(prediction)]

    # SHAP explanation
    shap_values = explainer.shap_values(input_data)
    shap_class2 = shap_values[:, :, 2][0]

    shap_explanation = pd.DataFrame({
        "feature": input_data.columns.tolist(),
        "impact": shap_class2
    }).sort_values("impact", ascending=False)

    top_reasons = shap_explanation[
        shap_explanation["impact"] > 0
    ].head(3)["feature"].tolist()

    return {
        "prediction": int(prediction),
        "priority_label": priority_label,
        "risk_score_percent": risk_score,
        "probabilities": {
            "low": round(float(probabilities[0]) * 100, 1),
            "medium": round(float(probabilities[1]) * 100, 1),
            "high": round(float(probabilities[2]) * 100, 1)
        },
        "top_risk_factors": top_reasons
    }

# ==============================
# BATCH PREDICT ENDPOINT
# ==============================
@app.get("/farmers")
def get_farmers():
    df = pd.read_csv("agroreach_farmer_priority_output.csv")
    df["risk_score"] = (df["risk_score"] * 100).round(1)
    df["priority_label"] = df["predicted_intervention_level"].map({
        0: "Low Priority",
        1: "Medium Priority",
        2: "High Priority"
    })
    return df.to_dict(orient="records")