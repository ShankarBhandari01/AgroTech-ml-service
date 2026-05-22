from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# ==============================
# LOAD MODEL AND FEATURES
# ==============================
model = joblib.load("agroreach_model.pkl")
features = joblib.load("agroreach_features.pkl")
explainer = joblib.load("agroreach_shap_explainer.pkl")

app = FastAPI(title="AgroReach API", version="1.0")

# ==============================
# CORS — Allow frontend to access API
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# API KEY AUTHENTICATION
# ==============================
API_KEY = os.environ.get("AGROREACH_API_KEY", "agroreach-dev-key-2024")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Include X-API-Key header."
        )
    return api_key

# ==============================
# INPUT SCHEMA
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
# HEALTH CHECK — No auth needed
# ==============================
@app.get("/")
def home():
    return {
        "message": "AgroReach API is running!",
        "version": "1.0",
        "status": "healthy"
    }

# ==============================
# STATS ENDPOINT
# ==============================
@app.get("/stats", dependencies=[Depends(verify_api_key)])
def get_stats():
    df = pd.read_csv("agroreach_farmer_priority_output.csv")
    df["risk_score_pct"] = df["risk_score"] * 100

    zone_mapping = {
        0: "North Central", 1: "North East", 2: "North West",
        3: "South East", 4: "South South", 5: "South West"
    }

    return {
        "total_farmers": len(df),
        "high_priority_count": int((df["predicted_intervention_level"] == 2).sum()),
        "medium_priority_count": int((df["predicted_intervention_level"] == 1).sum()),
        "low_priority_count": int((df["predicted_intervention_level"] == 0).sum()),
        "active_alerts": int((df["risk_score"] > 0.7).sum()),
        "avg_risk_score": round(float(df["risk_score_pct"].mean()), 1),
        "zones": {
            zone_mapping.get(int(zone), str(zone)): int(count)
            for zone, count in df.groupby("zone")["predicted_intervention_level"]
            .apply(lambda x: (x == 2).sum()).items()
        }
    }

# ==============================
# GET ALL FARMERS
# ==============================
@app.get("/farmers", dependencies=[Depends(verify_api_key)])
def get_farmers():
    df = pd.read_csv("agroreach_farmer_priority_output.csv")
    df["risk_score"] = (df["risk_score"] * 100).round(1)
    df["priority_label"] = df["predicted_intervention_level"].map({
        0: "Low Priority",
        1: "Medium Priority",
        2: "High Priority"
    })
    zone_mapping = {
        0: "North Central", 1: "North East", 2: "North West",
        3: "South East", 4: "South South", 5: "South West"
    }
    df["zone_name"] = df["zone"].map(zone_mapping)
    return df.to_dict(orient="records")

# ==============================
# GET HIGH PRIORITY ALERTS
# ==============================
@app.get("/alerts", dependencies=[Depends(verify_api_key)])
def get_alerts():
    df = pd.read_csv("agroreach_farmer_priority_output.csv")
    high_priority = df[df["predicted_intervention_level"] == 2].copy()
    high_priority["risk_score"] = (high_priority["risk_score"] * 100).round(1)

    zone_mapping = {
        0: "North Central", 1: "North East", 2: "North West",
        3: "South East", 4: "South South", 5: "South West"
    }
    high_priority["zone_name"] = high_priority["zone"].map(zone_mapping)

    alerts = []
    for _, row in high_priority.iterrows():
        reasons = []
        if row["yield_original"] < 1:
            reasons.append("Low crop yield")
        if row["has_extension_access"] == 0:
            reasons.append("No extension access")
        if row["shock_level"] > 0:
            reasons.append("Experienced agricultural shock")
        if row["received_credit"] == 0:
            reasons.append("No credit access")

        alerts.append({
            "hhid": int(row["hhid"]),
            "risk_score": row["risk_score"],
            "zone": row["zone_name"],
            "reasons": reasons,
            "shock_level": int(row["shock_level"]),
            "has_extension_access": int(row["has_extension_access"]),
            "yield_original": round(float(row["yield_original"]), 2)
        })

    return {
        "total_alerts": len(alerts),
        "alerts": alerts
    }

# ==============================
# PREDICT SINGLE FARMER
# ==============================
@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(farmer: FarmerInput):
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
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    risk_score = round(float(probabilities[2]) * 100, 1)

    priority_map = {0: "Low Priority", 1: "Medium Priority", 2: "High Priority"}

    shap_values = explainer.shap_values(input_data)
    shap_class2 = shap_values[:, :, 2][0]
    shap_df = pd.DataFrame({
        "feature": input_data.columns.tolist(),
        "impact": shap_class2
    }).sort_values("impact", ascending=False)

    top_reasons = shap_df[shap_df["impact"] > 0].head(3)["feature"].tolist()

    return {
        "prediction": int(prediction),
        "priority_label": priority_map[int(prediction)],
        "risk_score_percent": risk_score,
        "probabilities": {
            "low": round(float(probabilities[0]) * 100, 1),
            "medium": round(float(probabilities[1]) * 100, 1),
            "high": round(float(probabilities[2]) * 100, 1)
        },
        "top_risk_factors": top_reasons
    }