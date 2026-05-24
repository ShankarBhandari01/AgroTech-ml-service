from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import sqlite3
from typing import Optional

# ==============================
# LOAD MODEL AND FEATURES
# ==============================
model = joblib.load("agroreach_model.pkl")
features = joblib.load("agroreach_features.pkl")
explainer = joblib.load("agroreach_shap_explainer.pkl")

# ==============================
# DATABASE SETUP
# ==============================
DB_PATH = "agroreach.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS farmers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hhid INTEGER UNIQUE,
            intervention_level INTEGER,
            predicted_intervention_level INTEGER,
            risk_score REAL,
            yield_original REAL,
            land_size_original REAL,
            household_size INTEGER,
            used_fertilizer INTEGER,
            household_max_education INTEGER,
            has_extension_access INTEGER,
            shock_level INTEGER,
            asset_score REAL,
            postharvest_activity_score REAL,
            crop_loss_risk_score REAL,
            digital_access_score REAL,
            market_access_score REAL,
            transport_cost REAL,
            received_credit INTEGER,
            zone INTEGER,
            priority_label TEXT,
            zone_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    count = cursor.execute("SELECT COUNT(*) FROM farmers").fetchone()[0]
    if count == 0:
        print("Loading CSV data into database...")
        df = pd.read_csv("agroreach_farmer_priority_output.csv")

        zone_mapping = {
            0: "North Central", 1: "North East", 2: "North West",
            3: "South East", 4: "South South", 5: "South West"
        }
        df["priority_label"] = df["predicted_intervention_level"].map({
            0: "Low Priority", 1: "Medium Priority", 2: "High Priority"
        })
        df["zone_name"] = df["zone"].map(zone_mapping)
        df.to_sql("farmers", conn, if_exists="append", index=False)
        print(f"Loaded {len(df)} farmers into database!")

    conn.commit()
    conn.close()

# ==============================
# LIFESPAN
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

# ==============================
# APP
# ==============================
app = FastAPI(title="AgroReach API", version="2.0", lifespan=lifespan)

# ==============================
# CORS
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# API KEY AUTH
# ==============================
API_KEY = os.environ.get("AGROREACH_API_KEY", "agroreach-dev-key-2024")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key."
        )
    return api_key

# ==============================
# INPUT SCHEMAS
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

class FarmerUpdate(BaseModel):
    intervention_level: Optional[int] = None
    shock_level: Optional[int] = None
    has_extension_access: Optional[int] = None
    received_credit: Optional[int] = None

# ==============================
# HEALTH CHECK
# ==============================
@app.get("/")
def home():
    return {
        "message": "AgroReach API is running!",
        "version": "2.0",
        "status": "healthy"
    }

# ==============================
# STATS
# ==============================
@app.get("/stats", dependencies=[Depends(verify_api_key)])
def get_stats():
    conn = get_db()
    cursor = conn.cursor()

    total = cursor.execute("SELECT COUNT(*) FROM farmers").fetchone()[0]
    high = cursor.execute("SELECT COUNT(*) FROM farmers WHERE predicted_intervention_level = 2").fetchone()[0]
    medium = cursor.execute("SELECT COUNT(*) FROM farmers WHERE predicted_intervention_level = 1").fetchone()[0]
    low = cursor.execute("SELECT COUNT(*) FROM farmers WHERE predicted_intervention_level = 0").fetchone()[0]
    alerts = cursor.execute("SELECT COUNT(*) FROM farmers WHERE risk_score > 0.7").fetchone()[0]
    avg_risk = cursor.execute("SELECT AVG(risk_score) FROM farmers").fetchone()[0]

    zones = cursor.execute("""
        SELECT zone_name, COUNT(*) as count
        FROM farmers
        WHERE predicted_intervention_level = 2
        GROUP BY zone_name
    """).fetchall()

    conn.close()

    return {
        "total_farmers": total,
        "high_priority_count": high,
        "medium_priority_count": medium,
        "low_priority_count": low,
        "active_alerts": alerts,
        "avg_risk_score": round(float(avg_risk) * 100, 1) if avg_risk else 0,
        "zones": {row["zone_name"]: row["count"] for row in zones}
    }

# ==============================
# GET ALL FARMERS
# ==============================
@app.get("/farmers", dependencies=[Depends(verify_api_key)])
def get_farmers(
    zone: Optional[int] = None,
    priority: Optional[int] = None,
    limit: int = 100,
    offset: int = 0
):
    conn = get_db()
    cursor = conn.cursor()

    query = "SELECT * FROM farmers WHERE 1=1"
    params = []

    if zone is not None:
        query += " AND zone = ?"
        params.append(zone)

    if priority is not None:
        query += " AND predicted_intervention_level = ?"
        params.append(priority)

    query += " ORDER BY risk_score DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    farmers = cursor.execute(query, params).fetchall()
    total = cursor.execute("SELECT COUNT(*) FROM farmers").fetchone()[0]
    conn.close()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "farmers": [dict(f) for f in farmers]
    }

# ==============================
# GET SINGLE FARMER
# ==============================
@app.get("/farmers/{hhid}", dependencies=[Depends(verify_api_key)])
def get_farmer(hhid: int):
    conn = get_db()
    farmer = conn.execute(
        "SELECT * FROM farmers WHERE hhid = ?", (hhid,)
    ).fetchone()
    conn.close()

    if not farmer:
        raise HTTPException(status_code=404, detail=f"Farmer {hhid} not found")

    return dict(farmer)

# ==============================
# UPDATE FARMER
# ==============================
@app.patch("/farmers/{hhid}", dependencies=[Depends(verify_api_key)])
def update_farmer(hhid: int, update: FarmerUpdate):
    conn = get_db()

    farmer = conn.execute(
        "SELECT * FROM farmers WHERE hhid = ?", (hhid,)
    ).fetchone()

    if not farmer:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Farmer {hhid} not found")

    updates = {k: v for k, v in update.dict().items() if v is not None}

    if updates:
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [hhid]
        conn.execute(
            f"UPDATE farmers SET {set_clause} WHERE hhid = ?", values
        )
        conn.commit()

    updated = conn.execute(
        "SELECT * FROM farmers WHERE hhid = ?", (hhid,)
    ).fetchone()
    conn.close()

    return dict(updated)

# ==============================
# GET ALERTS
# ==============================
@app.get("/alerts", dependencies=[Depends(verify_api_key)])
def get_alerts():
    conn = get_db()
    high_priority = conn.execute("""
        SELECT * FROM farmers
        WHERE predicted_intervention_level = 2
        ORDER BY risk_score DESC
    """).fetchall()
    conn.close()

    alerts = []
    for row in high_priority:
        row = dict(row)
        reasons = []
        if row.get("yield_original", 1) < 1:
            reasons.append("Low crop yield")
        if row.get("has_extension_access", 1) == 0:
            reasons.append("No extension access")
        if row.get("shock_level", 0) > 0:
            reasons.append("Experienced agricultural shock")
        if row.get("received_credit", 1) == 0:
            reasons.append("No credit access")

        alerts.append({
            "hhid": row["hhid"],
            "risk_score": round(float(row["risk_score"]) * 100, 1),
            "priority_label": row["priority_label"],
            "zone_name": row["zone_name"],
            "reasons": reasons,
            "shock_level": row.get("shock_level", 0),
            "has_extension_access": row.get("has_extension_access", 0),
            "yield_original": round(float(row.get("yield_original", 0)), 2)
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