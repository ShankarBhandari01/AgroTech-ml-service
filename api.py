from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from cachetools import TTLCache
import joblib
import pandas as pd
import numpy as np
import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ==============================
# DATABASE SETUP — POSTGRESQL
# ==============================
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Railway uses postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==============================
# DATABASE MODELS
# ==============================
class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    prediction = Column(Integer)
    priority_label = Column(String)
    risk_score_percent = Column(Float)
    top_risk_factors = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key_hash = Column(String, unique=True, index=True)
    client_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
    print("PostgreSQL tables created successfully!")

# ==============================
# LOAD MODEL AND FEATURES
# ==============================
model = joblib.load("farmerxential_model.pkl")
features = joblib.load("farmerxential_features.pkl")
explainer = joblib.load("farmerxential_shap_explainer.pkl")

# ==============================
# RATE LIMITER
# ==============================
limiter = Limiter(key_func=get_remote_address)

# ==============================
# CACHE — stores responses for 5 minutes
# ==============================
stats_cache = TTLCache(maxsize=100, ttl=300)
farmers_cache = TTLCache(maxsize=100, ttl=300)

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
app = FastAPI(
    title="FarmerXential API",
    version="3.0",
    description="AI-powered agricultural intelligence system by Lalishank Holdings Limited",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
# OAUTH2 + DYNAMIC API KEY AUTH
# ==============================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def verify_api_key(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    key_hash = hash_key(token)
    api_key = db.query(APIKey).filter(
        APIKey.key_hash == key_hash,
        APIKey.is_active == 1
    ).first()
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key."
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
# HEALTH CHECK
# ==============================
@app.get("/")
def home():
    return {
        "message": "FarmerXential API is running!",
        "version": "3.0",
        "status": "healthy",
        "product": "FarmerXential by Lalishank Holdings Limited"
    }

# ==============================
# GENERATE API KEY — Dynamic key generation
# ==============================
@app.post("/auth/register")
@limiter.limit("5/minute")
def register_client(
    request: Request,
    client_name: str,
    db: Session = Depends(get_db)
):
    """
    Generate a dynamic API key for a new client.
    Think of it like: sign up and get your unique access key.
    """
    raw_key = secrets.token_urlsafe(32)
    key_hash = hash_key(raw_key)

    new_key = APIKey(
        key_hash=key_hash,
        client_name=client_name,
        created_at=datetime.utcnow(),
        is_active=1
    )
    db.add(new_key)
    db.commit()

    return {
        "message": f"API key generated for {client_name}",
        "api_key": raw_key,
        "warning": "Store this key securely. It will not be shown again.",
        "usage": "Include in requests as: Authorization: Bearer YOUR_KEY"
    }

# ==============================
# GET TOKEN — OAuth2 token endpoint
# ==============================
@app.post("/auth/token")
@limiter.limit("10/minute")
def get_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    OAuth2 token endpoint.
    Username = client_name, Password = api_key
    """
    key_hash = hash_key(form_data.password)
    api_key = db.query(APIKey).filter(
        APIKey.key_hash == key_hash,
        APIKey.is_active == 1
    ).first()

    if not api_key or api_key.client_name != form_data.username:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials."
        )

    return {
        "access_token": form_data.password,
        "token_type": "bearer"
    }

# ==============================
# STATS — Cached
# ==============================
@app.get("/stats")
@limiter.limit("30/minute")
def get_stats(
    request: Request,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key)
):
    # Check cache first
    if "stats" in stats_cache:
        return stats_cache["stats"]

    # Load from CSV since this is reference data
    df = pd.read_csv("farmerxential_farmer_priority_output.csv")
    df["risk_score_pct"] = df["risk_score"] * 100

    zone_mapping = {
        0: "North Central", 1: "North East", 2: "North West",
        3: "South East", 4: "South South", 5: "South West"
    }

    result = {
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
        },
        "cached": False
    }

    # Store in cache
    stats_cache["stats"] = {**result, "cached": True}
    return result

# ==============================
# GET FARMERS — Cached with pagination
# ==============================
@app.get("/farmers")
@limiter.limit("30/minute")
def get_farmers(
    request: Request,
    zone: Optional[int] = None,
    priority: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    api_key: APIKey = Depends(verify_api_key)
):
    cache_key = f"farmers_{zone}_{priority}_{limit}_{offset}"

    if cache_key in farmers_cache:
        return farmers_cache[cache_key]

    df = pd.read_csv("farmerxential_farmer_priority_output.csv")
    df["risk_score"] = (df["risk_score"] * 100).round(1)
    df["priority_label"] = df["predicted_intervention_level"].map({
        0: "Low Priority", 1: "Medium Priority", 2: "High Priority"
    })

    zone_mapping = {
        0: "North Central", 1: "North East", 2: "North West",
        3: "South East", 4: "South South", 5: "South West"
    }
    df["zone_name"] = df["zone"].map(zone_mapping)

    if zone is not None:
        df = df[df["zone"] == zone]
    if priority is not None:
        df = df[df["predicted_intervention_level"] == priority]

    df = df.sort_values("risk_score", ascending=False)
    total = len(df)
    df = df.iloc[offset:offset + limit]

    result = {
        "total": total,
        "limit": limit,
        "offset": offset,
        "farmers": df.to_dict(orient="records")
    }

    farmers_cache[cache_key] = result
    return result

# ==============================
# GET ALERTS — Cached
# ==============================
@app.get("/alerts")
@limiter.limit("30/minute")
def get_alerts(
    request: Request,
    api_key: APIKey = Depends(verify_api_key)
):
    if "alerts" in stats_cache:
        return stats_cache["alerts"]

    df = pd.read_csv("farmerxential_farmer_priority_output.csv")
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
            "priority_label": "High Priority",
            "zone_name": row["zone_name"],
            "reasons": reasons,
            "shock_level": int(row["shock_level"]),
            "has_extension_access": int(row["has_extension_access"]),
            "yield_original": round(float(row["yield_original"]), 2)
        })

    result = {"total_alerts": len(alerts), "alerts": alerts}
    stats_cache["alerts"] = result
    return result

# ==============================
# PREDICT — Core ML endpoint
# ==============================
@app.post("/predict")
@limiter.limit("20/minute")
def predict(
    request: Request,
    farmer: FarmerInput,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key)
):
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
    priority_label = priority_map[int(prediction)]

    shap_values = explainer.shap_values(input_data)
    shap_class2 = shap_values[:, :, 2][0]
    shap_df = pd.DataFrame({
        "feature": input_data.columns.tolist(),
        "impact": shap_class2
    }).sort_values("impact", ascending=False)

    top_reasons = shap_df[shap_df["impact"] > 0].head(3)["feature"].tolist()

    # Persist prediction to PostgreSQL
    log = PredictionLog(
        prediction=int(prediction),
        priority_label=priority_label,
        risk_score_percent=risk_score,
        top_risk_factors=", ".join(top_reasons),
        created_at=datetime.utcnow()
    )
    db.add(log)
    db.commit()

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
# PREDICTION HISTORY — from PostgreSQL
# ==============================
@app.get("/predictions/history")
@limiter.limit("20/minute")
def get_prediction_history(
    request: Request,
    limit: int = 50,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key)
):
    logs = db.query(PredictionLog)\
        .order_by(PredictionLog.created_at.desc())\
        .limit(limit)\
        .all()

    return {
        "total": len(logs),
        "predictions": [
            {
                "id": log.id,
                "prediction": log.prediction,
                "priority_label": log.priority_label,
                "risk_score_percent": log.risk_score_percent,
                "top_risk_factors": log.top_risk_factors,
                "created_at": log.created_at.isoformat()
            }
            for log in logs
        ]
    }