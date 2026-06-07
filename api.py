# import hashlib
# import os
# import secrets
# from contextlib import asynccontextmanager
# from datetime import datetime
# from typing import Optional
#
# import joblib
# import pandas as pd
# from cachetools import TTLCache
# from fastapi import FastAPI, HTTPException, Depends, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from pydantic import BaseModel
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.errors import RateLimitExceeded
# from slowapi.util import get_remote_address
# from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
# from sqlalchemy.orm import declarative_base, sessionmaker, Session
#
# # ==============================
# # DATABASE SETUP — POSTGRESQL
# # ==============================
# DATABASE_URL = os.environ.get("DATABASE_URL", "")
#
# # Railway uses postgres:// but SQLAlchemy needs postgresql://
# if DATABASE_URL.startswith("postgres://"):
#     DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
#
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()
#
# # ==============================
# # DATABASE MODELS
# # ==============================
# class PredictionLog(Base):
#     __tablename__ = "prediction_logs"
#
#     id = Column(Integer, primary_key=True, index=True)
#     prediction = Column(Integer)
#     priority_label = Column(String)
#     risk_score_percent = Column(Float)
#     top_risk_factors = Column(String)
#     created_at = Column(DateTime, default=datetime.utcnow)
#
# class APIKey(Base):
#     __tablename__ = "api_keys"
#
#     id = Column(Integer, primary_key=True, index=True)
#     key_hash = Column(String, unique=True, index=True)
#     client_name = Column(String)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     is_active = Column(Integer, default=1)
#
# # ==============================
# # NEW: INTERVENTION TABLE
# # ==============================
# # Think of this like a "visit record" form.
# # Every time a field officer visits a farmer and helps them,
# # they fill this in. We store it here in PostgreSQL.
# # The farmer_id links back to the farmer in our CSV.
# # One farmer can have MANY interventions over time.
# class Intervention(Base):
#     __tablename__ = "interventions"
#
#     id = Column(Integer, primary_key=True, index=True)
#
#     # Which farmer was helped — links to hhid in our CSV
#     farmer_id = Column(Integer, index=True)
#
#     # Who helped them — field officer's name
#     officer_name = Column(String)
#
#     # What type of help was given
#     # e.g. "fertilizer_support", "extension_visit", "credit_facilitation",
#     #      "seed_distribution", "training", "other"
#     intervention_type = Column(String)
#
#     # What was the result — did it help?
#     # e.g. "pending", "successful", "no_response", "follow_up_needed"
#     outcome = Column(String, default="pending")
#
#     # Any extra notes the officer wants to add
#     notes = Column(String, nullable=True)
#
#     # The farmer's risk score AT THE TIME of intervention
#     # Important — so we can track if it improves later
#     risk_score_at_intervention = Column(Float, nullable=True)
#
#     # When the intervention happened
#     created_at = Column(DateTime, default=datetime.utcnow)
#
#     # When the outcome was last updated
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
#
#
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
#
# def init_db():
#     Base.metadata.create_all(bind=engine)
#     print("PostgreSQL tables created successfully!")
#
# # ==============================
# # LOAD MODEL AND FEATURES
# # ==============================
# model = joblib.load("farmerxential_model.pkl")
# features = joblib.load("farmerxential_features.pkl")
# explainer = joblib.load("farmerxential_shap_explainer.pkl")
#
# # ==============================
# # RATE LIMITER
# # ==============================
# limiter = Limiter(key_func=get_remote_address)
#
# # ==============================
# # CACHE — stores responses for 5 minutes
# # ==============================
# stats_cache = TTLCache(maxsize=100, ttl=300)
# farmers_cache = TTLCache(maxsize=100, ttl=300)
#
# # ==============================
# # LIFESPAN
# # ==============================
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     init_db()
#     yield
#
# # ==============================
# # APP
# # ==============================
# app = FastAPI(
#     title="FarmerXential API",
#     version="3.0",
#     description="AI-powered agricultural intelligence system by Lalishank Holdings Limited",
#     lifespan=lifespan
# )
#
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
#
# # ==============================
# # CORS
# # ==============================
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # ==============================
# # OAUTH2 + DYNAMIC API KEY AUTH
# # ==============================
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")
#
# def hash_key(key: str) -> str:
#     return hashlib.sha256(key.encode()).hexdigest()
#
# def verify_api_key(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
#     key_hash = hash_key(token)
#     api_key = db.query(APIKey).filter(
#         APIKey.key_hash == key_hash,
#         APIKey.is_active == 1
#     ).first()
#     if not api_key:
#         raise HTTPException(
#             status_code=401,
#             detail="Invalid or expired API key."
#         )
#     return api_key
#
#
# # NEW: Schema for creating an intervention
# # Think of this like the fields on the form the field officer fills in
# class InterventionInput(BaseModel):
#     farmer_id: int                          # Which farmer (hhid)
#     officer_name: str                       # Field officer's name
#     intervention_type: str                  # Type of help given
#     notes: Optional[str] = None            # Extra notes (optional)
#     risk_score_at_intervention: Optional[float] = None  # Farmer's risk score now
#
# # NEW: Schema for updating an intervention outcome
# # After helping the farmer, officer comes back to say what happened
# class InterventionUpdate(BaseModel):
#     outcome: str                            # "successful", "no_response", "follow_up_needed"
#     notes: Optional[str] = None            # Updated notes
#
#
# # ==============================
# # HEALTH CHECK
# # ==============================
# @app.get("/")
# def home():
#     return {
#         "message": "FarmerXential API is running!",
#         "version": "3.0",
#         "status": "healthy",
#         "product": "FarmerXential by Lalishank Holdings Limited"
#     }
#
# # ==============================
# # GENERATE API KEY — Dynamic key generation
# # ==============================
# @app.post("/auth/register")
# @limiter.limit("5/minute")
# def register_client(
#     request: Request,
#     client_name: str,
#     db: Session = Depends(get_db)
# ):
#     """
#     Generate a dynamic API key for a new client.
#     Think of it like: sign up and get your unique access key.
#     """
#     raw_key = secrets.token_urlsafe(32)
#     key_hash = hash_key(raw_key)
#
#     new_key = APIKey(
#         key_hash=key_hash,
#         client_name=client_name,
#         created_at=datetime.utcnow(),
#         is_active=1
#     )
#     db.add(new_key)
#     db.commit()
#
#     return {
#         "message": f"API key generated for {client_name}",
#         "api_key": raw_key,
#         "warning": "Store this key securely. It will not be shown again.",
#         "usage": "Include in requests as: Authorization: Bearer YOUR_KEY"
#     }
#
# # ==============================
# # GET TOKEN — OAuth2 token endpoint
# # ==============================
# @app.post("/auth/token")
# @limiter.limit("10/minute")
# def get_token(
#     request: Request,
#     form_data: OAuth2PasswordRequestForm = Depends(),
#     db: Session = Depends(get_db)
# ):
#     """
#     OAuth2 token endpoint.
#     Username = client_name, Password = api_key
#     """
#     key_hash = hash_key(form_data.password)
#     api_key = db.query(APIKey).filter(
#         APIKey.key_hash == key_hash,
#         APIKey.is_active == 1
#     ).first()
#
#     if not api_key or api_key.client_name != form_data.username:
#         raise HTTPException(
#             status_code=401,
#             detail="Invalid credentials."
#         )
#
#     return {
#         "access_token": form_data.password,
#         "token_type": "bearer"
#     }
#
# # ==============================
# # STATS — Cached
# # ==============================
# @app.get("/stats")
# @limiter.limit("30/minute")
# def get_stats(
#     request: Request,
#     db: Session = Depends(get_db),
#     api_key: APIKey = Depends(verify_api_key)
# ):
#     # Check cache first
#     if "stats" in stats_cache:
#         return stats_cache["stats"]
#
#     # Load from CSV since this is reference data
#     df = pd.read_csv("farmerxential_farmer_priority_output.csv")
#     df["risk_score_pct"] = df["risk_score"] * 100
#
#     zone_mapping = {
#         0: "North Central", 1: "North East", 2: "North West",
#         3: "South East", 4: "South South", 5: "South West"
#     }
#
#     result = {
#         "total_farmers": len(df),
#         "high_priority_count": int((df["predicted_intervention_level"] == 2).sum()),
#         "medium_priority_count": int((df["predicted_intervention_level"] == 1).sum()),
#         "low_priority_count": int((df["predicted_intervention_level"] == 0).sum()),
#         "active_alerts": int((df["risk_score"] > 0.7).sum()),
#         "avg_risk_score": round(float(df["risk_score_pct"].mean()), 1),
#         "zones": {
#             zone_mapping.get(int(zone), str(zone)): int(count)
#             for zone, count in df.groupby("zone")["predicted_intervention_level"]
#             .apply(lambda x: (x == 2).sum()).items()
#         },
#         "cached": False
#     }
#
#     # Store in cache
#     stats_cache["stats"] = {**result, "cached": True}
#     return result
#
# # ==============================
# # GET FARMERS — Cached with pagination
# # ==============================
# @app.get("/farmers")
# @limiter.limit("30/minute")
# def get_farmers(
#     request: Request,
#     zone: Optional[int] = None,
#     priority: Optional[int] = None,
#     limit: int = 100,
#     offset: int = 0,
#     api_key: APIKey = Depends(verify_api_key)
# ):
#     cache_key = f"farmers_{zone}_{priority}_{limit}_{offset}"
#
#     if cache_key in farmers_cache:
#         return farmers_cache[cache_key]
#
#     df = pd.read_csv("farmerxential_farmer_priority_output.csv")
#     df["risk_score"] = (df["risk_score"] * 100).round(1)
#     df["priority_label"] = df["predicted_intervention_level"].map({
#         0: "Low Priority", 1: "Medium Priority", 2: "High Priority"
#     })
#
#     zone_mapping = {
#         0: "North Central", 1: "North East", 2: "North West",
#         3: "South East", 4: "South South", 5: "South West"
#     }
#     df["zone_name"] = df["zone"].map(zone_mapping)
#
#     if zone is not None:
#         df = df[df["zone"] == zone]
#     if priority is not None:
#         df = df[df["predicted_intervention_level"] == priority]
#
#     #df = df.sort_values("risk_score", ascending=False)
#     total = len(df)
#     df = df.iloc[offset:offset + limit]
#
#     result = {
#         "total": total,
#         "limit": limit,
#         "offset": offset,
#         "farmers": df.to_dict(orient="records")
#     }
#
#     farmers_cache[cache_key] = result
#     return result
#
# # ==============================
# # GET ALERTS — Cached
# # ==============================
# @app.get("/alerts")
# @limiter.limit("30/minute")
# def get_alerts(
#     request: Request,
#     api_key: APIKey = Depends(verify_api_key)
# ):
#     if "alerts" in stats_cache:
#         return stats_cache["alerts"]
#
#     df = pd.read_csv("farmerxential_farmer_priority_output.csv")
#     high_priority = df[df["predicted_intervention_level"] == 2].copy()
#     high_priority["risk_score"] = (high_priority["risk_score"] * 100).round(1)
#
#     zone_mapping = {
#         0: "North Central", 1: "North East", 2: "North West",
#         3: "South East", 4: "South South", 5: "South West"
#     }
#     high_priority["zone_name"] = high_priority["zone"].map(zone_mapping)
#
#     alerts = []
#     for _, row in high_priority.iterrows():
#         reasons = []
#         if row["yield_original"] < 1:
#             reasons.append("Low crop yield")
#         if row["has_extension_access"] == 0:
#             reasons.append("No extension access")
#         if row["shock_level"] > 0:
#             reasons.append("Experienced agricultural shock")
#         if row["received_credit"] == 0:
#             reasons.append("No credit access")
#
#         alerts.append({
#             "hhid": int(row["hhid"]),
#             "risk_score": row["risk_score"],
#             "priority_label": "High Priority",
#             "zone_name": row["zone_name"],
#             "reasons": reasons,
#             "shock_level": int(row["shock_level"]),
#             "has_extension_access": int(row["has_extension_access"]),
#             "yield_original": round(float(row["yield_original"]), 2)
#         })
#
#     result = {"total_alerts": len(alerts), "alerts": alerts}
#     stats_cache["alerts"] = result
#     return result
#
# # ==============================
# # PREDICT — Core ML endpoint
# # ==============================
# @app.post("/predict")
# @limiter.limit("20/minute")
# def predict(
#     request: Request,
#     farmer: PredictionRequest,
#     db: Session = Depends(get_db),
#     api_key: APIKey = Depends(verify_api_key)
# ):
#     input_data = pd.DataFrame([{
#         "yield": farmer.yield_value,
#         "has_extension_access": farmer.has_extension_access,
#         "household_max_education": farmer.household_max_education,
#         "shock_level": farmer.shock_level,
#         "received_assistance": farmer.received_assistance,
#         "used_fertilizer": farmer.used_fertilizer,
#         "land_size": farmer.land_size,
#         "household_size": farmer.household_size,
#         "zone": farmer.zone,
#         "transport_cost": farmer.transport_cost,
#         "dependency_ratio": farmer.dependency_ratio,
#         "asset_score": farmer.asset_score,
#         "postharvest_activity_score": farmer.postharvest_activity_score,
#         "crop_loss_risk_score": farmer.crop_loss_risk_score,
#         "crop_diversity_score": farmer.crop_diversity_score,
#         "digital_access_score": farmer.digital_access_score,
#         "has_veterinary_access": farmer.has_veterinary_access,
#         "market_access_score": farmer.market_access_score,
#         "is_rural": farmer.is_rural,
#         "rainfall_anomaly": farmer.rainfall_anomaly,
#         "drought_risk": farmer.drought_risk,
#         "cultivates_crops": farmer.cultivates_crops,
#         "received_credit": farmer.received_credit,
#         "head_gender": farmer.head_gender
#     }])
#
#     input_data = input_data[features]
#     prediction = model.predict(input_data)[0]
#     probabilities = model.predict_proba(input_data)[0]
#     risk_score = round(float(probabilities[2]) * 100, 1)
#
#     priority_map = {0: "Low Priority", 1: "Medium Priority", 2: "High Priority"}
#     priority_label = priority_map[int(prediction)]
#
#     shap_values = explainer.shap_values(input_data)
#     shap_class2 = shap_values[:, :, 2][0]
#     shap_df = pd.DataFrame({
#         "feature": input_data.columns.tolist(),
#         "impact": shap_class2
#     }).sort_values("impact", ascending=False)
#
#     top_reasons = shap_df[shap_df["impact"] > 0].head(3)["feature"].tolist()
#
#     # Persist prediction to PostgreSQL
#     log = PredictionLog(
#         prediction=int(prediction),
#         priority_label=priority_label,
#         risk_score_percent=risk_score,
#         top_risk_factors=", ".join(top_reasons),
#         created_at=datetime.utcnow()
#     )
#     db.add(log)
#     db.commit()
#
#     return {
#         "prediction": int(prediction),
#         "priority_label": priority_label,
#         "risk_score_percent": risk_score,
#         "probabilities": {
#             "low": round(float(probabilities[0]) * 100, 1),
#             "medium": round(float(probabilities[1]) * 100, 1),
#             "high": round(float(probabilities[2]) * 100, 1)
#         },
#         "top_risk_factors": top_reasons
#     }
#
# # ==============================
# # PREDICTION HISTORY — from PostgreSQL
# # ==============================
# @app.get("/predictions/history")
# @limiter.limit("20/minute")
# def get_prediction_history(
#     request: Request,
#     limit: int = 50,
#     db: Session = Depends(get_db),
#     api_key: APIKey = Depends(verify_api_key)
# ):
#     logs = db.query(PredictionLog)\
#         .order_by(PredictionLog.created_at.desc())\
#         .limit(limit)\
#         .all()
#
#     return {
#         "total": len(logs),
#         "predictions": [
#             {
#                 "id": log.id,
#                 "prediction": log.prediction,
#                 "priority_label": log.priority_label,
#                 "risk_score_percent": log.risk_score_percent,
#                 "top_risk_factors": log.top_risk_factors,
#                 "created_at": log.created_at.isoformat()
#             }
#             for log in logs
#         ]
#     }
#
#
# # ==============================
# # NEW: INTERVENTION ENDPOINTS
# # ==============================
#
# # --- Record a new intervention ---
# # This is called when a field officer visits a farmer and logs it
# @app.post("/interventions")
# @limiter.limit("30/minute")
# def create_intervention(
#     request: Request,
#     data: InterventionInput,
#     db: Session = Depends(get_db),
#     api_key: APIKey = Depends(verify_api_key)
# ):
#     """
#     Record that a field officer visited and helped a farmer.
#     Think of it like: ticking off a farmer on the to-do list
#     and writing what you did.
#     """
#     # Create the new intervention record
#     intervention = Intervention(
#         farmer_id=data.farmer_id,
#         officer_name=data.officer_name,
#         intervention_type=data.intervention_type,
#         notes=data.notes,
#         risk_score_at_intervention=data.risk_score_at_intervention,
#         outcome="pending",                  # Always starts as pending
#         created_at=datetime.utcnow(),
#         updated_at=datetime.utcnow()
#     )
#
#     db.add(intervention)
#     db.commit()
#     db.refresh(intervention)               # Get the new ID back from PostgreSQL
#
#     return {
#         "message": "Intervention recorded successfully",
#         "intervention_id": intervention.id,
#         "farmer_id": intervention.farmer_id,
#         "officer_name": intervention.officer_name,
#         "intervention_type": intervention.intervention_type,
#         "outcome": intervention.outcome,
#         "created_at": intervention.created_at.isoformat()
#     }
#
#
# # --- Get all interventions for one specific farmer ---
# # Field officer or NDDC wants to see the full history for farmer 1234
# @app.get("/interventions/farmer/{farmer_id}")
# @limiter.limit("30/minute")
# def get_farmer_interventions(
#     request: Request,
#     farmer_id: int,
#     db: Session = Depends(get_db),
#     api_key: APIKey = Depends(verify_api_key)
# ):
#     """
#     Get the full intervention history for one farmer.
#     Think of it like: opening a farmer's medical file and
#     reading all their past visits.
#     """
#     interventions = db.query(Intervention)\
#         .filter(Intervention.farmer_id == farmer_id)\
#         .order_by(Intervention.created_at.desc())\
#         .all()
#
#     return {
#         "farmer_id": farmer_id,
#         "total_interventions": len(interventions),
#         "interventions": [
#             {
#                 "id": i.id,
#                 "officer_name": i.officer_name,
#                 "intervention_type": i.intervention_type,
#                 "outcome": i.outcome,
#                 "notes": i.notes,
#                 "risk_score_at_intervention": i.risk_score_at_intervention,
#                 "created_at": i.created_at.isoformat(),
#                 "updated_at": i.updated_at.isoformat()
#             }
#             for i in interventions
#         ]
#     }
#
#
# # --- Get ALL interventions (for NDDC dashboard overview) ---
# # Government wants to see: how many farmers have been helped total?
# @app.get("/interventions")
# @limiter.limit("30/minute")
# def get_all_interventions(
#     request: Request,
#     outcome: Optional[str] = None,         # Filter by outcome e.g. "successful"
#     officer_name: Optional[str] = None,    # Filter by officer
#     limit: int = 100,
#     offset: int = 0,
#     db: Session = Depends(get_db),
#     api_key: APIKey = Depends(verify_api_key)
# ):
#     """
#     Get all interventions across all farmers.
#     Think of it like: the full register of every visit ever made.
#     NDDC can use this to see the total impact at a glance.
#     """
#     query = db.query(Intervention)
#
#     # Apply filters if provided
#     if outcome:
#         query = query.filter(Intervention.outcome == outcome)
#     if officer_name:
#         query = query.filter(Intervention.officer_name == officer_name)
#
#     total = query.count()
#
#     interventions = query\
#         .order_by(Intervention.created_at.desc())\
#         .offset(offset)\
#         .limit(limit)\
#         .all()
#
#     # Summary counts — useful for NDDC dashboard
#     all_records = db.query(Intervention).all()
#     summary = {
#         "total": len(all_records),
#         "pending": sum(1 for i in all_records if i.outcome == "pending"),
#         "successful": sum(1 for i in all_records if i.outcome == "successful"),
#         "no_response": sum(1 for i in all_records if i.outcome == "no_response"),
#         "follow_up_needed": sum(1 for i in all_records if i.outcome == "follow_up_needed")
#     }
#
#     return {
#         "summary": summary,
#         "total_filtered": total,
#         "limit": limit,
#         "offset": offset,
#         "interventions": [
#             {
#                 "id": i.id,
#                 "farmer_id": i.farmer_id,
#                 "officer_name": i.officer_name,
#                 "intervention_type": i.intervention_type,
#                 "outcome": i.outcome,
#                 "notes": i.notes,
#                 "risk_score_at_intervention": i.risk_score_at_intervention,
#                 "created_at": i.created_at.isoformat(),
#                 "updated_at": i.updated_at.isoformat()
#             }
#             for i in interventions
#         ]
#     }
#
#
# # --- Update intervention outcome ---
# # Officer comes back later: "I visited, here's what happened"
# @app.patch("/interventions/{intervention_id}")
# @limiter.limit("30/minute")
# def update_intervention(
#     request: Request,
#     intervention_id: int,
#     data: InterventionUpdate,
#     db: Session = Depends(get_db),
#     api_key: APIKey = Depends(verify_api_key)
# ):
#     """
#     Update the outcome of an intervention.
#     Think of it like: going back to the register and writing
#     what happened after the visit.
#     """
#     # Find the intervention
#     intervention = db.query(Intervention)\
#         .filter(Intervention.id == intervention_id)\
#         .first()
#
#     if not intervention:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Intervention {intervention_id} not found."
#         )
#
#     # Valid outcomes only
#     valid_outcomes = ["pending", "successful", "no_response", "follow_up_needed"]
#     if data.outcome not in valid_outcomes:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid outcome. Must be one of: {valid_outcomes}"
#         )
#
#     # Update the record
#     intervention.outcome = data.outcome
#     if data.notes:
#         intervention.notes = data.notes
#     intervention.updated_at = datetime.utcnow()
#
#     db.commit()
#     db.refresh(intervention)
#
#     return {
#         "message": "Intervention updated successfully",
#         "intervention_id": intervention.id,
#         "farmer_id": intervention.farmer_id,
#         "outcome": intervention.outcome,
#         "updated_at": intervention.updated_at.isoformat()
#     }