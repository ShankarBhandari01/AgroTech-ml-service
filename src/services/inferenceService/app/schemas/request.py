from pydantic import BaseModel, Field
from typing import Optional

class PredictionRequest(BaseModel):
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

class FarmerPredictionRequest(BaseModel):
    farmer_id: str
    model_name: Optional[str] = "farmerXential_model"
    model_alias: Optional[str] = "prod"