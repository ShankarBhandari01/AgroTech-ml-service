from pydantic import BaseModel, Field
from typing import Optional

class FarmerPredictionRequest(BaseModel):
    farmer_id: str
    model_name: Optional[str] = "farmerXential_powerful_model"
    model_alias: Optional[str] = "prod"

class CoordinatesColdStartPredictionRequest(BaseModel):
    latitude: float
    longitude: float
    state: Optional[str] = "Kaduna"
    crop_type: Optional[str] = "Maize"
    farm_size: Optional[float] = 1.5
    model_name: Optional[str] = "farmerXential_powerful_model"
    model_alias: Optional[str] = "prod"