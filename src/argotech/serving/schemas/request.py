from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

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

class OutcomeRequest(BaseModel):
    """What an agent, a diagnosis, or a harvest record observed for a field."""
    field_id: str
    observed_at: datetime
    outcome_type: Literal["agent_visit", "diagnosis", "harvest"]
    # Link to the prediction being evaluated. Optional, because a spontaneous field report is still
    # worth recording — it just cannot be scored against a specific prediction.
    prediction_id: Optional[int] = None
    stress_confirmed: Optional[bool] = None
    diagnosis: Optional[str] = None
    yield_t_ha: Optional[float] = Field(default=None, ge=0, le=30)
    notes: Optional[str] = None
    reported_by: Optional[str] = None
