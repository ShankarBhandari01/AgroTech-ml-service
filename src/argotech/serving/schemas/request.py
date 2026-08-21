from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# There is no per-request model selection. The vegetation hazard comes from whichever source
# `settings.VEGETATION_HAZARD_SOURCE` names, service-wide, and the response reports what actually
# answered in `model_version`. The old `model_name`/`model_alias` fields named an MLflow registry
# entry that no loader has read since the registry rewrite; pydantic ignores unknown fields, so a
# client still sending them keeps working.
class FarmerPredictionRequest(BaseModel):
    farmer_id: str

class CoordinatesColdStartPredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    state: str | None = "Kaduna"
    crop_type: str | None = "Maize"
    farm_size: float | None = 1.5

class OutcomeRequest(BaseModel):
    """What an agent, a diagnosis, or a harvest record observed for a field."""
    field_id: str
    observed_at: datetime
    outcome_type: Literal["agent_visit", "diagnosis", "harvest"]
    # Link to the prediction being evaluated. Optional, because a spontaneous field report is still
    # worth recording — it just cannot be scored against a specific prediction.
    prediction_id: int | None = None
    stress_confirmed: bool | None = None
    diagnosis: str | None = None
    yield_t_ha: float | None = Field(default=None, ge=0, le=30)
    notes: str | None = None
    reported_by: str | None = None
