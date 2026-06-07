from datetime import datetime
from pydantic import BaseModel,Field


class PredictionResponse(BaseModel):
    prediction: int
    priority_label: int
    risk_score_percent: float
    top_risk_factors: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
