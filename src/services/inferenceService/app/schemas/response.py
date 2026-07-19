from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict

class PredictionProbabilities(BaseModel):
    low: float
    medium: float
    high: float

class PredictionResponse(BaseModel):
    prediction: int
    priority_label: str
    risk_score_percent: float
    probabilities: PredictionProbabilities
    top_risk_factors: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
