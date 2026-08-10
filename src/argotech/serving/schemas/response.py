from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PredictionProbabilities(BaseModel):
    low: float
    medium: float
    high: float

class InferenceDetail(BaseModel):
    risk_level: str
    disease_type: str
    probability: float
    primary_drivers: List[str]

class SpatiotemporalIndices(BaseModel):
    ndvi: float
    ndwi: float
    evi: float
    canopy_stress_status: str
    # "sentinel-2" when the indices came from a real CDSE scene, "modelled" for the synthetic fallback.
    source: str = "modelled"

class MicroclimateMetrics(BaseModel):
    rh_85_consecutive_hrs: int
    incubation_hours: int
    soil_water_deficit_72h: float
    rainfall_anomaly: float

class CrossAttentionDetail(BaseModel):
    fusion_score: float
    peak_incubation_hour: int
    fusion_status: str

class PredictionResponse(BaseModel):
    field_id: str
    crop_type: str
    phenology_stage: str
    prediction: int
    priority_label: str
    risk_score_percent: float
    probabilities: PredictionProbabilities
    top_risk_factors: List[str]
    inference: InferenceDetail
    spatiotemporal_indices: SpatiotemporalIndices
    microclimate_metrics: MicroclimateMetrics
    cross_attention_fusion: CrossAttentionDetail
    recommended_action: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
