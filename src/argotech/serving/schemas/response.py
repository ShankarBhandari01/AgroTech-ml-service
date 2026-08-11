from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionProbabilities(BaseModel):
    """Learned model output over the three forward-stress classes. All-zero above `low` when no
    cloud-free scene was available and the model did not contribute."""
    low: float
    medium: float
    high: float


class InferenceDetail(BaseModel):
    risk_level: str
    dominant_hazard: str
    probability: float
    primary_drivers: List[str]
    # False when no Sentinel-2 scene was available and the assessment rests on physics alone.
    model_contributed: bool = True


class HazardDetail(BaseModel):
    drought: float
    disease: float
    heat: float
    vegetation: float
    combined: float
    dominant: str


class VulnerabilityDetail(BaseModel):
    score: float
    coping_capacity: float
    gaps: List[str]


class RiskAssessmentDetail(BaseModel):
    """Risk = Hazard x Exposure x Vulnerability, decomposed so the caller can act on the parts."""
    risk_score: float
    expected_loss_usd: float
    value_at_risk_usd: float
    hazard: HazardDetail
    vulnerability: VulnerabilityDetail


class CropHealthDetail(BaseModel):
    score: float
    vigour: float
    moisture: float
    anomaly: float
    status: str


class SpatiotemporalIndices(BaseModel):
    ndvi: float
    ndwi: float          # B08/B11, i.e. NDMI — kept under this name for API compatibility
    evi: float
    vci: float
    canopy_stress_status: str
    source: str = "modelled"
    sensing_date: Optional[str] = None


class MicroclimateMetrics(BaseModel):
    water_satisfaction_30d: float
    water_deficit_30d_mm: float
    longest_dry_spell_days: int
    rainfall_anomaly_30d_mm: float
    heat_stress_days: int
    cumulative_dsv: int
    spray_threshold_reached: bool
    gdd_since_onset: float


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
    risk_assessment: RiskAssessmentDetail
    crop_health: Optional[CropHealthDetail] = None
    spatiotemporal_indices: SpatiotemporalIndices
    microclimate_metrics: MicroclimateMetrics
    recommended_action: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
