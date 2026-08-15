from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionProbabilities(BaseModel):
    """Learned model output over the three forward-stress classes. All-zero above `low` when no
    cloud-free scene was available and the model did not contribute.

    **These describe the vegetation hazard term only, not overall risk.** They are the calibrated
    posterior for "will this field's canopy be stressed relative to its peers in 30 days" — one of
    four hazards that feed the noisy-OR, alongside drought, disease and heat. `prediction` and
    `risk_score_percent` come from the full Hazard x Exposure x Vulnerability composition, so they
    routinely disagree with the argmax here: a field can be 66% "low" on canopy stress and still be
    CRITICAL because accumulated blight severity crossed the spray threshold.

    The `of` field carries that caveat in the payload itself, because a consumer reading raw JSON
    does not see this docstring.
    """
    low: float
    medium: float
    high: float
    of: str = Field(
        default="vegetation hazard (peer-relative canopy stress, 30-day horizon)",
        description="What these probabilities are over. NOT overall risk — see `risk_score_percent`.",
    )


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
    # Handle for the audit row. Pass it back on POST /outcomes to link what happened to what was
    # predicted — this is the join that produces training labels.
    prediction_id: Optional[int] = None
    model_version: str = "none"
    # "precomputed" when served from the nightly feature table, "live" when computed in-request.
    feature_source: str = "live"
    features_computed_at: Optional[datetime] = None
    crop_type: str
    phenology_stage: str
    prediction: int
    priority_label: str
    risk_score_percent: float
    probabilities: PredictionProbabilities = Field(
        description="Calibrated posterior for the *vegetation hazard* term only. Do not read its "
                    "argmax as the overall risk class — that is `prediction`.",
    )
    top_risk_factors: List[str]
    inference: InferenceDetail
    risk_assessment: RiskAssessmentDetail
    crop_health: Optional[CropHealthDetail] = None
    spatiotemporal_indices: SpatiotemporalIndices
    microclimate_metrics: MicroclimateMetrics
    recommended_action: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
