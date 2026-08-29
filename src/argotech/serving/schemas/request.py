from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# The field list here is `data/backend_schema.py`'s `_FARMER_FEATURES_SQL`, copied field-for-field —
# do not add or rename a field without updating that query's SELECT list too. Every field defaults
# to None, the same value a LEFT JOIN against an absent `farms`/`farmers_ml_profiles` row produces,
# so a payload that omits a field is indistinguishable downstream from a database NULL for it. See
# `serving/pipeline.py::PredictionsService._fetch_farmer_data`.
class FarmerPayload(BaseModel):
    farm_size: float | None = None
    state: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    crops: list[str] | None = None
    yield_value: float | None = None
    has_extension_access: bool | None = None
    # Protected attributes (`domain/risk.py::PROTECTED_ATTRIBUTES`) — carried for fairness
    # *measurement* only. `PredictionsService._coping_signals` must never put either into the dict
    # it hands to `risk.assess_vulnerability`, which raises if they leak in.
    household_max_education: int | None = None
    head_gender: str | None = None
    shock_level: int | None = None
    received_assistance: bool | None = None
    used_fertilizer: bool | None = None
    household_size: int | None = None
    transport_cost: float | None = None
    dependency_ratio: float | None = None
    asset_score: float | None = None
    postharvest_activity_score: float | None = None
    digital_access_score: float | None = None
    has_veterinary_access: bool | None = None
    market_access_score: float | None = None
    received_credit: bool | None = None
    crop_diversity_score: int | None = None


# There is no per-request model selection. The vegetation hazard comes from whichever source
# `settings.VEGETATION_HAZARD_SOURCE` names, service-wide, and the response reports what actually
# answered in `model_version`. The old `model_name`/`model_alias` fields named an MLflow registry
# entry that no loader has read since the registry rewrite; pydantic ignores unknown fields, so a
# client still sending them keeps working.
class FarmerPredictionRequest(BaseModel):
    farmer_id: str
    # Optional and additive: the deployed Kotlin backend sends only `farmer_id` today, so this must
    # stay absent-by-default forever until that changes. When present, `PredictionsService` uses it
    # instead of reading the backend's tables; when absent, the database read is the fallback — see
    # `pipeline.py`. Do not make this required.
    farmer: FarmerPayload | None = None

class CoordinatesColdStartPredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    state: str | None = "Kaduna"
    crop_type: str | None = "Maize"
    farm_size: float | None = 1.5

# The same shape `data/backend_schema.py::list_fields` returns today — the contract is defined by
# that query, not invented here. See `jobs/precompute.py::run` and the nightly-job-ownership plan:
# this is what lets Kotlin hand in a field list instead of this service reading the backend's schema.
class PrecomputeField(BaseModel):
    field_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    crop: str


class PrecomputeBatchRequest(BaseModel):
    fields: list[PrecomputeField] = Field(default_factory=list)


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
