import logging
from types import SimpleNamespace

from fastapi import APIRouter, HTTPException

from argotech.serving.deps import DbSessionDep, ModelManagerDep
from argotech.serving.pipeline import PredictionsService
from argotech.serving.schemas.request import CoordinatesColdStartPredictionRequest, FarmerPredictionRequest
from argotech.serving.schemas.response import PredictionResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _summary(prediction: PredictionResponse) -> dict:
    """The response fields worth a log line. The full body is ~40 fields of nested detail, which
    belongs in the stored prediction row, not in the log."""
    return {
        "prediction_id": prediction.prediction_id,
        "model_version": prediction.model_version,
        "feature_source": prediction.feature_source,
        "risk_level": prediction.inference.risk_level,
        "risk_score_percent": prediction.risk_score_percent,
        "dominant_hazard": prediction.inference.dominant_hazard,
        "probabilities": prediction.probabilities.model_dump(),
    }


@router.post("/predict/farmer", response_model=PredictionResponse)
async def predict_farmer(
        payload: FarmerPredictionRequest,
        model_manager: ModelManagerDep,
        db: DbSessionDep,
):
    logger.info("predict/farmer request: %s", payload.model_dump())
    try:
        predictions_service = PredictionsService(
            model_manager=model_manager,
            data=payload,
            db=db
        )
        prediction = await predictions_service.predict()
        logger.info("predict/farmer response: %s", _summary(prediction))
        return prediction

    except HTTPException:
        raise
    except Exception as e:
        # Log the detail, return a generic message: str(e) on a SQLAlchemy error is the entire
        # statement plus parameters, and this response crosses a service boundary.
        logger.exception("predict/farmer failed for %s", payload.farmer_id)
        raise HTTPException(status_code=500,
                            detail="Prediction failed. See ml service logs.") from e


@router.post("/predict/coldstart", response_model=PredictionResponse)
async def predict_coldstart(
        payload: CoordinatesColdStartPredictionRequest,
        model_manager: ModelManagerDep,
        db: DbSessionDep,
):
    """
    Executes Zero-Cold-Start Remote Sensing & Microclimate ML Inference directly from GPS coordinates
    (Latitude, Longitude) without requiring pre-existing farmer database records.
    """
    logger.info("predict/coldstart request: %s", payload.model_dump())
    try:
        # Construct synthetic farmer request wrapper
        farmer_req = FarmerPredictionRequest(
            farmer_id=f"coldstart-{payload.latitude:.4f}-{payload.longitude:.4f}"
        )
        
        predictions_service = PredictionsService(
            model_manager=model_manager,
            data=farmer_req,
            db=db
        )

        # Inject coordinate data directly into mock farmer_data object
        mock_farmer_data = SimpleNamespace(
            user_id=farmer_req.farmer_id,
            farm_size=payload.farm_size,
            state=payload.state,
            latitude=payload.latitude,
            longitude=payload.longitude,
            crops=[payload.crop_type],
            yield_value=None,
            has_extension_access=None,
            household_max_education=None,
            shock_level=None,
            received_assistance=None,
            used_fertilizer=None,
            household_size=None,
            transport_cost=None,
            dependency_ratio=None,
            asset_score=None,
            postharvest_activity_score=None,
            digital_access_score=None,
            has_veterinary_access=None,
            market_access_score=None,
            received_credit=None,
            head_gender=None,
            crop_diversity_score=1
        )

        prediction = await predictions_service.predict_from_farmer_data(mock_farmer_data)
        logger.info("predict/coldstart response: %s", _summary(prediction))
        return prediction

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("predict/coldstart failed for (%s, %s)",
                         payload.latitude, payload.longitude)
        raise HTTPException(status_code=500,
                            detail="Prediction failed. See ml service logs.") from e
