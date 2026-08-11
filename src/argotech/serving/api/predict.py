from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from types import SimpleNamespace

from argotech.serving.schemas.request import (
    FarmerPredictionRequest, 
    CoordinatesColdStartPredictionRequest
)
from argotech.serving.schemas.response import PredictionResponse
from argotech.serving.deps import get_model_manager
from argotech.models.registry import ModelManager
from argotech.data.db import get_db
from argotech.serving.pipeline import PredictionsService

router = APIRouter()


@router.post("/predict/farmer", response_model=PredictionResponse)
async def predict_farmer(
        payload: FarmerPredictionRequest,
        model_manager: ModelManager = Depends(get_model_manager),
        db: Session = Depends(get_db)
):
    try:
        predictions_service = PredictionsService(
            model_manager=model_manager,
            data=payload,
            db=db
        )
        prediction = await predictions_service.predict()
        return prediction

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/coldstart", response_model=PredictionResponse)
async def predict_coldstart(
        payload: CoordinatesColdStartPredictionRequest,
        model_manager: ModelManager = Depends(get_model_manager),
        db: Session = Depends(get_db)
):
    """
    Executes Zero-Cold-Start Remote Sensing & Microclimate ML Inference directly from GPS coordinates
    (Latitude, Longitude) without requiring pre-existing farmer database records.
    """
    try:
        # Construct synthetic farmer request wrapper
        farmer_req = FarmerPredictionRequest(
            farmer_id=f"coldstart-{payload.latitude:.4f}-{payload.longitude:.4f}",
            model_name=payload.model_name,
            model_alias=payload.model_alias
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
        return prediction

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
