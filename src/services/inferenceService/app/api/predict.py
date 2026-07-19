from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.services.inferenceService.app.schemas.request import PredictionRequest, FarmerPredictionRequest
from src.services.inferenceService.app.schemas.response import PredictionResponse
from src.services.inferenceService.app.dependencies import get_model_manager, get_feature_store
from src.services.inferenceService.app.core.model_manager import ModelManager
from src.services.inferenceService.app.core.database import get_db
from src.services.inferenceService.app.service.PredictionsService import PredictionsService

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
        payload: PredictionRequest,
        model_manager: ModelManager = Depends(get_model_manager),
        features=Depends(get_feature_store)
):
    # This remains for legacy compatibility if called with full features
    # (though modified to match new PredictionResponse format)
    # We will raise an error or run static mapping if needed
    raise HTTPException(status_code=400, detail="Use /predict/farmer to run prediction via PostgreSQL mapping.")


@router.post("/predict/farmer", response_model=PredictionResponse)
async def predict_farmer(
        payload: FarmerPredictionRequest,
        model_manager: ModelManager = Depends(get_model_manager),
        features=Depends(get_feature_store),
        db: Session = Depends(get_db)
):
    try:
        predictions_service = PredictionsService(
            features=features,
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
