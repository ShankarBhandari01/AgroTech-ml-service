from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from src.services.inferenceService.app.schemas.request import PredictionRequest
from src.services.inferenceService.app.schemas.response import PredictionResponse
from src.services.inferenceService.app.dependencies import get_model_manager, get_feature_store
from src.services.inferenceService.app.core.model_manager import ModelManager

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
        payload: PredictionRequest,
        model_manager: ModelManager = Depends(get_model_manager),
        features=Depends(get_feature_store)
):
    try:
        df = features.build_features(payload.model_dump())

        prediction = await run_in_threadpool(
            model_manager.predict,
            df
        )

        # proba = await run_in_threadpool(
        #     model_manager.predict_proba,
        #     df
        # )
        #
        # pred = int(prediction[0])
        # risk_score = round(float(proba[0][2]) * 100, 1)

        priority_map = {
            0: "Low Priority",
            1: "Medium Priority",
            2: "High Priority"
        }

        return PredictionResponse(
            prediction=prediction,
            priority_label=priority_map[1],
            risk_score_percent=10
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
