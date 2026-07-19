from src.services.inferenceService.app.feature_store.store import FeatureStore
from src.services.inferenceService.app.schemas.request import FarmerPredictionRequest
from src.services.inferenceService.app.core.model_manager import ModelManager
from src.services.inferenceService.app.schemas.response import PredictionResponse, PredictionProbabilities
from fastapi.concurrency import run_in_threadpool
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
import requests
import pandas as pd


class PredictionsService:
    def __init__(
        self,
        features: FeatureStore,
        model_manager: ModelManager,
        data: FarmerPredictionRequest,
        db: Session
    ):
        self.features = features
        self.model_manager = model_manager
        self.data = data
        self.db = db

    async def _fetch_farmer_data(self):
        query = text("""
            SELECT 
                fp.user_id, fp.farm_size, fp.state, fp.latitude, fp.longitude,
                fmp.yield_value, fmp.has_extension_access, fmp.household_max_education,
                fmp.shock_level, fmp.received_assistance, fmp.used_fertilizer,
                fmp.household_size, fmp.transport_cost, fmp.dependency_ratio,
                fmp.asset_score, fmp.postharvest_activity_score, fmp.digital_access_score,
                fmp.has_veterinary_access, fmp.market_access_score, fmp.received_credit,
                fmp.head_gender,
                (SELECT COUNT(*) FROM farmers_crops WHERE farmer_id = fp.user_id) as crop_diversity_score
            FROM farmer_profiles fp
            LEFT JOIN farmers_ml_profiles fmp ON fmp.farmer_id = fp.user_id
            WHERE fp.user_id = :farmer_id
        """)
        
        result = await run_in_threadpool(
            self.db.execute(query, {"farmer_id": self.data.farmer_id}).fetchone
        )
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Farmer '{self.data.farmer_id}' not found in database.")
        
        return result

    async def _fetch_weather_data(self, lat: float, lon: float) -> dict:
        weather = {"rainfall_anomaly": 0.0, "drought_risk": 0}
        if lat is None or lon is None:
            return weather

        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=rain_sum&timezone=auto"
            res = await run_in_threadpool(requests.get, url, {"timeout": 3})
            if res.status_code == 200:
                daily_rain = res.json().get("daily", {}).get("rain_sum", [])
                total_rain = sum(daily_rain) if daily_rain else 0.0
                weather["drought_risk"] = 1 if total_rain < 5.0 else 0
        except Exception as e:
            print(f"Weather API request failed: {e}")
            
        return weather

    def _map_features(self, result, weather: dict) -> dict:
        state_map = {"Kaduna": 1, "Kano": 2, "Lagos": 3}
        zone_id = state_map.get(result.state, 0)
        
        try:
            land_size = float(result.farm_size) if result.farm_size else 1.0
        except ValueError:
            land_size = 1.0

        return {
            "yield_value": float(result.yield_value) if result.yield_value is not None else 1.2,
            "has_extension_access": 1 if result.has_extension_access else 0,
            "household_max_education": int(result.household_max_education) if result.household_max_education is not None else 0,
            "shock_level": int(result.shock_level) if result.shock_level is not None else 0,
            "received_assistance": 1 if result.received_assistance else 0,
            "used_fertilizer": 1 if result.used_fertilizer else 0,
            "land_size": land_size,
            "household_size": int(result.household_size) if result.household_size is not None else 1,
            "zone": zone_id,
            "transport_cost": float(result.transport_cost) if result.transport_cost is not None else 0.0,
            "dependency_ratio": float(result.dependency_ratio) if result.dependency_ratio is not None else 0.0,
            "asset_score": float(result.asset_score) if result.asset_score is not None else 0.0,
            "postharvest_activity_score": float(result.postharvest_activity_score) if result.postharvest_activity_score is not None else 0.0,
            "crop_loss_risk_score": 0.0,
            "crop_diversity_score": float(result.crop_diversity_score) if result.crop_diversity_score is not None else 0.0,
            "digital_access_score": float(result.digital_access_score) if result.digital_access_score is not None else 0.0,
            "has_veterinary_access": 1 if result.has_veterinary_access else 0,
            "market_access_score": float(result.market_access_score) if result.market_access_score is not None else 0.0,
            "is_rural": 1,
            "rainfall_anomaly": weather["rainfall_anomaly"],
            "drought_risk": weather["drought_risk"],
            "cultivates_crops": 1 if (result.crop_diversity_score and result.crop_diversity_score > 0) else 0,
            "received_credit": 1 if result.received_credit else 0,
            "head_gender": int(result.head_gender) if result.head_gender is not None else 0
        }

    def _generate_risk_factors(self, payload_dict: dict, weather: dict) -> list:
        risk_factors = []
        yield_val = payload_dict.get("yield_value", payload_dict.get("yield", 1.5))
        if yield_val < 1.5:
            risk_factors.append("Low crop yield (< 1.5 tons/ha)")
        if not payload_dict["has_extension_access"]:
            risk_factors.append("No agricultural extension worker visits")
        if payload_dict["shock_level"] > 1:
            risk_factors.append("High vulnerability to market/weather shocks")
        if not payload_dict["received_credit"]:
            risk_factors.append("No agricultural credit/financing access")
        if not payload_dict["used_fertilizer"]:
            risk_factors.append("No fertilizer or modern inputs used")
        if weather["drought_risk"] == 1:
            risk_factors.append("Localized rainfall deficit / drought warning")

        if not risk_factors:
            risk_factors.append("None detected")
            
        return risk_factors

    async def predict(self) -> PredictionResponse:
        try:
            result = await self._fetch_farmer_data()
            weather = await self._fetch_weather_data(result.latitude, result.longitude)
            payload_dict = self._map_features(result, weather)

            # Generate DataFrame for prediction
            df = self.features.build_features(payload_dict)

            # Run ML Model Prediction
            prediction = await run_in_threadpool(
                self.model_manager.predict,
                df,
                self.data.model_name,
                self.data.model_alias
            )

            # Extract probabilities
            proba = await run_in_threadpool(
                self.model_manager.predict_proba,
                df,
                self.data.model_name,
                self.data.model_alias
            )

            pred_val = int(prediction[0])
            risk_score = round(float(proba[0][2]) * 100, 1) if (proba is not None and len(proba[0]) > 2) else 10.0
            
            probabilities = PredictionProbabilities(
                low=round(float(proba[0][0]), 3) if proba is not None else 0.9,
                medium=round(float(proba[0][1]), 3) if proba is not None else 0.08,
                high=round(float(proba[0][2]), 3) if proba is not None else 0.02
            )

            priority_map = {0: "Low Priority", 1: "Medium Priority", 2: "High Priority"}
            priority_label = priority_map.get(pred_val, "Low Priority")

            risk_factors = self._generate_risk_factors(payload_dict, weather)

            return PredictionResponse(
                prediction=pred_val,
                priority_label=priority_label,
                risk_score_percent=risk_score,
                probabilities=probabilities,
                top_risk_factors=risk_factors
            )

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Prediction pipeline failed: {str(e)}")
