import functools
from typing import Tuple, Dict, Any, List, Optional

from argotech.features.builder import FeatureStore
from argotech.serving.schemas.request import FarmerPredictionRequest
from argotech.models.registry import ModelManager
from argotech.models.fusion import CrossAttentionFusionLayer
from argotech.data.sentinel import sentinel_client
from argotech.serving.schemas.response import (
    PredictionResponse, 
    PredictionProbabilities,
    InferenceDetail,
    SpatiotemporalIndices,
    MicroclimateMetrics,
    CrossAttentionDetail
)
from fastapi.concurrency import run_in_threadpool
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
import requests
import pandas as pd
import numpy as np
import math

# Open-Meteo is a free public API with no SLA. This is the budget the call gets before the
# prediction falls back to its default microclimate metrics.
OPEN_METEO_TIMEOUT_SECONDS = 4


class PredictionsService:
    """
    Early Warning Machine Learning Inference & Risk Assessment Service.
    Integrates Spatiotemporal Sentinel-2 Spectral Indices (NDVI, NDWI, EVI),
    Open-Meteo Microclimate Lag Feature Engineering, Cross-Attention Multi-Modal Fusion Layer,
    and TreeSHAP Explainer.
    """
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
        self.cross_attention_layer = CrossAttentionFusionLayer(
            spatial_dim=4,
            temporal_dim=4,
            embed_dim=32,
            num_heads=4
        )

    async def _fetch_farmer_data(self):
        """
        Retrieves farmer profile, land specs, and crop metadata from PostgreSQL.
        """
        query = text("""
            SELECT 
                fp.user_id, fp.farm_size, fp.state, fp.latitude, fp.longitude, fp.crops,
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

    async def _fetch_microclimate_weather_data(self, lat: Optional[float], lon: Optional[float]) -> dict:
        """
        Fetches 7-day hourly Open-Meteo weather telemetry to construct microclimate lag features:
        - Consecutive hours where RH >= 85%
        - Optimal incubation hours (18°C <= T <= 24°C and RH >= 80%)
        - Volumetric soil moisture deficit over 72 hours
        - Rainfall anomaly
        """
        default_metrics = {
            "rainfall_anomaly": 0.0,
            "drought_risk": 0,
            "rh_85_consecutive_hrs": 0,
            "incubation_hours": 0,
            "soil_water_deficit_72h": 0.0,
            "hourly_sequence": np.zeros((168, 4)) # 7 days * 24 hours = 168 steps (temp, rh, soil_m, rain)
        }
        if lat is None or lon is None:
            return default_metrics

        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&hourly=temperature_2m,relative_humidity_2m,soil_moisture_0_to_7cm,rain"
                f"&past_days=7&forecast_days=1&timezone=auto"
            )
            # timeout is a requests kwarg, NOT a query parameter. Passed inside params= it was
            # appended to the Open-Meteo URL as ?timeout=4 and the call had no timeout at all —
            # a hung upstream would block this threadpool worker forever, and enough of them
            # would stop the service answering anything, /health included.
            res = await run_in_threadpool(
                functools.partial(requests.get, url, timeout=OPEN_METEO_TIMEOUT_SECONDS)
            )
            if res.status_code == 200:
                hourly = res.json().get("hourly", {})
                temps = hourly.get("temperature_2m", [])
                rhs = hourly.get("relative_humidity_2m", [])
                soil_m = hourly.get("soil_moisture_0_to_7cm", [])
                rains = hourly.get("rain", [])

                total_rain = sum(rains) if rains else 0.0
                drought_risk = 1 if total_rain < 5.0 else 0

                # Compute consecutive high humidity hours (RH >= 85%)
                max_consecutive_rh = 0
                curr_consecutive = 0
                incubation_hrs = 0

                # Build hourly sequence tensor for Cross-Attention
                steps = min(len(temps), len(rhs), len(soil_m), len(rains))
                seq = np.zeros((steps, 4))
                for idx in range(steps):
                    t = temps[idx] if temps[idx] is not None else 22.0
                    rh = rhs[idx] if rhs[idx] is not None else 65.0
                    sm = soil_m[idx] if soil_m[idx] is not None else 0.25
                    r = rains[idx] if rains[idx] is not None else 0.0

                    seq[idx] = [t, rh, sm, r]

                    if rh >= 85:
                        curr_consecutive += 1
                        max_consecutive_rh = max(max_consecutive_rh, curr_consecutive)
                    else:
                        curr_consecutive = 0

                    if (18.0 <= t <= 24.0) and (rh >= 80):
                        incubation_hrs += 1

                # Soil moisture deficit over 72 hours
                soil_deficit = 0.0
                if soil_m and len(soil_m) >= 72:
                    val_start = soil_m[-72] if soil_m[-72] is not None else 0.0
                    val_end = soil_m[-1] if soil_m[-1] is not None else 0.0
                    soil_deficit = max(0.0, val_start - val_end)

                # ponytail: `rainfall_anomaly` here is raw 8-day rainfall (0-60 mm), while the
                # training set builds it as `six_month_total - 450` (-450 to +400). Same column
                # name, incompatible distributions — the single worst train/serve skew in the
                # service. Not fixed in place because correcting the serving side alone shifts
                # the input distribution under an un-retrained model; fix it together with the
                # retrain (docs/model-design.md, P0-1).
                return {
                    "rainfall_anomaly": round(total_rain, 2),
                    "drought_risk": drought_risk,
                    "rh_85_consecutive_hrs": max_consecutive_rh,
                    "incubation_hours": incubation_hrs,
                    "soil_water_deficit_72h": round(soil_deficit, 4),
                    "hourly_sequence": seq if steps > 0 else np.zeros((168, 4))
                }

        except Exception as e:
            print(f"Microclimate Weather API query warning: {e}")

        return default_metrics

    async def _resolve_spatiotemporal_indices(self, lat: Optional[float], lon: Optional[float], state: Optional[str] = None) -> dict:
        """
        Real Sentinel-2 spectral indices (NDVI/NDWI/EVI) via the CDSE Statistical API when
        credentials are configured and a cloud-free scene exists, otherwise the synthetic model.
        Tagged with `source` ("sentinel-2" vs "modelled") so downstream/UI can tell them apart.
        The blocking Sentinel call is off-loaded to a threadpool so the event loop isn't blocked.
        """
        real = await run_in_threadpool(sentinel_client.fetch_indices, lat, lon)
        if real:
            ndvi, ndwi, evi = real["ndvi"], real["ndwi"], real["evi"]
            return {
                "ndvi": round(ndvi, 3),
                "ndwi": round(ndwi, 3),
                "evi": round(evi, 3),
                "canopy_stress_status": self._canopy_status(ndvi, ndwi),
                "source": "sentinel-2",
                "sensing_date": real.get("sensing_date"),
            }
        return self._synthetic_indices(lat, lon, state)

    def _synthetic_indices(self, lat: Optional[float], lon: Optional[float], state: Optional[str] = None) -> dict:
        """
        Deterministic fallback spectral indices when real Sentinel-2 data isn't available. Kept as a
        graceful degradation path so inference never hard-fails on a missing scene / credentials.
        """
        lat_val = float(lat) if lat is not None else 11.8333
        lon_val = float(lon) if lon is not None else 13.1500

        base_signal = math.sin(lat_val * 0.1) * math.cos(lon_val * 0.1)
        ndvi = round(max(0.20, min(0.85, 0.55 + base_signal * 0.2)), 3)
        ndwi = round(max(-0.15, min(0.50, 0.25 + base_signal * 0.15)), 3)
        evi = round(max(0.15, min(0.75, 0.45 + base_signal * 0.18)), 3)

        return {
            "ndvi": ndvi,
            "ndwi": ndwi,
            "evi": evi,
            "canopy_stress_status": self._canopy_status(ndvi, ndwi),
            "source": "modelled",
        }

    @staticmethod
    def _canopy_status(ndvi: float, ndwi: float) -> str:
        if ndwi < 0.05:
            return "High Water Stress"
        elif ndvi < 0.35:
            return "Vegetation Degradation"
        return "Healthy Canopy"

    @staticmethod
    def _parse_farm_size(val: Any) -> float:
        """
        Parses farm_size attribute: handles float/int directly or string like "12 Hectares" / "12.5 ha".
        """
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str) and val.strip():
            try:
                return float(val.strip().split(' ')[0])
            except (ValueError, IndexError):
                return 1.0
        return 1.0

    def _map_features(self, result, weather: dict, indices: dict) -> Tuple[dict, str]:
        state_map = {"Kaduna": 1, "Kano": 2, "Lagos": 3}
        zone_id = state_map.get(result.state, 0)
        
        land_size = self._parse_farm_size(result.farm_size)

        is_cold_start = (result.yield_value is None and result.shock_level is None)
        mode = "GEOSPATIAL_COLDSTART_REMOTE_SENSING" if is_cold_start else "HYBRID_SURVEY_REMOTE_SENSING"

        # In cold-start mode, derive yield & vulnerability proxies directly from Sentinel-2 & Open-Meteo
        if is_cold_start:
            ndvi_val = indices.get("ndvi", 0.5)
            estimated_yield = round(max(0.8, min(4.5, 1.5 + (ndvi_val - 0.4) * 3.5)), 2)
            estimated_shock = 2 if weather.get("drought_risk", 0) == 1 else 0
            asset_proxy = round(max(20.0, min(80.0, 45.0 + (indices.get("evi", 0.4) - 0.3) * 50.0)), 1)
        else:
            estimated_yield = float(result.yield_value) if result.yield_value is not None else 1.2
            estimated_shock = int(result.shock_level) if result.shock_level is not None else 0
            asset_proxy = float(result.asset_score) if result.asset_score is not None else 45.0

        mapped = {
            "yield_value": estimated_yield,
            "has_extension_access": 1 if result.has_extension_access else 0,
            "household_max_education": int(result.household_max_education) if result.household_max_education is not None else 1,
            "shock_level": estimated_shock,
            "received_assistance": 1 if result.received_assistance else 0,
            "used_fertilizer": 1 if result.used_fertilizer else 0,
            "land_size": land_size,
            "household_size": int(result.household_size) if result.household_size is not None else 4,
            "zone": zone_id,
            "transport_cost": float(result.transport_cost) if result.transport_cost is not None else 2500.0,
            "dependency_ratio": float(result.dependency_ratio) if result.dependency_ratio is not None else 0.8,
            "asset_score": asset_proxy,
            "postharvest_activity_score": float(result.postharvest_activity_score) if result.postharvest_activity_score is not None else 3.5,
            "crop_loss_risk_score": 0.0,
            "crop_diversity_score": float(result.crop_diversity_score) if result.crop_diversity_score is not None else 2.0,
            "digital_access_score": float(result.digital_access_score) if result.digital_access_score is not None else 50.0,
            "has_veterinary_access": 1 if result.has_veterinary_access else 0,
            "market_access_score": float(result.market_access_score) if result.market_access_score is not None else 55.0,
            "is_rural": 1,
            "rainfall_anomaly": weather["rainfall_anomaly"],
            "drought_risk": weather["drought_risk"],
            "cultivates_crops": 1,
            "received_credit": 1 if result.received_credit else 0,
            "head_gender": int(result.head_gender) if result.head_gender is not None else 0,
            # Spectral indices as direct model inputs — real Sentinel-2 values when available,
            # synthetic fallback otherwise (see _resolve_spatiotemporal_indices). Must stay in lockstep
            # with FEATURES and the training record schema.
            "ndvi": float(indices.get("ndvi", 0.5)),
            "ndwi": float(indices.get("ndwi", 0.2)),
            "evi": float(indices.get("evi", 0.4))
        }

        return mapped, mode

    def _generate_risk_drivers(self, payload_dict: dict, weather: dict, indices: dict) -> list:
        drivers = []
        if weather.get("rh_85_consecutive_hrs", 0) >= 12:
            drivers.append(f"Relative humidity > 85% for {weather['rh_85_consecutive_hrs']} consecutive hours")
        if weather.get("incubation_hours", 0) >= 15:
            drivers.append(f"Fungal incubation window active for {weather['incubation_hours']} hours (18–24°C)")
        if weather.get("soil_water_deficit_72h", 0) > 0.05:
            drivers.append(f"Soil water deficit drop of {weather['soil_water_deficit_72h']:.3f} m³/m³ over 72h")

        if indices["ndwi"] < 0.10:
            drivers.append(f"Leaf water stress detected (NDWI: {indices['ndwi']:.2f})")
        if indices["ndvi"] < 0.35:
            drivers.append(f"Low canopy greenness / biomass density (NDVI: {indices['ndvi']:.2f})")

        yield_val = payload_dict.get("yield_value", 1.5)
        if yield_val < 1.5:
            drivers.append("Low crop yield (< 1.5 tons/ha)")
        if not payload_dict["has_extension_access"]:
            drivers.append("No agricultural extension worker visits")
        if payload_dict["shock_level"] > 1:
            drivers.append("High vulnerability to climate/market shocks")
        if not payload_dict["received_credit"]:
            drivers.append("No access to agricultural credit or micro-financing")
        if not payload_dict["used_fertilizer"]:
            drivers.append("No modern fertilizer inputs applied")
        if weather["drought_risk"] == 1:
            drivers.append("Localized rainfall deficit / drought warning")

        if not drivers:
            drivers.append("Optimal growing conditions — no critical stress drivers detected")
            
        return drivers

    async def predict(self) -> PredictionResponse:
        farmer_data = await self._fetch_farmer_data()
        return await self.predict_from_farmer_data(farmer_data)

    async def predict_from_farmer_data(self, farmer_data) -> PredictionResponse:
        try:
            weather = await self._fetch_microclimate_weather_data(farmer_data.latitude, farmer_data.longitude)
            indices = await self._resolve_spatiotemporal_indices(farmer_data.latitude, farmer_data.longitude, farmer_data.state)
            payload_dict, inference_mode = self._map_features(farmer_data, weather, indices)


            #  Execute Cross-Attention Fusion
            spatial_vector = np.array([indices["ndvi"], indices["ndwi"], indices["evi"], payload_dict["land_size"]])
            temporal_seq = weather["hourly_sequence"]
            
            fused_representation, attn_weights = self.cross_attention_layer.forward(spatial_vector, temporal_seq)
            fusion_score = round(float(np.mean(fused_representation)), 4)
            peak_incubation_hour = int(np.argmax(attn_weights[0])) if attn_weights is not None and attn_weights.shape[1] > 0 else 0

            # Build DataFrame for ML model
            df = self.features.build_features(payload_dict)

            # 3. Run Scikit-Learn / XGBoost Prediction
            prediction = await run_in_threadpool(
                self.model_manager.predict,
                df,
                self.data.model_name,
                self.data.model_alias
            )

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

            # Dynamic Risk Drivers
            risk_drivers = self._generate_risk_drivers(payload_dict, weather, indices)

            crop_list = farmer_data.crops if isinstance(farmer_data.crops, list) else (farmer_data.crops.split(',') if farmer_data.crops else ['Maize'])
            crop_name = crop_list[0].strip() if crop_list else "Maize"

            if pred_val >= 2 or risk_score >= 70:
                risk_level = "CRITICAL"
                disease_type = f"Late Blight / Fungal Leaf Rust in {crop_name}"
                recommended_action = "Apply protective copper-based fungicide spray within 24–48 hours and dispatch extension agent for immediate field inspection."
            elif pred_val == 1 or risk_score >= 40:
                risk_level = "ELEVATED"
                disease_type = f"Early Leaf Water & Moisture Stress in {crop_name}"
                recommended_action = "Initiate supplemental drip irrigation and monitor canopy NDWI water stress index closely over 48 hours."
            else:
                risk_level = "NORMAL"
                disease_type = "No Disease Outbreak Detected"
                recommended_action = "Maintain standard agronomic fertilizer & weeding schedule."

            inference_detail = InferenceDetail(
                risk_level=risk_level,
                disease_type=disease_type,
                probability=round(float(proba[0][pred_val]), 3) if proba is not None else 0.85,
                primary_drivers=risk_drivers[:4]
            )

            spatiotemporal = SpatiotemporalIndices(
                ndvi=indices["ndvi"],
                ndwi=indices["ndwi"],
                evi=indices["evi"],
                canopy_stress_status=indices["canopy_stress_status"],
                source=indices.get("source", "modelled")
            )

            microclimate = MicroclimateMetrics(
                rh_85_consecutive_hrs=weather["rh_85_consecutive_hrs"],
                incubation_hours=weather["incubation_hours"],
                soil_water_deficit_72h=weather["soil_water_deficit_72h"],
                rainfall_anomaly=weather["rainfall_anomaly"]
            )

            cross_attention = CrossAttentionDetail(
                fusion_score=fusion_score,
                peak_incubation_hour=peak_incubation_hour,
                fusion_status="Active Spatiotemporal Cross-Attention Align"
            )

            return PredictionResponse(
                field_id=self.data.farmer_id,
                crop_type=crop_name,
                phenology_stage="Vegetative / Flowering",
                prediction=pred_val,
                priority_label=priority_label,
                risk_score_percent=risk_score,
                probabilities=probabilities,
                top_risk_factors=risk_drivers,
                inference=inference_detail,
                spatiotemporal_indices=spatiotemporal,
                microclimate_metrics=microclimate,
                cross_attention_fusion=cross_attention,
                recommended_action=recommended_action
            )

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Prediction pipeline failed: {str(e)}")
