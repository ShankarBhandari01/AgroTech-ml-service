"""Prediction pipeline: measurements in, decomposed risk out.

The feature row is built by `features.agronomic.build` — the same function the training set was
built with, over the same daily variables. That is the structural guarantee against train/serve
skew, replacing the previous pipeline where `rainfall_anomaly` meant a six-month archive anomaly at
training and eight-day raw rainfall at serving.

The learned model does not produce the risk score on its own. It contributes the *vegetation hazard*
term to the `domain.risk` composition, alongside the physics-derived drought, disease and heat
terms. See docs/model-design.md §4.5.
"""

from __future__ import annotations

import functools
import uuid
from typing import Any, Optional

import pandas as pd
import requests
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text
from sqlalchemy.orm import Session

from argotech.data import meteo, store
from argotech.data.sentinel import sentinel_client
from argotech.domain import agronomy, indices, risk
from argotech.features.agronomic import WINDOW_DAYS, build, satellite_block
from argotech.models.registry import ModelManager
from argotech.serving.schemas.request import FarmerPredictionRequest
from argotech.serving.schemas.response import (
    CropHealthDetail,
    HazardDetail,
    InferenceDetail,
    MicroclimateMetrics,
    PredictionProbabilities,
    PredictionResponse,
    RiskAssessmentDetail,
    SpatiotemporalIndices,
    VulnerabilityDetail,
)

OPEN_METEO_TIMEOUT_SECONDS = 6

# ponytail: farm-gate prices as a static per-crop table in USD/tonne. Exposure needs a price to be
# expressed in currency at all; wire it to a market-price feed when one exists.
FARMGATE_PRICE_USD_PER_T = {"maize": 250.0, "sorghum": 230.0, "rice": 420.0, "cassava": 120.0}
DEFAULT_PRICE_USD_PER_T = 250.0


def _leaf_wetness(lat: float, lon: float) -> list:
    """Per-day (mean temperature during the wet period, wet hours) from hourly Open-Meteo.

    The disease model needs sub-daily resolution — leaf wetness duration is the input, and a daily
    mean humidity cannot express it. One hourly call, separate from the daily call that feeds the
    model features.
    """
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=temperature_2m,relative_humidity_2m&past_days=14&forecast_days=1&timezone=UTC")
    try:
        res = requests.get(url, timeout=OPEN_METEO_TIMEOUT_SECONDS)
        res.raise_for_status()
        hourly = res.json().get("hourly", {})
        temps, rhs = hourly.get("temperature_2m", []), hourly.get("relative_humidity_2m", [])
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] hourly weather unavailable: {e}")
        return []

    days = []
    for start in range(0, len(temps) - 23, 24):
        t_day = [t for t in temps[start:start + 24] if t is not None]
        wet = [t for t, rh in zip(temps[start:start + 24], rhs[start:start + 24])
               if t is not None and rh is not None and rh >= 90]
        if wet:
            days.append((sum(wet) / len(wet), len(wet)))
        elif t_day:
            days.append((sum(t_day) / len(t_day), 0))
    return days


async def gather_upstream(lat: float, lon: float, crop: str) -> tuple[dict, dict]:
    """Every upstream call for one field, in one place: ERA5 daily, Sentinel-2 history, hourly leaf
    wetness, and the 4-year rainfall climatology.

    Returns `(feature_row, context)`. The nightly precompute job calls this and stores the result;
    a request calls it only when no fresh stored row exists. Keeping one implementation is what
    guarantees a precomputed prediction and a live one are the same computation.
    """
    payload = await run_in_threadpool(functools.partial(meteo.fetch_recent, lat, lon, 92))
    history = await run_in_threadpool(sentinel_client.fetch_history, lat, lon, 365)
    wet_days = await run_in_threadpool(_leaf_wetness, lat, lon)

    if not payload:
        raise HTTPException(503, "Weather upstream unavailable; cannot build features.")
    daily = meteo.daily_frame(payload)
    if len(daily["time"]) < WINDOW_DAYS:
        raise HTTPException(503, "Insufficient weather history for the 90-day feature window.")

    # ---- canopy state ----
    crop_health = None
    if history:
        latest = history[-1]
        series = [o["ndvi"] for o in history]
        mean = sum(series) / len(series)
        sat = satellite_block(latest, series[:-1], series[:-1])
        crop_health = indices.crop_health_index(
            current_ndvi=latest["ndvi"], ndmi_value=latest["ndwi"],
            ndvi_min=min(series), ndvi_max=max(series),
            peer_mean=mean,
            peer_std=(sum((v - mean) ** 2 for v in series) / len(series)) ** 0.5,
        ).as_dict()
        index_source, sensing_date = "sentinel-2", latest["sensing_date"]
    else:
        # No cloud-free scene. Rather than invent indices, mark the canopy block unavailable and let
        # the physical hazards carry the assessment — they need no satellite at all.
        sat = {"ndvi": 0.0, "ndmi": 0.0, "evi": 0.0, "vci": 50.0, "ndvi_z_peer": 0.0}
        index_source, sensing_date = "unavailable", None

    # ---- feature row (identical builder to training) ----
    clim = await run_in_threadpool(meteo.climatological_rain_30, lat, lon, daily["time"][-1][5:])
    site = {"latitude": lat, "longitude": lon,
            "elevation": payload.get("elevation", 0.0),
            "clim_rain_30": clim if clim is not None else sum(daily["precipitation_sum"][-30:])}
    row = build(daily, sat, site, crop)

    dsv_total, spray_due = agronomy.accumulate_dsv(
        [agronomy.daily_severity_value(t, h) for t, h in wet_days])

    return row, {
        "sat": sat,
        "index_source": index_source,
        "sensing_date": sensing_date,
        "crop_health": crop_health,
        "dsv_total": dsv_total,
        "spray_due": spray_due,
    }


class PredictionsService:
    def __init__(self, model_manager: ModelManager, data: FarmerPredictionRequest, db: Session):
        self.model_manager = model_manager
        self.data = data
        self.db = db

    # ------------------------------------------------------------------ data

    async def _fetch_farmer_data(self):
        # farmer_profiles.user_id is a uuid column, so a malformed id reaches Postgres as a cast
        # error rather than an empty result — a 500 with the whole SQL statement in the response
        # body. Reject it here as the not-found it actually is.
        try:
            uuid.UUID(str(self.data.farmer_id))
        except (ValueError, AttributeError, TypeError):
            raise HTTPException(404, f"Farmer '{self.data.farmer_id}' not found in database.")

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
            raise HTTPException(404, f"Farmer '{self.data.farmer_id}' not found in database.")
        return result

    # ------------------------------------------------------------------ inference

    async def predict(self) -> PredictionResponse:
        return await self.predict_from_farmer_data(await self._fetch_farmer_data())

    async def predict_from_farmer_data(self, farmer) -> PredictionResponse:
        lat, lon = farmer.latitude, farmer.longitude
        if lat is None or lon is None:
            raise HTTPException(422, "Prediction requires latitude and longitude.")
        lat, lon = float(lat), float(lon)
        crop = self._crop_name(farmer)
        field_id = self.data.farmer_id

        # Precomputed row when the nightly job has produced a fresh one, live upstream otherwise.
        # The fallback is deliberately kept: a new field registered this morning still gets a
        # prediction today, it just pays the latency once.
        cached = store.read_latest_features(self.db, field_id) if self.db is not None else None
        if cached:
            row, ctx, feature_source = cached["features"], cached["context"], "precomputed"
        else:
            row, ctx = await gather_upstream(lat, lon, crop)
            feature_source = "live"

        sat = ctx["sat"]
        index_source = ctx["index_source"]
        crop_health = ctx.get("crop_health")
        dsv_total, spray_due = ctx["dsv_total"], ctx["spray_due"]

        # ---- learned vegetation hazard ------------------------------------
        vegetation_hazard, probabilities = 0.0, None
        model_version = "none"
        if index_source == "sentinel-2":
            model, columns, model_version = self.model_manager.agronomic_model()
            proba = await run_in_threadpool(model.predict_proba, pd.DataFrame([row])[columns])
            p = proba[0]
            # Expected severity on 0-1: the ranking score, and the term fed into the composition.
            vegetation_hazard = float(p[1] * 0.5 + p[2] * 1.0)
            probabilities = PredictionProbabilities(low=round(float(p[0]), 3),
                                                    medium=round(float(p[1]), 3),
                                                    high=round(float(p[2]), 3))

        # ---- composition ---------------------------------------------------
        hazard = risk.assess_hazard(
            water_satisfaction=row["water_satisfaction_30"],
            dry_spell_days=row["dry_spell_30"],
            cumulative_dsv=dsv_total,
            heat_days=row["heat_stress_days"],
            vegetation=vegetation_hazard,
        )
        exposure = risk.assess_exposure(
            area_ha=self._parse_farm_size(farmer.farm_size),
            expected_yield_t_ha=self._expected_yield(farmer, sat["ndvi"]),
            price_per_t=FARMGATE_PRICE_USD_PER_T.get(crop.lower(), DEFAULT_PRICE_USD_PER_T),
        )
        vulnerability = risk.assess_vulnerability(self._coping_signals(farmer))
        assessment = risk.assess_risk(hazard, exposure, vulnerability)

        # ---- audit row -----------------------------------------------------
        # Written before the response is returned so the id can be handed back: the agent app links
        # its outcome report to this prediction, which is what turns advisories into training data.
        prediction_id = None
        if self.db is not None:
            prediction_id = store.write_prediction(
                self.db, field_id, model_version, feature_source, row, assessment,
                probabilities.model_dump() if probabilities else None)

        # ---- response ------------------------------------------------------
        severity_to_class = {"NORMAL": 0, "WATCH": 0, "ELEVATED": 1, "CRITICAL": 2}
        pred_val = severity_to_class[assessment.severity]
        stage = agronomy.phenology_stage(row["gdd_since_onset"], crop)
        drivers = self._drivers(row, hazard, vulnerability, dsv_total, spray_due, crop_health)

        return PredictionResponse(
            field_id=field_id,
            prediction_id=prediction_id,
            model_version=model_version,
            feature_source=feature_source,
            features_computed_at=cached["computed_at"] if cached else None,
            crop_type=crop,
            phenology_stage=stage,
            prediction=pred_val,
            priority_label={0: "Low Priority", 1: "Medium Priority", 2: "High Priority"}[pred_val],
            risk_score_percent=assessment.risk_score,
            probabilities=probabilities or PredictionProbabilities(low=1.0, medium=0.0, high=0.0),
            top_risk_factors=drivers,
            inference=InferenceDetail(
                risk_level=assessment.severity,
                dominant_hazard=hazard.dominant,
                probability=round(hazard.combined, 3),
                primary_drivers=drivers[:4],
                model_contributed=index_source == "sentinel-2",
            ),
            risk_assessment=RiskAssessmentDetail(
                risk_score=assessment.risk_score,
                expected_loss_usd=assessment.expected_loss,
                value_at_risk_usd=exposure.value_at_risk,
                hazard=HazardDetail(**hazard.as_dict()),
                vulnerability=VulnerabilityDetail(**vulnerability.as_dict()),
            ),
            crop_health=CropHealthDetail(**crop_health) if crop_health else None,
            spatiotemporal_indices=SpatiotemporalIndices(
                ndvi=round(sat["ndvi"], 3), ndwi=round(sat["ndmi"], 3), evi=round(sat["evi"], 3),
                vci=round(sat["vci"], 1),
                canopy_stress_status=crop_health["status"] if crop_health else "Unavailable",
                source=index_source, sensing_date=ctx.get("sensing_date"),
            ),
            microclimate_metrics=MicroclimateMetrics(
                water_satisfaction_30d=row["water_satisfaction_30"],
                water_deficit_30d_mm=row["water_deficit_30"],
                longest_dry_spell_days=row["dry_spell_30"],
                rainfall_anomaly_30d_mm=row["rain_anomaly_30"],
                heat_stress_days=row["heat_stress_days"],
                cumulative_dsv=dsv_total,
                spray_threshold_reached=spray_due,
                gdd_since_onset=row["gdd_since_onset"],
            ),
            recommended_action=self._action(assessment, hazard, spray_due, stage, crop),
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _crop_name(farmer) -> str:
        crops = farmer.crops
        if isinstance(crops, list) and crops:
            return str(crops[0]).strip()
        if isinstance(crops, str) and crops.strip():
            return crops.split(",")[0].strip()
        return "Maize"

    @staticmethod
    def _parse_farm_size(val: Any) -> float:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str) and val.strip():
            try:
                return float(val.strip().split(" ")[0])
            except (ValueError, IndexError):
                return 1.0
        return 1.0

    @staticmethod
    def _expected_yield(farmer, ndvi: float) -> float:
        """Reported yield when the survey has one; otherwise an NDVI-anchored estimate.

        Explicitly a placeholder for the yield-anomaly regression in docs/model-design.md §4.4,
        which needs a season of harvest records that do not exist yet.
        """
        reported = getattr(farmer, "yield_value", None)
        # A zero yield is an unfilled survey field, not a farm that harvests nothing. Taken
        # literally it makes value_at_risk exactly 0, so a CRITICAL farm sorts to the *bottom* of a
        # queue ranked by expected loss — the ranking inverts precisely for the farms that matter.
        if reported is not None and float(reported) > 0:
            return float(reported)
        return round(max(0.5, min(5.0, 1.5 + (ndvi - 0.4) * 3.5)), 2)

    @staticmethod
    def _coping_signals(farmer) -> dict:
        """Vulnerability inputs. `head_gender` and `household_max_education` are deliberately absent
        — `risk.assess_vulnerability` rejects them (docs/model-design.md, P1-4)."""
        def num(attr, default=0.0):
            v = getattr(farmer, attr, None)
            return float(v) if v is not None else default

        return {
            "has_irrigation": 0.0,   # not captured by the current schema; add to farmers_ml_profiles
            "has_extension_access": 1.0 if getattr(farmer, "has_extension_access", None) else 0.0,
            "received_credit": 1.0 if getattr(farmer, "received_credit", None) else 0.0,
            "used_fertilizer": 1.0 if getattr(farmer, "used_fertilizer", None) else 0.0,
            "crop_diversity": min(1.0, num("crop_diversity_score") / 4.0),
            "asset_score": min(1.0, num("asset_score", 45.0) / 100.0),
            "market_access": min(1.0, num("market_access_score", 50.0) / 100.0),
        }

    @staticmethod
    def _drivers(row, hazard, vulnerability, dsv_total, spray_due, crop_health) -> list[str]:
        out = []
        if hazard.drought > 0.2:
            out.append(f"Water satisfaction {row['water_satisfaction_30']:.0%} over 30 days "
                       f"({row['water_deficit_30']:.0f} mm unmet demand)")
        if row["dry_spell_30"] >= 8:
            out.append(f"Longest dry spell {row['dry_spell_30']} days in the last 30")
        if spray_due:
            out.append(f"Blight severity values at {dsv_total}, spray threshold (18) reached")
        elif dsv_total >= 9:
            out.append(f"Blight severity values accumulating ({dsv_total}/18)")
        if row["heat_stress_days"]:
            out.append(f"{row['heat_stress_days']} days above 35 °C during flowering")
        if hazard.vegetation > 0.35:
            out.append(f"Model projects canopy stress relative to peers within 30 days "
                       f"(hazard {hazard.vegetation:.2f})")
        if crop_health and crop_health["score"] < 45:
            out.append(f"Crop Health Index {crop_health['score']:.0f}/100 — {crop_health['status']}")
        if row["rain_anomaly_30"] < -20:
            out.append(f"Rainfall {abs(row['rain_anomaly_30']):.0f} mm below the site's normal "
                       f"for this time of year")
        for gap in vulnerability.gaps[:2]:
            out.append(f"Limited coping capacity: {gap.replace('_', ' ')}")
        return out or ["No critical stress drivers detected"]

    @staticmethod
    def _action(assessment, hazard, spray_due, stage, crop) -> str:
        if spray_due:
            return ("Apply a protective fungicide within 24-48 h: accumulated blight severity has "
                    "reached the spray threshold, and dispatch an agent to confirm the diagnosis.")
        if hazard.dominant == "drought" and hazard.drought > 0.4:
            return (f"Prioritise supplemental irrigation — {crop} at {stage} is in water deficit. "
                    "Advise mulching and, if the deficit persists, staggered replanting.")
        if hazard.dominant == "heat":
            return ("Heat stress during flowering: advise irrigation timed to early morning to "
                    "reduce canopy temperature; expect pollination loss and plan for it.")
        if hazard.dominant == "disease":
            return (f"Blight severity is accumulating on {crop} at {stage}. Scout the lower canopy "
                    "for lesions now and have fungicide staged before the threshold is reached.")
        if hazard.dominant == "vegetation":
            return ("Canopy is projected to fall behind neighbouring fields. Schedule a scouting "
                    "visit to identify the cause before it shows in yield.")
        if assessment.severity in ("ELEVATED", "WATCH"):
            return "Increase monitoring frequency; no single dominant hazard yet."
        return "Maintain the standard fertiliser and weeding schedule."
