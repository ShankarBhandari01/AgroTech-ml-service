"""Task 1 of docs/superpowers/plans/2026-08-29-cut-backend-schema-coupling.md.

`/predict/farmer` now accepts an optional `farmer` payload and prefers it over reading the Kotlin
backend's tables, but the database read must keep working unchanged — the deployed backend sends
`{farmer_id}` alone and nothing else, so a regression here takes down production.

Three things are proven here, each because a green suite alone does not prove them:

1. Given the same farmer, the payload path and the database path produce **identical**
   predictions — including when the payload omits fields the database would have returned as
   NULL. That equality is the entire safety argument for eventually deleting the database read.
2. `domain/risk.py`'s protected-attribute guard fires on a payload-sourced farmer object, not only
   a database Row (`test_domain.py` already covers the Row case).
3. The legacy `{farmer_id}`-only request — the deployed contract — still works unchanged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient

from argotech.data import backend_schema, meteo
from argotech.data.db import get_db
from argotech.data.sentinel import sentinel_client
from argotech.domain import risk
from argotech.serving import pipeline
from argotech.serving.main import app
from argotech.serving.pipeline import PredictionsService
from argotech.serving.schemas.request import FarmerPayload

FIXTURES = Path(__file__).parent / "fixtures"
LEAF_WETNESS_DAYS = [(20.0, 12)] * 14
FARMER_ID = "11111111-1111-1111-1111-111111111111"

# A database row where every `farmers_ml_profiles` column is NULL (no survey for this farmer) and
# `farm_size`/`crops` are NULL too (farm registered, plot detail never entered) — the shape
# `data/backend_schema.py`'s LEFT JOINs actually produce for an incomplete record. `crop_diversity_
# score` is the one column that is never SQL NULL (it is a COUNT, floor 0), kept as a literal 0 to
# prove a real DB zero and an omitted-payload None collapse to the same behaviour downstream.
DB_ROW_WITH_NULLS = dict(
    farm_size=None, state="Kaduna", latitude=10.8, longitude=7.9, crops=None,
    yield_value=None, has_extension_access=None, household_max_education=None, shock_level=None,
    received_assistance=None, used_fertilizer=None, household_size=None, transport_cost=None,
    dependency_ratio=None, asset_score=None, postharvest_activity_score=None,
    digital_access_score=None, has_veterinary_access=None, market_access_score=None,
    received_credit=None, head_gender=None, crop_diversity_score=0,
)

# The same farmer over the wire: only what a payload is documented to require (coordinates),
# every other field omitted so `FarmerPayload`'s defaults do the work.
PAYLOAD_SPARSE = {"latitude": 10.8, "longitude": 7.9}

# A fully-specified farmer, used for the plain legacy-path smoke test.
DB_ROW_POPULATED = dict(
    farm_size=2.0, state="Kaduna", latitude=10.8, longitude=7.9, crops=["Maize"],
    yield_value=3.2, has_extension_access=True, household_max_education=2, shock_level=1,
    received_assistance=False, used_fertilizer=True, household_size=6, transport_cost=15.0,
    dependency_ratio=0.5, asset_score=60.0, postharvest_activity_score=40.0,
    digital_access_score=30.0, has_veterinary_access=False, market_access_score=55.0,
    received_credit=True, head_gender="F", crop_diversity_score=3,
)


@pytest.fixture
def client(monkeypatch):
    """Same substitution as `test_e2e_predict.py`'s fixture: real app/routing/features/model, fake
    network and database."""
    weather = json.loads((FIXTURES / "meteo_recent.json").read_text())
    history = json.loads((FIXTURES / "sentinel_history.json").read_text())
    sar = json.loads((FIXTURES / "sentinel_sar_history.json").read_text())

    monkeypatch.setattr(meteo, "fetch_recent", lambda *a, **k: weather)
    monkeypatch.setattr(meteo, "climatological_rain_30", lambda *a, **k: 95.0)
    monkeypatch.setattr(sentinel_client, "fetch_history", lambda *a, **k: history)
    monkeypatch.setattr(sentinel_client, "fetch_sar_history", lambda *a, **k: sar)
    monkeypatch.setattr(pipeline, "_leaf_wetness", lambda *a, **k: list(LEAF_WETNESS_DAYS))

    def _no_network(*a, **k):
        raise AssertionError(f"live HTTP call from a hermetic test: {a[:1]}")
    for verb in ("get", "post", "put", "request"):
        monkeypatch.setattr(requests, verb, _no_network)

    app.dependency_overrides[get_db] = lambda: None
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_the_legacy_farmer_id_only_request_still_works(client, monkeypatch):
    """The deployed contract: `{farmer_id}` and nothing else. If this breaks, production breaks."""
    monkeypatch.setattr(backend_schema, "fetch_farmer_features",
                        lambda db, fid: SimpleNamespace(**DB_ROW_POPULATED))
    resp = client.post("/predict/farmer", json={"farmer_id": FARMER_ID})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["prediction"] in (0, 1, 2)
    assert 0.0 <= body["risk_score_percent"] <= 100.0


def test_payload_and_database_paths_agree_on_nulls_and_omissions(client, monkeypatch, caplog):
    """The money test. A payload omitting a field must not produce a different prediction from a
    database NULL for that column — otherwise the same farmer gets a different risk score
    depending only on which path happened to serve the request."""
    monkeypatch.setattr(backend_schema, "fetch_farmer_features",
                        lambda db, fid: SimpleNamespace(**DB_ROW_WITH_NULLS))
    with caplog.at_level(logging.INFO):
        db_resp = client.post("/predict/farmer", json={"farmer_id": FARMER_ID})
    assert db_resp.status_code == 200, db_resp.text
    assert "farmer_source=database" in caplog.text

    def _must_not_be_called(db, fid):
        raise AssertionError("a payload was sent — the database fallback must not run")
    monkeypatch.setattr(backend_schema, "fetch_farmer_features", _must_not_be_called)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        payload_resp = client.post(
            "/predict/farmer", json={"farmer_id": FARMER_ID, "farmer": PAYLOAD_SPARSE})
    assert payload_resp.status_code == 200, payload_resp.text
    assert "farmer_source=payload" in caplog.text

    from_db, from_payload = db_resp.json(), payload_resp.json()
    assert from_payload["prediction"] == from_db["prediction"]
    assert from_payload["risk_score_percent"] == from_db["risk_score_percent"]
    assert from_payload["risk_assessment"] == from_db["risk_assessment"]
    assert from_payload["top_risk_factors"] == from_db["top_risk_factors"]
    assert from_payload["probabilities"] == from_db["probabilities"]
    assert from_payload["crop_type"] == from_db["crop_type"] == "Maize", (
        "an absent `crops` column defaults to the same dominant crop on both paths")


def test_protected_attributes_still_raise_when_sourced_from_the_payload():
    """`domain/risk.py`'s guard is proven on a database Row in `test_domain.py`. A payload is a
    different object type carrying the same fields, and the guard must fire on it too — a fairness
    constraint that only holds on the route you happen to test is not a constraint."""
    farmer = FarmerPayload(**DB_ROW_POPULATED)
    signals = PredictionsService._coping_signals(farmer)
    # Guard-safe by construction: `_coping_signals` never puts either protected attribute into the
    # dict it hands to `risk.assess_vulnerability`, for a payload-sourced farmer exactly as for a
    # database Row.
    assert "head_gender" not in signals
    assert "household_max_education" not in signals

    # A regression that adds either back in must still be caught, at the domain boundary,
    # whichever object supplied the value.
    with pytest.raises(ValueError, match="head_gender"):
        risk.assess_vulnerability({**signals, "head_gender": farmer.head_gender})
    with pytest.raises(ValueError, match="household_max_education"):
        risk.assess_vulnerability(
            {**signals, "household_max_education": farmer.household_max_education})
