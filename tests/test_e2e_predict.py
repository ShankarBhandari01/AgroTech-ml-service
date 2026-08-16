"""End-to-end tests: a real HTTP request through the real app to the real model artifact.

What is real here, and what is substituted
------------------------------------------
**Real:** the FastAPI app and routing, request/response validation, the feature builder, the joblib
model, the FAO-56 water balance, the BLITECAST severity accumulation, the noisy-OR composition, and
the response copy. That is the whole chain a defect can hide in, and none of it is mocked.

**Substituted:** the two network upstreams and the database.

* **Upstreams** — Open-Meteo and CDSE are slow (the live smoke test that found the copy defects took
  41 s), rate-limited, and non-deterministic: the same coordinate returns different weather tomorrow,
  so an assertion on a risk score would rot within a day. The fixtures are *recorded from the real
  caches*, so the shapes and value ranges are genuine rather than invented.
* **Database** — `PredictionsService` skips the audit write when `db is None`, so overriding the
  dependency exercises the response path without requiring Postgres in CI.

What this catches that unit tests do not
----------------------------------------
The unit suite tests each component against its own contract. This tests that the *composition*
holds: that the risk score really is `hazard x (0.5 + 0.5 x vulnerability) x MAX_LOSS`, that the
severity band matches the score, and that the response a caller receives is internally consistent.
Both defects found by running the service by hand — inverted coping-gap copy, and probabilities that
appear to contradict the prediction — were in exactly that gap.

For a genuine live check against the real upstreams, run the service and curl it; see the README.
That belongs in a manual or nightly job, not in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from argotech.data import meteo
from argotech.data.sentinel import sentinel_client
from argotech.data.db import get_db
from argotech.domain import risk
from argotech.serving.main import app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client(monkeypatch):
    """The app with its network and database edges replaced, and nothing else."""
    payload = json.loads((FIXTURES / "meteo_recent.json").read_text())
    history = json.loads((FIXTURES / "sentinel_history.json").read_text())

    monkeypatch.setattr(meteo, "fetch_recent", lambda *a, **k: payload)
    monkeypatch.setattr(meteo, "climatological_rain_30", lambda *a, **k: 95.0)
    monkeypatch.setattr(sentinel_client, "fetch_history", lambda *a, **k: history)

    app.dependency_overrides[get_db] = lambda: None
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _predict(client) -> dict:
    resp = client.post("/predict/coldstart", json={
        "latitude": 10.8, "longitude": 7.9, "crop_type": "Maize", "farm_size": 2.0,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_a_bare_coordinate_yields_a_complete_prediction(client):
    """The coldstart contract: GPS in, full decomposed assessment out, no farmer record needed."""
    body = _predict(client)

    for key in ("field_id", "model_version", "prediction", "priority_label",
                "risk_score_percent", "probabilities", "top_risk_factors",
                "inference", "risk_assessment", "spatiotemporal_indices",
                "microclimate_metrics", "recommended_action"):
        assert key in body, f"missing {key}"

    assert body["prediction"] in (0, 1, 2)
    assert 0.0 <= body["risk_score_percent"] <= 100.0
    assert body["recommended_action"].strip()


def test_the_learned_model_actually_contributed(client):
    """Guards the silent-degradation path.

    `pipeline` runs the model only when a Sentinel-2 scene is available, and slices the feature row
    by the artifact's own `feature_columns`. If the builder and the artifact ever disagree the
    service does not crash — it quietly returns a physics-only assessment, which is a far worse
    failure than an exception because the numbers still look plausible.
    """
    body = _predict(client)
    assert body["inference"]["model_contributed"] is True
    assert body["model_version"].startswith("agro-")
    assert body["risk_assessment"]["hazard"]["vegetation"] > 0.0


def test_the_risk_composition_arithmetic_holds_end_to_end(client):
    """risk_score = hazard x (0.5 + 0.5 x vulnerability), as a percentage of MAX_LOSS_FRACTION.

    No unit test covers this: each term is tested in isolation, but the multiplication happens in
    the serving path and is what the farmer's number actually is.
    """
    body = _predict(client)
    ra = body["risk_assessment"]
    hazard = ra["hazard"]["combined"]
    vuln = ra["vulnerability"]["score"]

    expected = 100.0 * hazard * (0.5 + 0.5 * vuln)
    assert abs(ra["risk_score"] - expected) < 0.15, (
        f"score {ra['risk_score']} != hazard {hazard} x (0.5 + 0.5 x {vuln})")
    assert abs(body["risk_score_percent"] - ra["risk_score"]) < 1e-6


def test_severity_band_and_prediction_class_agree(client):
    """`prediction` is derived from `severity`; a mismatch means the mapping drifted."""
    body = _predict(client)
    score = body["risk_score_percent"]
    severity = body["inference"]["risk_level"]

    expected = ("CRITICAL" if score >= 65 else "ELEVATED" if score >= 35
                else "WATCH" if score >= 15 else "NORMAL")
    assert severity == expected, f"score {score} should be {expected}, got {severity}"
    assert body["prediction"] == {"NORMAL": 0, "WATCH": 0, "ELEVATED": 1, "CRITICAL": 2}[severity]


def test_expected_loss_is_a_share_of_the_value_at_risk(client):
    """The currency figure a triage queue ranks by has to be bounded by what is actually at stake."""
    ra = _predict(client)["risk_assessment"]
    assert 0.0 <= ra["expected_loss_usd"] <= ra["value_at_risk_usd"]
    assert ra["value_at_risk_usd"] > 0.0


def test_no_coping_gap_reads_as_though_the_farmer_has_the_thing(client):
    """The defect a live request exposed, pinned at the HTTP boundary.

    Coldstart passes no farmer record, so every coping factor is absent and this list is populated —
    which makes it the right request to assert on.
    """
    body = _predict(client)
    for factor in body["top_risk_factors"]:
        assert not factor.startswith("has "), f"inverted meaning: {factor!r}"
        assert "capacity: has" not in factor
    assert any(f in body["top_risk_factors"] for f in risk.COPING_GAP_LABELS.values()), \
        "coldstart has no coping data, so at least one gap should be reported"


def test_probabilities_say_what_they_are_over(client):
    """They describe the vegetation hazard only, and routinely disagree with `prediction`.

    The caveat sits at the response root, not inside `probabilities`: the Kotlin client binds that
    object to `Map<String, Double>`, so a string entry inside it would fail to coerce and send
    every prediction down the FALLBACK path.
    """
    body = _predict(client)
    probs = body["probabilities"]
    assert set(probs) == {"low", "medium", "high"}, "this object must stay numeric for the client"
    assert all(isinstance(v, (int, float)) for v in probs.values())
    assert "vegetation hazard" in body["probabilities_of"]
    assert abs(sum(probs.values()) - 1.0) < 0.02


def test_a_second_identical_request_returns_the_same_answer(client):
    """Determinism at the HTTP boundary.

    Two identical requests against identical inputs must agree; if they do not, something in the
    path is reading wall-clock time or unseeded randomness, and every metric in the design doc is
    built on sand.
    """
    first, second = _predict(client), _predict(client)
    for key in ("prediction", "risk_score_percent", "priority_label"):
        assert first[key] == second[key]
    assert first["risk_assessment"]["hazard"] == second["risk_assessment"]["hazard"]


def test_crop_health_endpoint_serves_the_index(client):
    resp = client.get("/predict/crop-health", params={"latitude": 10.8, "longitude": 7.9})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["available"] is True
    assert 0.0 <= body["crop_health"]["score"] <= 100.0
    assert body["baseline"]["observations"] > 0
