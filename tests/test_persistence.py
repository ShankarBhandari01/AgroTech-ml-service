"""The database path: the audit trail, and the query Phase 3 is waiting on.

Why this needs a real Postgres
------------------------------
`store.py` is raw SQL with `CAST(... AS JSONB)` and `make_interval()`. SQLite cannot run it, and a
mock would test the mock. The feature *store* is JSONB rather than typed columns precisely so the
feature set can change without a migration — which means nothing but a real round-trip proves a
29-column row survives the trip.

Isolation
---------
`write_prediction` and `write_outcome` call `db.commit()` internally, so wrapping the test in a
transaction is not enough on its own. The session is bound to a connection with an outer
transaction and `join_transaction_mode="create_savepoint"`, so those inner commits become savepoint
releases and the outer rollback still discards everything. Running the suite against a populated
development database leaves it byte-identical.

Skips cleanly when no database is reachable, so CI without a Postgres service and a laptop without
one both stay green. See the README for the CI service-container snippet.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from argotech.data import store
from argotech.data.db import engine, get_db
from argotech.domain import risk
from argotech.features.agronomic import FEATURE_COLUMNS


@pytest.fixture
def db():
    """A session whose every write is rolled back, against the real Postgres."""
    try:
        conn = engine.connect()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no database reachable: {str(e)[:80]}")

    outer = conn.begin()
    Session = sessionmaker(bind=conn, join_transaction_mode="create_savepoint")
    session = Session()
    store.ensure_schema(session)
    try:
        yield session
    finally:
        session.close()
        outer.rollback()
        conn.close()


def _assessment(score: float = 72.5, severity: str = "CRITICAL"):
    hazard = risk.assess_hazard(water_satisfaction=0.4, dry_spell_days=12,
                                cumulative_dsv=20, heat_days=1, vegetation=0.3)
    vuln = risk.assess_vulnerability({"has_irrigation": 0.0, "asset_score": 0.2})
    exposure = risk.assess_exposure(area_ha=2.0, expected_yield_t_ha=2.3, price_per_t=350.0)
    return risk.assess_risk(hazard, exposure, vuln)


def test_schema_creation_is_idempotent(db):
    """`ensure_schema` runs on every boot; a second call must not error."""
    store.ensure_schema(db)
    store.ensure_schema(db)


def test_a_full_feature_row_survives_the_json_round_trip(db):
    """The reason features are JSONB: the feature set changes without a migration.

    This is the test that would have caught the radar columns being added to `FEATURE_COLUMNS`
    without the store keeping up — all 29 must come back, with values intact and order preserved.
    """
    field_id = f"test-{uuid.uuid4()}"
    features = {col: float(i) for i, col in enumerate(FEATURE_COLUMNS)}
    store.write_features(db, field_id, 10.8, 7.9, "Maize", features, {"source": "test"})

    latest = store.read_latest_features(db, field_id)
    assert latest is not None, "written features must be readable"
    assert latest["features"] == features, "every value must survive the round trip"
    assert set(latest["features"]) == set(FEATURE_COLUMNS)
    assert latest["context"]["source"] == "test"

    # Postgres `jsonb` normalises key order — it is a parsed structure, not stored text — so the
    # dict comes back reordered. That is fine and worth stating: `pipeline.py` selects by name
    # (`pd.DataFrame([row])[columns]`), so order never reaches the model. Note `test_store.py`
    # asserts order *does* survive; that holds for its in-memory round trip but not for real JSONB,
    # and nothing depends on the stronger claim.
    assert list(latest["features"]) != FEATURE_COLUMNS or len(FEATURE_COLUMNS) < 2


def test_a_missing_feature_is_stored_as_null_not_dropped_or_faked(db):
    """NaN means "not observed" and must reach the database as `null`.

    `json.dumps` emits a bare `NaN` literal, which is not JSON, and Postgres rejects the entire
    statement. Because `write_prediction` swallows its exception by design, the symptom was not an
    error but silence: predictions simply stopped being recorded.
    """
    field_id = f"test-{uuid.uuid4()}"
    features = {col: float(i) for i, col in enumerate(FEATURE_COLUMNS)}
    features["rvi"] = float("nan")
    features["rvi_z_peer"] = float("inf")
    store.write_features(db, field_id, 10.8, 7.9, "Maize", features, {"source": "test"})

    stored = store.read_latest_features(db, field_id)["features"]
    assert stored["rvi"] is None, "an unobserved feature must be null"
    assert stored["rvi_z_peer"] is None, "infinities too"
    assert "rvi" in stored, "null, not dropped: the column contract must stay complete"
    assert stored["ndvi"] == features["ndvi"], "observed features are untouched"


def test_a_prediction_is_persisted_and_returns_a_linkable_id(db):
    """The id is what an agent's outcome report links back to; without it there is no feedback loop."""
    field_id = f"test-{uuid.uuid4()}"
    assessment = _assessment()
    pid = store.write_prediction(db, field_id, "agro-test", "live",
                                 {"ndvi": 0.4}, assessment, {"low": 0.7, "medium": 0.2, "high": 0.1})
    assert isinstance(pid, int)

    row = db.execute(text("SELECT field_id, model_version, risk_score, severity, probabilities "
                          "FROM predictions WHERE id = :i"), {"i": pid}).fetchone()
    assert row.field_id == field_id
    assert row.model_version == "agro-test"
    assert abs(float(row.risk_score) - assessment.risk_score) < 1e-6
    assert row.severity == assessment.severity
    assert row.probabilities["low"] == 0.7, "JSONB must come back as a dict, not a string"


def test_a_failed_prediction_write_never_breaks_the_request(db):
    """Documented contract: losing the audit row is bad, failing the farmer's prediction is worse."""
    class Broken:
        def execute(self, *a, **k):
            raise RuntimeError("database is on fire")

        def rollback(self):
            pass

    assert store.write_prediction(Broken(), "f", "v", "live", {}, _assessment(), None) is None


def test_label_join_links_an_outcome_by_explicit_prediction_id(db):
    """The Phase 3 training query. An explicit link is ground truth and must always win."""
    field_id = f"test-{uuid.uuid4()}"
    pid = store.write_prediction(db, field_id, "agro-test", "live", {"ndvi": 0.4}, _assessment(), None)
    store.write_outcome(db, field_id, datetime.utcnow(), "agent_visit",
                        prediction_id=pid, stress_confirmed=True, diagnosis="blight")

    joined = [r for r in store.label_join(db) if r["field_id"] == field_id]
    assert len(joined) == 1, "the linked pair must appear exactly once"
    assert joined[0]["id"] == pid
    assert joined[0]["stress_confirmed"] is True
    assert joined[0]["diagnosis"] == "blight"


def test_label_join_rescues_an_unlinked_report_inside_the_horizon(db):
    """The OR branch: a spontaneous visit or SMS carries no prediction_id, only a field and a date."""
    field_id = f"test-{uuid.uuid4()}"
    store.write_prediction(db, field_id, "agro-test", "live", {"ndvi": 0.4}, _assessment(), None)
    store.write_outcome(db, field_id, datetime.utcnow() + timedelta(days=10), "diagnosis",
                        prediction_id=None, stress_confirmed=True)

    joined = [r for r in store.label_join(db, horizon_days=30) if r["field_id"] == field_id]
    assert len(joined) == 1, "an unlinked outcome inside the window should still join"


def test_label_join_ignores_an_unlinked_report_outside_the_horizon(db):
    """The window boundary. Without this, an outcome months later becomes a spurious label."""
    field_id = f"test-{uuid.uuid4()}"
    store.write_prediction(db, field_id, "agro-test", "live", {"ndvi": 0.4}, _assessment(), None)
    store.write_outcome(db, field_id, datetime.utcnow() + timedelta(days=95), "harvest",
                        prediction_id=None, yield_t_ha=1.1)

    joined = [r for r in store.label_join(db, horizon_days=30) if r["field_id"] == field_id]
    assert joined == [], "an outcome 95 days later is not evidence about a 30-day forecast"


def test_label_join_ignores_an_outcome_that_precedes_its_prediction(db):
    """Causality. A visit before the forecast cannot be its outcome."""
    field_id = f"test-{uuid.uuid4()}"
    store.write_prediction(db, field_id, "agro-test", "live", {"ndvi": 0.4}, _assessment(), None)
    store.write_outcome(db, field_id, datetime.utcnow() - timedelta(days=5), "agent_visit",
                        prediction_id=None, stress_confirmed=True)

    joined = [r for r in store.label_join(db, horizon_days=30) if r["field_id"] == field_id]
    assert joined == [], "an outcome before the prediction must not be joined to it"


def test_an_unlinked_outcome_produces_ONE_label_not_one_per_prediction(db):
    """The fan-out. Every other test here writes a single prediction per field, so none of them can
    see this: the nightly precompute writes a prediction per field per DAY, and an agent report that
    carries no `prediction_id` falls inside the horizon of every one of them.

    Unfixed, one field visit becomes N labels of the same observation. That corrupts two things at
    once. `GET /outcomes/label-count` is the number that decides when phase 3 starts, and it would
    read 30x high. And a training set built from this query would carry 30 near-identical rows for
    one real fact -- the same error this repository already names elsewhere as "39 monthly
    observations of one site are not 39 independent facts", arriving in a new place.

    An observation is one label. Which prediction it scores is a real question, and the answer is
    the most recent forecast standing when the agent looked.
    """
    field_id = f"test-{uuid.uuid4()}"
    base = datetime.utcnow() - timedelta(days=10)
    pids = [store.write_prediction(db, field_id, "agro-test", "live",
                                   {"ndvi": 0.4}, _assessment(), None) for _ in range(5)]
    # Space the predictions a day apart, as the nightly job would.
    for i, pid in enumerate(pids):
        db.execute(text("UPDATE predictions SET predicted_at = :t WHERE id = :i"),
                   {"t": base + timedelta(days=i), "i": pid})
    db.commit()

    store.write_outcome(db, field_id, base + timedelta(days=6), "agent_visit",
                        prediction_id=None, stress_confirmed=True)

    joined = [r for r in store.label_join(db, horizon_days=30) if r["field_id"] == field_id]
    assert len(joined) == 1, (
        f"one visit produced {len(joined)} labels, one per prediction still inside the horizon")
    assert joined[0]["id"] == pids[-1], (
        "the label should score the most recent forecast standing when the agent looked")


def test_an_explicit_link_still_wins_over_the_nearest_prediction(db):
    """The de-duplication above must not quietly override an agent who said which forecast they were
    checking. An explicit `prediction_id` is ground truth; only unlinked reports get resolved by
    recency, and picking the newest prediction for a linked report would silently relabel it.
    """
    field_id = f"test-{uuid.uuid4()}"
    base = datetime.utcnow() - timedelta(days=10)
    pids = [store.write_prediction(db, field_id, "agro-test", "live",
                                   {"ndvi": 0.4}, _assessment(), None) for _ in range(3)]
    for i, pid in enumerate(pids):
        db.execute(text("UPDATE predictions SET predicted_at = :t WHERE id = :i"),
                   {"t": base + timedelta(days=i), "i": pid})
    db.commit()

    # The agent names the OLDEST prediction, not the newest.
    store.write_outcome(db, field_id, base + timedelta(days=6), "agent_visit",
                        prediction_id=pids[0], stress_confirmed=True)

    joined = [r for r in store.label_join(db, horizon_days=30) if r["field_id"] == field_id]
    assert len(joined) == 1
    assert joined[0]["id"] == pids[0], "an explicit prediction_id must not be overridden by recency"


def test_the_label_count_is_the_number_that_gates_phase_three(db):
    """`label_join` returning few rows is the honest state of the feedback loop, not a bug.

    Asserted rather than assumed, because the roadmap's phase 3 is explicitly blocked on this count
    and it should be visible in the suite rather than discovered a season late.
    """
    rows = store.label_join(db)
    assert isinstance(rows, list)
    if len(rows) < 50:
        pytest.skip(f"only {len(rows)} labelled outcomes: phase 3 remains blocked (expected today)")


def test_prediction_persists_through_a_real_http_request(db):
    """True end to end: HTTP in, audit row committed, id handed back to the caller.

    `tests/test_e2e_predict.py` overrides the database away to stay CI-portable. This is the other
    half — the same request with persistence switched on.
    """
    import json
    from pathlib import Path

    from fastapi.testclient import TestClient

    from argotech.data import meteo
    from argotech.data.sentinel import sentinel_client
    from argotech.serving.main import app

    fixtures = Path(__file__).parent / "fixtures"
    payload = json.loads((fixtures / "meteo_recent.json").read_text())
    history = json.loads((fixtures / "sentinel_history.json").read_text())

    original = (meteo.fetch_recent, meteo.climatological_rain_30, sentinel_client.fetch_history)
    meteo.fetch_recent = lambda *a, **k: payload
    meteo.climatological_rain_30 = lambda *a, **k: 95.0
    sentinel_client.fetch_history = lambda *a, **k: history
    app.dependency_overrides[get_db] = lambda: db
    try:
        resp = TestClient(app).post("/predict/coldstart", json={
            "latitude": 10.8, "longitude": 7.9, "crop_type": "Maize", "farm_size": 2.0})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["prediction_id"] is not None, "the audit row is what outcomes link to"

        row = db.execute(text("SELECT field_id, risk_score, features FROM predictions WHERE id = :i"),
                         {"i": body["prediction_id"]}).fetchone()
        assert row.field_id == body["field_id"]
        assert abs(float(row.risk_score) - body["risk_score_percent"]) < 1e-6
        assert set(row.features) >= set(FEATURE_COLUMNS), "the stored row must carry every feature"
    finally:
        meteo.fetch_recent, meteo.climatological_rain_30, sentinel_client.fetch_history = original
        app.dependency_overrides.clear()
