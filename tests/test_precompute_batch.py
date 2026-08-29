"""`jobs.precompute.run` with a supplied field list, and the `POST /precompute/batch` endpoint
that hands it one.

This is Task 1 of docs/superpowers/plans/2026-08-29-nightly-job-ownership.md: cut the job's direct
read of `backend_schema.list_fields` by letting a caller (eventually Kotlin, via a batch endpoint)
supply the field list instead, while the crontab path (`fields=None`) keeps querying it exactly as
before.

Every upstream a real run would hit — the model artifact, Open-Meteo/Sentinel via `gather_upstream`,
and the `field_features` write — is monkeypatched out, so this needs no network and no database
connection: with an explicit `fields` list, `run` never calls `store.ensure_schema` or
`list_fields`, and the mocked `store.write_features`/`store.prune_features` never touch a session
either, so `SessionLocal()` is created but never opened.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from argotech.data import backend_schema, store
from argotech.jobs import precompute
from argotech.serving.container import model_manager
from argotech.serving.main import app

FIELDS = [
    {"field_id": "f1", "latitude": 9.0, "longitude": 7.0, "crop": "Maize"},
    {"field_id": "f2", "latitude": 9.1, "longitude": 7.1, "crop": "Maize"},
    {"field_id": "f3", "latitude": 9.2, "longitude": 7.2, "crop": "Rice"},
]


async def _fake_gather_upstream(lat, lon, crop, bounds, peer_ref):
    return {}, {"index_source": "sentinel-2"}


@pytest.fixture(autouse=True)
def _stub_upstreams(monkeypatch):
    """Everything a real precompute pass would hit, replaced. `list_fields` itself is left alone
    (not called) so a test that fails to pass `fields` fails loudly instead of silently reaching
    the real backend schema."""
    monkeypatch.setattr(model_manager, "agronomic_model",
                        lambda: (None, None, None, None, None, None))
    monkeypatch.setattr(precompute, "gather_upstream", _fake_gather_upstream)
    monkeypatch.setattr(store, "write_features", lambda *a, **k: None)
    monkeypatch.setattr(store, "prune_features", lambda *a, **k: 0)
    monkeypatch.setattr(store, "ensure_schema", lambda *a, **k: None)


def _forbid_list_fields(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("list_fields was called even though an explicit field list was given")
    monkeypatch.setattr(precompute, "list_fields", _boom)


# ---------------------------------------------------------------------------
# run(): explicit fields vs. the crontab's default query
# ---------------------------------------------------------------------------

def test_explicit_fields_are_used_as_is_not_queried(monkeypatch):
    _forbid_list_fields(monkeypatch)
    summary = asyncio.run(precompute.run(fields=list(FIELDS), delay=0))
    assert summary["fields"] == 3
    assert summary["ok"] == 3


def test_no_fields_argument_falls_back_to_list_fields(monkeypatch):
    """The crontab path (`python -m argotech.jobs.precompute`, `fields=None`) must keep working
    untouched — Task 4, not this one, retires `list_fields`."""
    monkeypatch.setattr(precompute, "list_fields", lambda db, limit: list(FIELDS))
    summary = asyncio.run(precompute.run(delay=0))
    assert summary["fields"] == 3


def test_empty_field_list_is_a_zero_report_not_an_error():
    summary = asyncio.run(precompute.run(fields=[], delay=0))
    assert summary == {"fields": 0, "ok": 0, "degraded": 0, "failed": 0, "pruned": 0,
                       "degraded_rate": None}


def test_degraded_rate_is_in_the_summary():
    summary = asyncio.run(precompute.run(fields=list(FIELDS), delay=0))
    assert summary["degraded_rate"] == 0.0  # all three faked as sentinel-2, i.e. not degraded


# ---------------------------------------------------------------------------
# Pacing: the property a refactor drops silently.
# ---------------------------------------------------------------------------

def test_a_batch_of_n_fields_takes_at_least_n_minus_1_times_delay():
    """`DELAY_SECONDS` exists because a single training backfill once exhausted Open-Meteo's daily
    quota — a batch endpoint is exactly the shape that tempts someone to parallelise the loop and
    silently drop it. N=3 fields and a short real delay keep this fast without mocking
    `asyncio.sleep`, since `run` already takes `delay` as a parameter."""
    delay = 0.05
    n = len(FIELDS)
    started = time.perf_counter()
    asyncio.run(precompute.run(fields=list(FIELDS), delay=delay))
    elapsed = time.perf_counter() - started
    assert elapsed >= (n - 1) * delay


# ---------------------------------------------------------------------------
# POST /precompute/batch
# ---------------------------------------------------------------------------

client = TestClient(app)


def test_batch_endpoint_returns_the_run_summary():
    """Real end-to-end wiring: the real (mocked-upstream) `run`, real pacing delay included. Two
    fields keeps this quick while still proving the endpoint forwards a multi-item list."""
    resp = client.post("/precompute/batch", json={"fields": FIELDS[:2]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["fields"] == 2
    assert data["ok"] == 2
    assert "degraded_rate" in data


def test_batch_endpoint_empty_list_is_a_no_op():
    resp = client.post("/precompute/batch", json={"fields": []})
    assert resp.status_code == 200
    assert resp.json() == {"fields": 0, "ok": 0, "degraded": 0, "failed": 0, "pruned": 0,
                           "degraded_rate": None}


def test_batch_endpoint_field_shape_matches_list_fields(monkeypatch):
    """The payload shape is the same one `backend_schema.list_fields` already returns — not
    invented here — so a real result from that query validates and reaches `run` as a plain dict.
    `run` itself is stubbed here: this test is about request validation and forwarding, which the
    real-pacing end-to-end test above already covers together with the actual `run`."""
    from argotech.serving.api import precompute as precompute_api

    captured = {}

    async def _fake_run(fields):
        captured["fields"] = fields
        return {"fields": len(fields), "ok": 0, "degraded": 0, "failed": 0, "pruned": 0,
                "degraded_rate": None}

    monkeypatch.setattr(precompute_api, "precompute_run", _fake_run)

    real_shaped = {
        "field_id": "abc-123",
        "latitude": 10.5,
        "longitude": 8.25,
        "crop": backend_schema.DEFAULT_CROP,
    }
    resp = client.post("/precompute/batch", json={"fields": [real_shaped]})
    assert resp.status_code == 200
    assert resp.json()["fields"] == 1
    assert captured["fields"] == [real_shaped]
