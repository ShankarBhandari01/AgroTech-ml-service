"""Self-check for the feature/prediction store. Run: python tests/test_store.py

No database needed. The two things worth testing here are the freshness rule (which decides whether
a stored row may be served) and JSONB round-trip fidelity (because a precomputed prediction and a
live one must be the same computation, and serialisation is the only place they can diverge).
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argotech.data.store import MAX_FEATURE_AGE, is_fresh
from argotech.features.agronomic import FEATURE_COLUMNS


def test_is_fresh():
    now = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)

    assert is_fresh(now - timedelta(hours=1), now)
    assert is_fresh(now, now)                                    # just written
    assert is_fresh(now - MAX_FEATURE_AGE, now)                  # exactly at the boundary
    assert not is_fresh(now - MAX_FEATURE_AGE - timedelta(seconds=1), now)
    assert not is_fresh(None, now)                               # nothing stored

    # A naive timestamp (psycopg2 can hand one back depending on the column type) is read as UTC
    # rather than crashing on an offset-naive/aware comparison.
    assert is_fresh(datetime(2026, 8, 11, 11, 0), now)
    assert not is_fresh(datetime(2026, 8, 1, 0, 0), now)

    # A row from the future means clock skew or a bad write; refuse it rather than serve it.
    assert not is_fresh(now + timedelta(hours=2), now)


def test_context_round_trip():
    """The context dict must survive JSON exactly, because the serving path reads it back by key."""
    context = {
        "sat": {"ndvi": 0.31, "ndmi": -0.077, "evi": 0.219, "vci": 33.5, "ndvi_z_peer": -0.42},
        "index_source": "sentinel-2",
        "sensing_date": "2026-08-06",
        "crop_health": {"score": 36.2, "vigour": 0.335, "moisture": 0.206,
                        "anomaly": 0.556, "status": "Stressed"},
        "dsv_total": 8,
        "spray_due": False,
    }
    restored = json.loads(json.dumps(context, default=str))
    assert restored == context

    # Everything the pipeline reads out of the context must be present after the round trip.
    for key in ("sat", "index_source", "sensing_date", "crop_health", "dsv_total", "spray_due"):
        assert key in restored
    assert isinstance(restored["spray_due"], bool)
    assert isinstance(restored["dsv_total"], int)
    for key in ("ndvi", "ndmi", "evi", "vci", "ndvi_z_peer"):
        assert key in restored["sat"]
    assert set(restored["crop_health"]) == {"score", "vigour", "moisture", "anomaly", "status"}


def test_feature_row_round_trip():
    """A feature row must come back with every model column intact and numeric."""
    row = {col: 1.5 for col in FEATURE_COLUMNS}
    row["dry_spell_30"] = 4          # ints stay ints
    row["days_since_onset"] = 61

    restored = json.loads(json.dumps(row))
    assert list(restored) == FEATURE_COLUMNS, "column order must survive for DataFrame[columns]"
    assert restored["dry_spell_30"] == 4
    assert all(isinstance(v, (int, float)) for v in restored.values())


def test_expected_yield_treats_zero_as_missing():
    """A zero survey yield must not zero out exposure — see pipeline._expected_yield."""
    from types import SimpleNamespace

    from argotech.serving.pipeline import PredictionsService

    unfilled = SimpleNamespace(yield_value=0.0)
    reported = SimpleNamespace(yield_value=2.4)
    absent = SimpleNamespace(yield_value=None)

    assert PredictionsService._expected_yield(unfilled, ndvi=0.5) > 0.0
    assert PredictionsService._expected_yield(absent, ndvi=0.5) > 0.0
    assert PredictionsService._expected_yield(reported, ndvi=0.5) == 2.4


if __name__ == "__main__":
    test_is_fresh()
    test_context_round_trip()
    test_feature_row_round_trip()
    test_expected_yield_treats_zero_as_missing()
    print("store self-check passed")
