"""This service's own tables: precomputed features, predictions, and outcomes.

Three things live here, and only the first is an optimisation:

* `field_features` — the expensive upstream work (ERA5, Sentinel-2, hourly humidity, the 4-year
  rainfall climatology), computed once nightly per field instead of three blocking calls to two
  free public APIs inside every request.
* `predictions` — every prediction with the exact feature vector that produced it. Without this, a
  prediction cannot be explained after the fact, reproduced, or joined to what actually happened.
* `field_outcomes` — what the agent found, what was diagnosed, what was harvested. This is the
  label stream. Everything in phase 3 of docs/model-design.md is blocked on it existing, and it
  only starts accumulating from the day it is switched on — which is why it ships now, months
  before anything can train on it.

These are *our* tables, distinct from the Kotlin backend's `farmer_profiles` / `farmers_ml_profiles`
which we only read.

Features are stored as JSONB rather than 26 typed columns on purpose: the feature set will change,
and a column-per-feature schema costs a migration every time. The column contract lives with the
model artifact (`feature_columns`), so the ordering is enforced at load time where it matters.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

SCHEMA = """
CREATE TABLE IF NOT EXISTS field_features (
    field_id     TEXT        NOT NULL,
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    latitude     DOUBLE PRECISION NOT NULL,
    longitude    DOUBLE PRECISION NOT NULL,
    crop         TEXT,
    features     JSONB       NOT NULL,
    context      JSONB       NOT NULL,
    PRIMARY KEY (field_id, computed_at)
);
CREATE INDEX IF NOT EXISTS field_features_latest
    ON field_features (field_id, computed_at DESC);

CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    field_id        TEXT        NOT NULL,
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version   TEXT        NOT NULL,
    feature_source  TEXT        NOT NULL,
    features        JSONB       NOT NULL,
    risk_score      DOUBLE PRECISION NOT NULL,
    severity        TEXT        NOT NULL,
    dominant_hazard TEXT,
    expected_loss   DOUBLE PRECISION,
    hazard          JSONB       NOT NULL,
    vulnerability   JSONB       NOT NULL,
    probabilities   JSONB
);
CREATE INDEX IF NOT EXISTS predictions_field_time
    ON predictions (field_id, predicted_at DESC);

CREATE TABLE IF NOT EXISTS field_outcomes (
    id               BIGSERIAL PRIMARY KEY,
    prediction_id    BIGINT REFERENCES predictions(id) ON DELETE SET NULL,
    field_id         TEXT        NOT NULL,
    observed_at      TIMESTAMPTZ NOT NULL,
    recorded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    outcome_type     TEXT        NOT NULL,
    stress_confirmed BOOLEAN,
    diagnosis        TEXT,
    yield_t_ha       DOUBLE PRECISION,
    notes            TEXT,
    reported_by      TEXT
);
CREATE INDEX IF NOT EXISTS field_outcomes_field_time
    ON field_outcomes (field_id, observed_at DESC);
"""

# A precomputed row older than this is not used. Sentinel-2 aggregates over 30 days and the weather
# window is 90 days, so a day-old row is agronomically identical — but a stale row after an outage
# would silently serve last week's weather as today's.
MAX_FEATURE_AGE = timedelta(hours=48)


def ensure_schema(db: Session) -> None:
    """Idempotent DDL. Called by the batch job, which is single-instance — not on API startup,
    where N replicas would race."""
    db.execute(text(SCHEMA))
    db.commit()


def is_fresh(computed_at: datetime | None, now: datetime | None = None,
             max_age: timedelta = MAX_FEATURE_AGE) -> bool:
    """Whether a precomputed row may be served. Naive timestamps are assumed UTC."""
    if computed_at is None:
        return False
    now = now or datetime.now(UTC)
    if computed_at.tzinfo is None:
        computed_at = computed_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return timedelta(0) <= now - computed_at <= max_age


def _jsonb(payload, **kwargs) -> str:
    """Serialise for a JSONB column, mapping non-finite floats to `null`.

    `json.dumps` emits bare `NaN` / `Infinity` literals. Python accepts them on the way back in;
    the JSON spec does not contain them, and Postgres rejects the whole statement with
    "invalid input syntax for type json".

    This is not hypothetical. A feature that is legitimately absent — a radar band for a field with
    no Sentinel-1 pass, a canopy index under cloud — is NaN by design, because the alternative is
    fabricating a value. The moment such a feature entered `FEATURE_COLUMNS`, every prediction
    write began failing, and `write_prediction` swallows its exception on purpose so that a broken
    audit table cannot break a farmer's prediction. The result was a service that looked healthy
    and silently stopped recording the outcomes the whole training roadmap depends on.

    `null` is the right mapping, not 0.0: it means "not observed", which is exactly what NaN meant.
    """
    def clean(value):
        if isinstance(value, float) and not isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    return json.dumps(clean(payload), **kwargs)


# ------------------------------------------------------------------ features

def write_features(db: Session, field_id: str, lat: float, lon: float, crop: str,
                   features: dict, context: dict) -> None:
    db.execute(text("""
        INSERT INTO field_features (field_id, latitude, longitude, crop, features, context)
        VALUES (:field_id, :lat, :lon, :crop, CAST(:features AS JSONB), CAST(:context AS JSONB))
        ON CONFLICT (field_id, computed_at) DO NOTHING
    """), {"field_id": field_id, "lat": lat, "lon": lon, "crop": crop,
           "features": _jsonb(features), "context": _jsonb(context, default=str)})
    db.commit()


def read_latest_features(db: Session, field_id: str) -> dict | None:
    """Most recent precomputed row for a field, or None if there is none or it is stale."""
    row = db.execute(text("""
        SELECT features, context, computed_at
        FROM field_features WHERE field_id = :field_id
        ORDER BY computed_at DESC LIMIT 1
    """), {"field_id": field_id}).fetchone()
    if row is None or not is_fresh(row.computed_at):
        return None
    return {"features": row.features, "context": row.context, "computed_at": row.computed_at}


def prune_features(db: Session, keep_days: int = 90) -> int:
    """Drop feature history beyond `keep_days`. Retained that long because a prediction under
    investigation must be reproducible from the row that produced it."""
    result = db.execute(text(
        "DELETE FROM field_features WHERE computed_at < now() - make_interval(days => :d)"),
        {"d": keep_days})
    db.commit()
    return result.rowcount or 0


# ------------------------------------------------------------------ predictions

def write_prediction(db: Session, field_id: str, model_version: str, feature_source: str,
                     features: dict, assessment: Any, probabilities: dict | None) -> int | None:
    """Persist a prediction, returning its id so an outcome can be linked to it later.

    Never raises into the request path: losing the audit row is bad, failing the farmer's prediction
    because the audit table is unavailable is worse.
    """
    try:
        row = db.execute(text("""
            INSERT INTO predictions (field_id, model_version, feature_source, features, risk_score,
                                     severity, dominant_hazard, expected_loss, hazard,
                                     vulnerability, probabilities)
            VALUES (:field_id, :model_version, :feature_source, CAST(:features AS JSONB),
                    :risk_score, :severity, :dominant_hazard, :expected_loss,
                    CAST(:hazard AS JSONB), CAST(:vulnerability AS JSONB),
                    CAST(:probabilities AS JSONB))
            RETURNING id
        """), {
            "field_id": field_id,
            "model_version": model_version,
            "feature_source": feature_source,
            "features": _jsonb(features),
            "risk_score": assessment.risk_score,
            "severity": assessment.severity,
            "dominant_hazard": assessment.hazard.dominant,
            "expected_loss": assessment.expected_loss,
            "hazard": _jsonb(assessment.hazard.as_dict()),
            "vulnerability": _jsonb(assessment.vulnerability.as_dict()),
            "probabilities": _jsonb(probabilities) if probabilities else None,
        }).fetchone()
        db.commit()
        return int(row.id)
    except Exception as e:  # noqa: BLE001
        print(f"[store] prediction not persisted: {e}")
        db.rollback()
        return None


# ------------------------------------------------------------------ outcomes

def write_outcome(db: Session, field_id: str, observed_at: datetime, outcome_type: str,
                  prediction_id: int | None = None, stress_confirmed: bool | None = None,
                  diagnosis: str | None = None, yield_t_ha: float | None = None,
                  notes: str | None = None, reported_by: str | None = None) -> int:
    row = db.execute(text("""
        INSERT INTO field_outcomes (prediction_id, field_id, observed_at, outcome_type,
                                    stress_confirmed, diagnosis, yield_t_ha, notes, reported_by)
        VALUES (:prediction_id, :field_id, :observed_at, :outcome_type,
                :stress_confirmed, :diagnosis, :yield_t_ha, :notes, :reported_by)
        RETURNING id
    """), {"prediction_id": prediction_id, "field_id": field_id, "observed_at": observed_at,
           "outcome_type": outcome_type, "stress_confirmed": stress_confirmed,
           "diagnosis": diagnosis, "yield_t_ha": yield_t_ha, "notes": notes,
           "reported_by": reported_by}).fetchone()
    db.commit()
    return int(row.id)


def label_join(db: Session, horizon_days: int = 30) -> list[dict]:
    """Predictions joined to the outcomes that followed them within `horizon_days`.

    The training query for phase 3. It returns nothing today, and that is the point: it is the
    query whose row count tells you when supervised modelling on real outcomes becomes possible.
    """
    rows = db.execute(text("""
        SELECT p.id, p.field_id, p.predicted_at, p.model_version, p.features, p.risk_score,
               p.severity, o.outcome_type, o.stress_confirmed, o.diagnosis, o.yield_t_ha,
               o.observed_at
        FROM predictions p
        JOIN field_outcomes o
          -- An explicit prediction_id always wins: when an agent says which forecast they were
          -- checking, that link is ground truth and must not be second-guessed by a time window.
          -- The window only rescues unlinked reports (a spontaneous field visit, an SMS).
          ON o.prediction_id = p.id
          OR (o.prediction_id IS NULL
              AND o.field_id = p.field_id
              AND o.observed_at BETWEEN p.predicted_at
                                   AND p.predicted_at + make_interval(days => :h))
        ORDER BY p.predicted_at
    """), {"h": horizon_days}).fetchall()
    return [dict(r._mapping) for r in rows]
