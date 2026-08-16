"""Outcome capture — the label stream.

Every advisory this service issues should be reconcilable against what an agent actually found.
That reconciliation is the only source of the T1/T2 labels in docs/model-design.md, and it only
starts accumulating from the day the endpoint exists, which is why it ships long before anything
can train on it.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from argotech.data import store
from argotech.data.db import get_db
from argotech.serving.schemas.request import OutcomeRequest

router = APIRouter()


@router.post("/outcomes", status_code=201)
def record_outcome(payload: OutcomeRequest, db: Session = Depends(get_db)) -> dict:
    """Record what actually happened to a field, optionally linked to the prediction that flagged it."""
    try:
        outcome_id = store.write_outcome(
            db,
            field_id=payload.field_id,
            observed_at=payload.observed_at,
            outcome_type=payload.outcome_type,
            prediction_id=payload.prediction_id,
            stress_confirmed=payload.stress_confirmed,
            diagnosis=payload.diagnosis,
            yield_t_ha=payload.yield_t_ha,
            notes=payload.notes,
            reported_by=payload.reported_by,
        )
    except Exception as e:  # noqa: BLE001
        db.rollback()
        raise HTTPException(500, f"Could not record outcome: {e}")
    return {"outcome_id": outcome_id, "field_id": payload.field_id}


@router.get("/outcomes/label-count")
def label_count(horizon_days: int = 30, db: Session = Depends(get_db)) -> dict:
    """How many prediction/outcome pairs exist — i.e. how close phase 3 is to being possible.

    Deliberately exposed: "we have N labels" is the number that decides when supervised modelling
    on real outcomes can start, and it should be visible on a dashboard rather than discovered by
    someone running a query once a quarter.
    """
    return {"horizon_days": horizon_days, "labelled_pairs": len(store.label_join(db, horizon_days))}
