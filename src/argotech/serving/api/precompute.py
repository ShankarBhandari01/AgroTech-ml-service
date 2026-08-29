"""Batch precompute — the field list Kotlin supplies instead of this service enumerating farms
itself. See `jobs/precompute.py` for why: this is the last consumer of `run`, and the endpoint's
only job is to hand it a caller-supplied field list instead of letting it query the backend's
schema via `data/backend_schema.list_fields`.

No auth, tenancy or rate limiting here by design (see README): this service has none, and the
Kotlin backend is responsible for all three before it ever calls this. This endpoint only accepts
field coordinates and writes `field_features` — nothing a caller could misuse beyond that.
"""

from fastapi import APIRouter

from argotech.jobs.precompute import run as precompute_run
from argotech.serving.schemas.request import PrecomputeBatchRequest

router = APIRouter()


@router.post("/precompute/batch")
async def precompute_batch(payload: PrecomputeBatchRequest) -> dict:
    """Precompute `field_features` for exactly the fields given, pacing upstream calls the same
    way the nightly cron job does. An empty list is a no-op that reports zero fields."""
    fields = [f.model_dump() for f in payload.fields]
    return await precompute_run(fields=fields)
