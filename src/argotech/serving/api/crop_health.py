import statistics

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool

from argotech.data.sentinel import sentinel_client
from argotech.domain.indices import crop_health_index

router = APIRouter()


@router.get("/predict/crop-health")
async def get_crop_health(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
) -> dict:
    """Crop Health Index for a field, from Sentinel-2 observations and the field's own history.

    Deterministic and label-free: this is a measured state, not a prediction, so it is available
    for any coordinate on day one. `baseline` reports which reference distribution the VCI and
    anomaly sub-scores were computed against, so a caller can tell a 12-month-history score from a
    thin one instead of treating both as equally trustworthy.
    """
    history = await run_in_threadpool(sentinel_client.fetch_history, latitude, longitude, 365)
    if not history:
        return {
            "available": False,
            "reason": "no cloud-free Sentinel-2 observation in the last 365 days"
                      if sentinel_client.enabled else "Sentinel credentials not configured",
        }

    latest = history[-1]
    series = [obs["ndvi"] for obs in history]
    # The field's own 12-month distribution is the peer group until field boundaries and an
    # agro-ecological-zone cohort are stored; see docs/model-design.md, "baselines".
    peer_mean = statistics.fmean(series)
    peer_std = statistics.pstdev(series) if len(series) > 1 else 0.0

    health = crop_health_index(
        current_ndvi=latest["ndvi"],
        ndmi_value=latest["ndwi"],
        ndvi_min=min(series),
        ndvi_max=max(series),
        peer_mean=peer_mean,
        peer_std=peer_std,
    )

    return {
        "available": True,
        "crop_health": health.as_dict(),
        "observed": {k: round(latest[k], 3) for k in ("ndvi", "ndwi", "evi")},
        "sensing_date": latest["sensing_date"],
        "baseline": {
            "kind": "self_history_12m",
            "observations": len(series),
            "ndvi_mean": round(peer_mean, 3),
            "ndvi_std": round(peer_std, 3),
        },
    }
