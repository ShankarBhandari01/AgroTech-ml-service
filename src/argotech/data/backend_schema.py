"""Every read of the Kotlin backend's schema, in one place.

The companion to `store.py`, which owns *our* tables. These are the backend's — `farmer_profiles`,
`farms`, `farm_crops`, `crops`, `farmers_ml_profiles` — and we only ever read them.

**Why this module exists.** The same "a farmer's most recent farm, and what is grown on it" query was
written out twice, in `serving/pipeline.py` and `jobs/precompute.py`. They drifted, and the drift was
invisible: when the backend replaced the free-text `farms.crops` array with a `farm_crops` join table
(its migration V47), one copy started failing outright and the other silently fell through to a
hardcoded default. A shape this fragile — a schema owned by another service, joined across four
tables — needs exactly one definition, so that the next migration breaks one query loudly rather than
two queries quietly.

**What the backend's schema actually looks like**, as of its V47–V49:

* A farmer's coordinates, size and state live on `farms`, NOT on `farmer_profiles`. The profile table
  still carries those columns but nothing has written them since plot data moved; reading them yields
  NULL for every farmer.
* A farm's crops are `farm_crops(farm_id, crop_id)` referencing the `crops` catalogue. The old
  `farms.crops` text array is gone. Names come from the join, which is the point of it: a farmer's
  crops can now be joined to `crop_health`, which has always keyed on `crops.id`.
* `farmers_crops` still exists but is the pre-split, farmer-level join table. It has held zero rows
  since plot data moved to `farms`, so anything counting it counts nothing.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

DEFAULT_CROP = "Maize"

# The one definition of "this farmer's current farm, and its crops".
#
# Correlated on `fp.user_id`, so every query using it must expose `farmer_profiles fp`. A farmer may
# hold several farms (`farms.farmer_profile_id` is not unique); this takes the most recent, which is
# a placeholder for a real choice — a prediction is about a *field*, so the caller should eventually
# name which one it means.
LATEST_FARM_LATERAL = """
    LEFT JOIN LATERAL (
        SELECT
            fa.id, fa.farm_size, fa.state, fa.latitude, fa.longitude,
            (
                SELECT array_agg(c.name ORDER BY c.name)
                FROM farm_crops fc
                JOIN crops c ON c.id = fc.crop_id
                WHERE fc.farm_id = fa.id
            ) AS crops
        FROM farms fa
        WHERE fa.farmer_profile_id = fp.user_id
        ORDER BY fa.created_at DESC NULLS LAST
        LIMIT 1
    ) f ON TRUE
"""

# Distinct crops recorded across ALL of a farmer's plots — a diversity measure, so it counts the
# farmer rather than the one farm the lateral picked.
CROP_DIVERSITY_SUBQUERY = """
    (
        SELECT COUNT(DISTINCT fc.crop_id)
        FROM farm_crops fc
        JOIN farms fa ON fa.id = fc.farm_id
        WHERE fa.farmer_profile_id = fp.user_id
    )
"""

_FARMER_FEATURES_SQL = text(f"""
    SELECT
        fp.user_id, f.farm_size, f.state, f.latitude, f.longitude, f.crops,
        fmp.yield_value, fmp.has_extension_access, fmp.household_max_education,
        fmp.shock_level, fmp.received_assistance, fmp.used_fertilizer,
        fmp.household_size, fmp.transport_cost, fmp.dependency_ratio,
        fmp.asset_score, fmp.postharvest_activity_score, fmp.digital_access_score,
        fmp.has_veterinary_access, fmp.market_access_score, fmp.received_credit,
        fmp.head_gender,
        {CROP_DIVERSITY_SUBQUERY} AS crop_diversity_score
    FROM farmer_profiles fp
    LEFT JOIN farmers_ml_profiles fmp ON fmp.farmer_id = fp.user_id
    {LATEST_FARM_LATERAL}
    WHERE fp.user_id = :farmer_id
""")

_FIELDS_SQL = f"""
    SELECT fp.user_id AS field_id, f.latitude, f.longitude, f.crops
    FROM farmer_profiles fp
    {LATEST_FARM_LATERAL}
    WHERE f.latitude IS NOT NULL AND f.longitude IS NOT NULL
    ORDER BY fp.user_id
"""


def fetch_farmer_features(db, farmer_id: str):
    """One farmer's model inputs, or None if there is no such farmer."""
    return db.execute(_FARMER_FEATURES_SQL, {"farmer_id": farmer_id}).fetchone()


def dominant_crop(crops: Any, default: str = DEFAULT_CROP) -> str:
    """
    The crop a field is treated as growing.

    A list (what `array_agg` returns) or a comma-separated string are both accepted — the latter
    only so a caller holding an older shape still resolves rather than silently defaulting. Falls
    back to [DEFAULT_CROP] when nothing is recorded, which is a real assumption and not a neutral
    one: it makes an uncaptured field look like a maize field to the model.
    """
    if isinstance(crops, (list, tuple)) and crops:
        first = str(crops[0]).strip()
        if first:
            return first
    if isinstance(crops, str) and crops.strip():
        return crops.split(",")[0].strip()
    return default


def list_fields(db, limit: int | None = None) -> list[dict]:
    """
    Registered fields with usable coordinates, for the nightly precompute.

    Coordinates come from `farms`. They were read from `farmer_profiles` before, where they are
    always NULL — so the filter matched no rows and the job silently precomputed nothing at all.
    """
    sql = _FIELDS_SQL + ("LIMIT :limit" if limit else "")
    rows = db.execute(text(sql), {"limit": limit} if limit else {}).fetchall()

    return [
        {
            "field_id": r.field_id,
            "latitude": float(r.latitude),
            "longitude": float(r.longitude),
            "crop": dominant_crop(r.crops),
        }
        for r in rows
    ]
