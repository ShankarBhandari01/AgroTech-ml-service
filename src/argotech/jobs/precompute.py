"""Nightly feature precomputation.

Moves the expensive part of a prediction off the request path. Every registered field gets its ERA5
window, Sentinel-2 history, leaf wetness and rainfall climatology computed once here and written to
`field_features`; a prediction then costs a database read plus a model call.

The old shape made three blocking calls to two free public APIs inside every request, one of which
(the 4-year rainfall climatology) took seconds on a cache miss. A single training backfill was
enough to exhaust Open-Meteo's daily quota — at farmer volume that is a daily outage. Batching also
bounds the upstream call rate to something predictable: one pass per field per night, retried
tomorrow if it fails, instead of unbounded fan-out driven by traffic.

Run: `python -m argotech.jobs.precompute` (cron, nightly, single instance)
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

from argotech.data import store
from argotech.data.db import SessionLocal
from argotech.serving.pipeline import gather_upstream

# Upstream courtesy: both APIs are free and rate-limited, and this job has all night. Sequential
# with a pause is deliberate — parallelism here is what triggered the 429s during training.
DELAY_SECONDS = 1.5


def list_fields(db, limit: int | None = None) -> list[dict]:
    """Registered fields with usable coordinates, most recently updated first."""
    rows = db.execute(text(f"""
        SELECT fp.user_id AS field_id, fp.latitude, fp.longitude, fp.crops
        FROM farmer_profiles fp
        WHERE fp.latitude IS NOT NULL AND fp.longitude IS NOT NULL
        ORDER BY fp.user_id
        {'LIMIT :limit' if limit else ''}
    """), {"limit": limit} if limit else {}).fetchall()

    fields = []
    for r in rows:
        crops = r.crops
        if isinstance(crops, list) and crops:
            crop = str(crops[0]).strip()
        elif isinstance(crops, str) and crops.strip():
            crop = crops.split(",")[0].strip()
        else:
            crop = "Maize"
        fields.append({"field_id": r.field_id, "latitude": float(r.latitude),
                       "longitude": float(r.longitude), "crop": crop})
    return fields


async def run(limit: int | None = None, delay: float = DELAY_SECONDS) -> dict:
    db = SessionLocal()
    try:
        store.ensure_schema(db)
        fields = list_fields(db, limit)
        print(f"Precomputing features for {len(fields)} fields ...")

        ok = failed = 0
        degraded = 0     # succeeded, but with no cloud-free Sentinel-2 scene
        for i, f in enumerate(fields, 1):
            try:
                row, ctx = await gather_upstream(f["latitude"], f["longitude"], f["crop"])
                store.write_features(db, f["field_id"], f["latitude"], f["longitude"],
                                     f["crop"], row, ctx)
                ok += 1
                if ctx["index_source"] != "sentinel-2":
                    degraded += 1
            except Exception as e:  # noqa: BLE001 — one bad field must not end the run
                failed += 1
                print(f"  [{i}/{len(fields)}] {f['field_id']}: {e}")
            if i % 50 == 0:
                print(f"  {i}/{len(fields)} — ok {ok}, degraded {degraded}, failed {failed}")
            await asyncio.sleep(delay)

        pruned = store.prune_features(db)
        # `degraded` is the quality metric worth alerting on: it is the share of the farmer base for
        # which the canopy signal — and therefore the learned model — is unavailable tonight.
        summary = {"fields": len(fields), "ok": ok, "degraded": degraded,
                   "failed": failed, "pruned": pruned,
                   "degraded_rate": round(degraded / ok, 3) if ok else None}
        print(f"Done: {summary}")
        return summary
    finally:
        db.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only the first N fields")
    ap.add_argument("--delay", type=float, default=DELAY_SECONDS)
    args = ap.parse_args()
    asyncio.run(run(args.limit, args.delay))


if __name__ == "__main__":
    main()
