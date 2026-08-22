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

from argotech.data import backend_schema, store
from argotech.data.db import SessionLocal
from argotech.serving.container import model_manager
from argotech.serving.pipeline import gather_upstream

# Upstream courtesy: both APIs are free and rate-limited, and this job has all night. Sequential
# with a pause is deliberate — parallelism here is what triggered the 429s during training.
DELAY_SECONDS = 1.5


# `list_fields` moved to data/backend_schema.py, next to the same query the prediction endpoint runs.
# Its version read coordinates from `farmer_profiles`, where nothing has written them since plot data
# moved to `farms` — so the filter matched no rows and this job precomputed nothing at all, quietly.
list_fields = backend_schema.list_fields


async def run(limit: int | None = None, delay: float = DELAY_SECONDS) -> dict:
    db = SessionLocal()
    try:
        store.ensure_schema(db)
        fields = list_fields(db, limit)
        print(f"Precomputing features for {len(fields)} fields ...")

        # The peer reference is a fitted statistic carried in the artifact, so the nightly job needs
        # it too: a stored row whose `ndvi_z_peer` was standardised against anything else is the
        # skew this replaced, written to the feature table instead of computed in a request.
        _, _, _, bounds, _, peer_ref = model_manager.agronomic_model()

        ok = failed = 0
        degraded = 0     # succeeded, but with no cloud-free Sentinel-2 scene
        for i, f in enumerate(fields, 1):
            try:
                row, ctx = await gather_upstream(f["latitude"], f["longitude"], f["crop"],
                                                 bounds, peer_ref)
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
