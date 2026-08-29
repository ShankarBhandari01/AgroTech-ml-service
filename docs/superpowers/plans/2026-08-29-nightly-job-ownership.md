# The nightly job: Kotlin schedules and supplies, Python executes

**Goal:** put the most important scheduled job in the system under the same supervision as
everything else, and cut the last read of the backend's schema.

## Why

The nightly precompute is what makes every prediction fast and bounds upstream API calls to one pass
per field per night instead of unbounded fan-out driven by traffic. It is triggered by **a line in a
crontab on the VM**:

```
0 2 * * *  python -m argotech.jobs.precompute
```

Meanwhile the Kotlin backend properly schedules two far less important jobs through Spring:
`MlHealthMonitorCronService` (`fixedRate = 300000`) and `StrandedPredictionReaper`
(`fixedDelayString = ${ml.stranded-sweep-ms:600000}`).

So the job that matters most is the one nothing watches: it is not in the deployment, a redeploy
does not update it, nothing alerts if it stops, and the `degraded_rate` it already computes — the
share of fields with no cloud-free scene, which is a data-quality alarm — goes nowhere.

It is also where the **last schema coupling** lives. `jobs/precompute.py` calls
`backend_schema.list_fields`, enumerating every farm with coordinates. The farmer-payload change
could not fix this, because a per-request payload cannot enumerate.

Both problems have one answer.

## Design

```
Kotlin @Scheduled (nightly)
  -> loads its own field list              <- the coupling is cut here
  -> MlTaskQueue, in batches               <- the queue already exists
    -> PredictionQueueConsumer
      -> POST /precompute/batch            <- new endpoint, fields supplied not queried
```

**Pacing must survive, and it belongs on the Python side.** The current job is deliberately
sequential with delays because both upstreams are free and rate-limited — a single training backfill
once exhausted Open-Meteo's daily quota. Python is the side that knows which upstream it is about to
hit and already carries the retry/backoff. Kotlin must therefore **not** fan out: it enqueues
batches and lets Python pace within one.

## Global Constraints
- **Runs after** the `lab/` regrouping lands (it touches `jobs/precompute.py`, which imports from
  `serving/`), and its Python half overlaps the queued `core/` plan's Task 2 — check both first.
- Backward compatible in both directions. The crontab keeps working until the new path is proven.
- No new runtime dependency. Reproduction gate unchanged. Boundary test must pass.

---

### Task 1 (Python) — `POST /precompute/batch`
- [ ] Accepts a list of fields (`field_id`, `latitude`, `longitude`, `crop`) — the same shape
      `list_fields` returns today, so the contract is already defined by the existing query.
- [ ] Refactor `jobs/precompute.run` so the field list is a **parameter**, not something it queries.
      Keep `list_fields` as the default for the crontab path until Task 4 retires it.
- [ ] **Preserve the pacing**: the existing `DELAY_SECONDS` between fields must apply within a batch.
      Add a test that a batch of N fields takes at least (N−1)×delay — the one property that protects
      the free-tier quota, and the one a refactor silently drops.
- [ ] Keep `degraded_rate` in the response so the caller can surface it.
- [ ] Suite, ruff, gate, commit.

### Task 2 (Kotlin, in the `agri-ml-payload` worktree) — schedule and supply
- [ ] A `@Scheduled` nightly service alongside the two existing ones, following their pattern
      (externalised cron expression, same logging and error handling).
- [ ] Loads the field list from its **own** repositories and enqueues it through `MlTaskQueue` in
      batches — never one job per field, which would defeat the pacing.
- [ ] Consumer calls `POST /precompute/batch` and records `degraded_rate`.
- [ ] Test: a scheduled run enqueues the expected batches; an empty field list enqueues nothing
      rather than an empty job.
- [ ] Build and test via the wrapper JAR (**there is no tracked `gradlew`** — jar and properties
      only). Commit; do not push.

### Task 3 — make the alarm visible
- [ ] `degraded_rate` surfaces wherever the other ML health metrics go. It is the only near-real-time
      signal that the model has silently lost its inputs — labels arrive months late, so nothing else
      tells you.
- [ ] Commit.

### Task 4 (LATER — only when the new path is proven in production)
- [ ] Delete the crontab entry.
- [ ] Remove `list_fields` and `_FIELDS_SQL` from `data/backend_schema.py`.
- [ ] With `fetch_farmer_features` already gone by then, **`backend_schema.py` disappears entirely**
      and the ML service stops reading the backend's schema at all — which is the point of the whole
      exercise.

## Not in scope
- Moving the batch to its own deployment. Worth doing, separate concern.
- Changing what precompute computes. This is about who triggers it and where the field list
  comes from.
