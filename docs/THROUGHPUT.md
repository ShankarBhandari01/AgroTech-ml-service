# Throughput design review — the ML service

Scope: `argotech.serving` as deployed today, one container behind a Kotlin backend that is the only
caller. Report only; no code changed. Every claim below is anchored to a file and line so it can be
checked rather than believed.

## Summary

The architecture is right. The **implementation serialises**, and it does so in three places that
are individually cheap to fix. The single most valuable structural idea in the service —
precomputing features off the request path — is already built and is where the throughput ceiling
actually lives.

**Ranked by throughput gained per unit of work:**

| # | Finding | Cost to fix | Expected effect |
| --- | --- | --- | --- |
| 1 | Blocking DB calls on the event loop | ~3 lines | Removes 2 serialisation points **per request** |
| 2 | `uvicorn` runs a single worker | 1 line | ~N× on an N-core VM |
| 3 | No HTTP connection reuse to upstreams | ~10 lines | Removes a TLS handshake per upstream call |
| 4 | Untuned SQLAlchemy pool | 1 line | Raises the real concurrency ceiling once 1–2 land |
| 5 | Audit write on the response path | design change | Removes a DB round trip from p99 |
| 6 | No batch endpoint | new route | Amortises per-request overhead for the ranking use case |

**Nothing here should be done before measuring.** There is not a single latency or throughput number
committed anywhere in this repository. Every figure below is a mechanism, not a measurement.

---

## 1. Blocking database calls on the event loop — the worst one

`serving/pipeline.py:258` and `:335` call synchronous SQLAlchemy **directly inside `async def`**:

```python
cached = store.read_latest_features(self.db, field_id)      # :258, every request
prediction_id = store.write_prediction(self.db, ...)        # :335, every request
```

In an async handler, a synchronous call holds the event loop for its full duration. With one uvicorn
worker that means **every concurrent request waits on it** — the service degrades to serial
execution under load, no matter how fast Postgres is.

What makes this clearly a defect rather than a choice: the same file already does it correctly three
lines away. `fetch_farmer_features` is wrapped (`:224`), as is `model.predict` (`:290`) and every
upstream fetch (`:134-137`). Two calls were missed.

**Fix:** wrap both in `run_in_threadpool`, exactly as the neighbours are. Roughly three lines.

## 2. One uvicorn worker

`Dockerfile:37` and `Procfile:1` both start uvicorn with no `--workers`. That is one process and one
event loop — on a multi-core VM, every core but one is idle.

**Fix:** `--workers N`. But **order matters**: adding workers before fixing (1) multiplies the number
of independently-serialising loops rather than removing the serialisation. Fix (1) first, then size
workers to cores, then re-measure.

Note the interaction with memory: each worker loads its own copy of the model artifact via
`serving/container.py`'s module-level singleton. Size the VM accordingly, or move to a pre-fork
model where the artifact is loaded before the fork.

## 3. No HTTP connection reuse

`data/meteo.py:58,90` and `data/sentinel.py:135,232` use bare `requests.get`/`requests.post`. Each
call opens a new TCP connection and performs a **full TLS handshake**. Against CDSE — which also
requires a token exchange against `identity.dataspace.copernicus.eu` — that is real, repeated
latency on every live request.

**Fix:** a module-level `requests.Session` per upstream, with an `HTTPAdapter` sized to the worker
count. Keep the existing timeouts and the 429 backoff; this changes the transport, not the retry
policy.

**Caveat that limits the value:** this only helps requests that miss the precompute cache (§6). If
the hit rate is high, this is a small win — which is exactly why it is ranked third and not first.

## 4. Untuned connection pool

`data/db.py:11` is `create_engine(DATABASE_URL)` with no arguments — SQLAlchemy's default
`QueuePool`, `pool_size=5`, `max_overflow=10`. Once (1) is fixed and DB work moves to the
threadpool, **the pool becomes the true concurrency limit**: 15 connections shared across all
workers' threadpools.

**Fix:** set `pool_size` and `max_overflow` deliberately against `workers × threadpool_size`, and
add `pool_pre_ping=True` — a long-lived container against a Postgres that recycles connections will
otherwise serve stale-connection errors after an idle period.

## 5. The audit write is on the response path

`pipeline.py:335` writes the `predictions` row **before** returning, and the comment explains why:
the response carries `prediction_id` so the agent app can link an outcome back to it. That is a
genuine product requirement, not an oversight — the audit trail is what turns advisories into
training data, and this project's central finding is that it has no outcome labels.

So the round trip cannot simply be deleted. Options, in increasing order of change:

- **Offload it** (§1) so it does not block the loop. Cheapest, keeps semantics identical.
- **Generate the id client-side** (UUID) and write asynchronously after the response. Removes the
  write from p99 entirely; costs you the guarantee that a returned id is durable.
- **Batch writes** behind a queue. Highest throughput, most machinery, and it weakens the
  "persistence never fails a prediction" property the service currently holds.

I would do the first and stop, until measurement says otherwise.

## 6. The precompute path is the real ceiling — and it is already built

`jobs/precompute.py` runs nightly and writes one `field_features` row per registered field;
`pipeline.py:258` reads it and only falls back to live upstream fetches when no fresh row exists.
This is the correct architecture, and it was built for the right reason: every prediction previously
made three blocking calls to two free public APIs, and one training backfill exhausted Open-Meteo's
daily quota.

**Therefore the dominant throughput variable is the cache hit rate, and it is not instrumented.**
A precomputed request touches Postgres and the model. A live request adds four upstream calls with
TLS handshakes and a rainfall climatology that "took seconds on a cache miss".

**The highest-value observability work is not a latency histogram. It is the ratio
`feature_source=precomputed` to `feature_source=live`.** That field is already recorded on every
audit row (`pipeline.py:337`), so the number can be computed today from the `predictions` table with
no code change at all. Do that before optimising anything in §3.

Related: `jobs/precompute.py` already reports a `degraded_rate` — the share of fields with no
cloud-free scene. That is a quality metric masquerading as an ops metric and it belongs on the same
dashboard.

## 7. Batch endpoint — the shape of the caller's actual problem

The service exposes per-farmer prediction. The product question is *"which 50 farmers should an
agent visit this week"* — a ranking over many fields, not one lookup. If the Kotlin backend fans out
N single calls, it pays N× the per-request overhead: N HTTP round trips, N DB sessions, N model
invocations of one row each.

A `POST /predict/batch` taking a list of field ids would let the service do one query, **one
vectorised `model.predict` over a DataFrame of N rows** instead of N calls with one row each, and
one audit insert. Scikit-learn's per-call overhead is largely fixed, so this is close to free
throughput for the ranking path.

This is the largest structural win available, and it is also the one requiring coordination with the
Kotlin team — so it belongs on a roadmap, not in a hotfix.

## 8. What is already correct, and should not be "optimised"

- The model artifact is loaded **once** as a module-level singleton (`serving/container.py`,
  `serving/deps.py`) and injected. Not per request.
- `model.predict` is already offloaded (`pipeline.py:290`).
- All four upstream fetches are already offloaded (`pipeline.py:134-137`).
- Auth, rate limiting, circuit breaking, retry and bulkheading are the **Kotlin backend's**
  (Resilience4j). Do not reimplement them here; a second circuit breaker in the callee makes failure
  behaviour harder to reason about, not safer.
- Graceful degradation is deliberate: a missing satellite scene yields a prediction with a named
  reason rather than a 500. Do not "fix" that into an error path.

## Recommended sequence

1. **Measure.** Compute `precomputed` vs `live` from the `predictions` table — no code change
   needed. Add request duration by `feature_source`.
2. **Fix the two blocking DB calls.** Small, safe, strictly correct.
3. **Add workers**, sized to cores, and re-measure. Watch memory: one artifact per worker.
4. **Tune the pool** against `workers × threadpool`, add `pool_pre_ping`.
5. **Then** decide whether §3 (session reuse) and §5 (async audit) are worth it — driven by the hit
   rate from step 1, not by this document.
6. **Batch endpoint** as a roadmap item with the Kotlin team.

Steps 2–4 are perhaps twenty lines. Step 1 is the one that tells you whether any of the rest matters.
