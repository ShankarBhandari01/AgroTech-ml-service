# How the Python side should be designed, given the Kotlin backend

Report only. Written after reading both repositories, not from a template.

## The problem, stated plainly

The Python service is currently trying to be **two things with opposite requirements**:

| | inference | research |
| --- | --- | --- |
| dependencies | as few as possible | torch, mlflow, jupyter |
| startup | fast, every deploy | irrelevant |
| stability | the contract must not move | churns daily, by design |
| failure | pages someone | expected, informative |
| lifetime | milliseconds | hours |

You said it yourself: no robust model yet, lots of EDA and experiments ahead. That means **the
research surface will churn hard for months** while the inference contract must stay still. A single
deployable that is both cannot do both — every experiment becomes a production risk, and production
stability becomes a brake on experiments.

## What already exists, and is right

Both sides have more of the right structure than a greenfield design would give you:

**Kotlin** — `MlTaskQueue` is a broker abstraction with the implementation held behind it
(`RedissonMlTaskQueue`), explicitly so Redis can become Kafka or SQS without touching callers.
`FastApiMlClient` exposes `suspend` functions with Resilience4j around them, and
`MlServiceUnavailableException` means ML failure is already a modelled outcome rather than a 500.

**Python** — `domain/` is pure (no I/O, no DB, no network) and imported by both the training and
serving paths, which is what stops a feature meaning two different things at train and serve time.
`lab/export.py` produces a versioned artifact; `models/registry.py` validates its contract at load.
`jobs/precompute.py` already moves feature computation off the request path.

**The single most important consequence:** you already have an artifact-as-contract and a queue.
The design below is mostly *naming what exists and enforcing it*, not building something new.

## The design: three surfaces, one contract

```
RESEARCH  (never deployed)          BATCH  (scheduled)              INFERENCE  (always up)
  lab/, notebooks/, experiments/      jobs/precompute.py              serving/
  torch, mlflow, jupyter              panel builds, retraining        fastapi + sklearn only
  churns daily                        minutes to hours                 milliseconds
        │                                     │                              ▲
        └──────── produces ──────────►  model artifact  ────── loads ────────┘
                                    (versioned + feature schema)
```

**The contract between research and production is the artifact, and nothing else.** Not a shared
process, not an import, not a database table. `lab/export.py` writes it, `models/registry.py`
refuses to load one whose feature columns serving cannot build. That refusal is the whole point: it
is why an experiment cannot silently become a production incident.

### Rule 1 — research code must be unable to reach production

Not "should not". *Unable.* Enforced two ways: a packaging exclusion so the image cannot contain
`lab/`, and an import-boundary test so the violation fails on a laptop rather than in a container.
This is in flight now.

The direction that stays open is `lab → domain`. That is load-bearing, not a leak: one feature
builder used by both paths is what prevents train/serve skew. This project has already been bitten —
`rainfall_anomaly` once meant `six_month_rain − 450` in training and `eight_day_rain` at serving,
so every served row landed in a region of feature space the model had never seen.

### Rule 2 — the Kotlin queue owns anything slow

`MlTaskQueue` already exists. Use it as the boundary of what Python may be asked to do
synchronously:

- **Synchronous HTTP** → bounded, fast, one field. `/predict/farmer` reading a precomputed feature
  row. Target: single-digit milliseconds plus a model call.
- **Queued job** → anything unbounded. Batch scoring a district, a backfill, a retrain trigger,
  anything that touches an upstream API.

The failure mode to design out is a long-running ML task behind a synchronous call. Resilience4j
will trip, the caller retries, the ML service is now doing the work twice, and the queue you already
have was the answer.

### Rule 3 — batch is a first-class surface, not a script

`jobs/precompute.py` is the highest-leverage code in the repository and it reads like an
afterthought. It is what makes a prediction fast, bounds upstream call rate to one pass per field
per night instead of unbounded fan-out driven by traffic, and it already reports `degraded_rate` —
the share of fields with no cloud-free scene, which is a data-quality alarm nothing is watching.

Batch deserves its own deployment, schedule, alerting and dashboard. Same image as inference is
fine (it shares `domain/` and `features/`); same *process* is not.

## What I would change, in order

1. **Finish the serving/lab split.** In flight. It is the precondition for everything else.
2. **Instrument the precompute hit rate.** `feature_source` is already written to every audit row,
   so the ratio is computable from the `predictions` table today with no code change. This number
   determines whether any latency work is worth doing — see `docs/THROUGHPUT.md`.
3. **Move batch to its own scheduled deployment** with its own alerting on `degraded_rate`.
4. **Add a batch prediction endpoint**, and have the Kotlin side call it through the queue for
   ranking. The product question is "which 50 farmers this week" — a ranking over many fields — but
   the API is one-field-at-a-time. One vectorised `model.predict` over N rows beats N calls of one.
5. **Only then** consider splitting repositories. Not before: two repos means two copies of
   `domain/`, or publishing it as a library, and a duplicated domain layer is precisely how
   train/serve skew comes back.

## What not to do

- **Do not put research behind an HTTP endpoint.** "Run this experiment via the API" turns every
  exploratory mistake into an outage.
- **Do not reimplement resilience in Python.** Auth, tenancy, rate limiting, circuit breaking and
  retry are the Kotlin backend's, and it already has them. A second circuit breaker in the callee
  makes failure behaviour harder to reason about, not safer.
- **Do not let the ML service own product state.** It owns `field_features`, `predictions` and
  `field_outcomes` — its own audit trail. Everything about farmers, farms and crops belongs to the
  backend. Note the ML service currently *reads the backend's tables directly*
  (`data/backend_schema.py`), which means a backend migration can break inference silently. It has
  already happened once, when `farms.crops` became a `farm_crops` join table. **The backend should
  pass the farmer payload rather than the ML service reaching into its schema** — the DTO
  (`PythonFarmerPayload`) already exists on the Kotlin side.
- **Do not chase a model before the labels exist.** The finding from this investigation is that at
  four independent spatial units, with a target whose predictable signal is essentially all
  field-effect, no configuration demonstrates out-of-region skill. More modelling will not fix that.
  Outcome capture will — `POST /outcomes` and `store.label_join` are built and, until recently,
  had never been joined.

## The honest summary

The architecture is already close to right. What is missing is **enforcement** — boundaries that a
tired person cannot cross by accident — and **measurement**, because there is not a single latency,
throughput or cache-hit number committed anywhere in either repository.

Research churn is the expected state for the next several months. Design for it: let the lab be
messy, keep the artifact contract narrow and validated, and let the queue absorb anything slow.
