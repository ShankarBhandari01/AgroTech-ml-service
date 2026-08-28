# Cut the backend-schema coupling — the backend passes the payload

**Goal:** the ML service stops reading the Kotlin backend's tables on the request path, so a backend
migration can no longer break inference silently.

**Why this matters, with precedent.** `data/backend_schema.py` reads `farmer_profiles`, `farms`,
`farm_crops`, `crops` and `farmers_ml_profiles` — tables another service owns and migrates. This has
already failed once: when the backend's V47 replaced the `farms.crops` text array with a
`farm_crops` join table, one copy of the query failed outright and another silently fell back to a
hardcoded crop. A service that reaches into another service's schema is coupled to its migration
schedule, and the failure is silent by construction.

**Current state, measured.** Kotlin's `PythonFarmerPayload` carries only
`{farmer_id, model_name, model_alias}` — an id plus two dead fields (Python's `request.py` records
that no loader has read them since the registry rewrite). Python's `FarmerPredictionRequest` accepts
only `farmer_id`, then queries ~22 fields out of the backend's schema.

**This is a cross-repository change.** Only the Python half is in scope here, and it must be
**backward compatible**: the deployed backend sends an id and nothing else, so a breaking change
takes down production. Expand → migrate → contract.

## Global Constraints
- No new runtime dependency. Ruff clean. Reproduction gate unchanged.
- **The deployed contract must keep working unchanged** at every commit.
- `domain/` untouched.
- Do not change `jobs/precompute.py`'s behaviour — see the open question below.

---

### Task 1 — expand: accept an optional payload, keep the fallback

- [ ] Extend `FarmerPredictionRequest` with an optional `farmer` object carrying what
      `fetch_farmer_features` returns today. Read `data/backend_schema.py`'s `_FARMER_FEATURES_SQL`
      for the exact field list — do not invent it. It includes `farm_size`, `state`, `latitude`,
      `longitude`, `crops`, and the `farmers_ml_profiles` block.
- [ ] `pipeline._fetch_farmer_data` uses the payload when present and falls back to the DB read when
      absent. **Log which path served**, at the same place `feature_source` is already recorded, so
      the cutover is measurable rather than assumed.
- [ ] Every field optional with an explicit default, and the defaults must be the same ones the
      vulnerability model already applies to a NULL column — a payload missing a field must not
      produce a different answer from a database NULL. Check `domain/risk.py`'s handling before
      choosing.
- [ ] **The protected-attribute guard still applies.** `head_gender` and
      `household_max_education` may arrive in the payload for fairness *measurement* and must still
      raise if used as vulnerability inputs. Add a test proving the guard fires on the payload path,
      not only the DB path.
- [ ] Tests: payload path and DB path produce **identical** predictions for the same farmer. That
      equality is the whole safety argument for the cutover.
- [ ] Full suite, ruff, commit.

### Task 2 — the contract, written down

- [ ] Document the payload in `docs/` as the ML service's inbound contract: every field, its type,
      its default, and which backend table it came from. The Kotlin team implements against this.
- [ ] State the deprecation explicitly: the `farmer_id`-only form is supported until the backend
      ships the payload, then removed. Name what triggers removal — the log ratio from Task 1
      reaching zero.
- [ ] Commit.

### Task 3 — contract (LATER, do not do now)

Only once the backend ships and the Task 1 log shows zero DB-path requests:
- [ ] Delete `fetch_farmer_features` and the `_FARMER_FEATURES_SQL` it wraps.
- [ ] Reduce the ML service's database grants to its own tables.

---

## Open question for the Kotlin team — the batch path

`jobs/precompute.py` calls `backend_schema.list_fields`, which enumerates **every** farm with
coordinates. A per-request payload cannot replace an enumeration, so this coupling survives Task 1.
Three options, in the order I would prefer them:

1. **An internal, paged endpoint on the backend** (`GET /internal/ml/fields`) that the nightly job
   pulls. Cleanest: the backend keeps ownership of its schema and decides what to expose. Costs one
   endpoint.
2. **The backend enqueues the job with the field list**, via the `MlTaskQueue` that already exists.
   Fits the existing architecture, but a nightly payload of every field is large and the queue was
   built for tasks, not bulk data.
3. **A read-only database view** scoped to exactly `(field_id, latitude, longitude, crop)`. Narrowest
   possible coupling, and it survives most migrations because the backend controls the view. Least
   pure, most pragmatic.

I would take (1). Until it is decided, `precompute` keeps its current read — cutting the request
path is worth doing on its own and does not depend on this.

## What this does NOT change

The ML service keeps its own tables — `field_features`, `predictions`, `field_outcomes` — and its
own database access for them. Those are ML-owned state, not the backend's, and the audit trail they
carry is what turns advisories into training data.
