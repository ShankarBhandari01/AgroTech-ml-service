# `core/` and a service layer — adopt the standard layout where it fits

> Runs AFTER the serving/lab split completes (Tasks 4–5 outstanding). Touches files the split edits.

**Goal:** adopt the conventional FastAPI production layout for the serving surface, without losing
the two directories that layout has no slot for and this project cannot lose.

## The target, and what maps to what

| Standard layout | Here | |
| --- | --- | --- |
| `src/api/` | `serving/api/` | already correct |
| `src/services/` | `serving/services/` | **rename** from `serving/pipeline.py` |
| `src/models/` | `models/` | already correct (registry, artifact loading) |
| `src/core/` | `core/` | **new** — config + logging |
| `src/utils/` | — | **deliberately not created**, see below |
| — | `domain/` | **kept**: pure agronomy, no I/O, imported by serving *and* lab |
| — | `lab/` | **kept**: research surface, never deployed |

`domain/` is what stops a feature meaning two different things at train and serve time — this repo
has already been bitten (`rainfall_anomaly` was `six_month_rain − 450` in training and
`eight_day_rain` at serving, so every served row landed where the model had never seen). No name in
the standard layout preserves that property.

No `utils/`: every helper here already belongs to a named layer, and a directory named for what it
is *not* becomes a junk drawer.

## Global Constraints
- No new runtime dependency. `[project.dependencies]` byte-identical.
- Ruff clean. Reproduction gate prints `0.345 [0.2357, 0.4353] 0.0`.
- **`tests/test_import_boundary.py` must keep passing at every commit** — it is the layering guard.
- `git mv` for every move, so history survives.
- Baseline at start: whatever the split leaves. Record it first.

---

### Task 1 — `core/`
The directory is justified by **logging**, not config. `logging.basicConfig` and the
`argotech.access` logger are configured inside `serving/main.py`, so `jobs/precompute.py` and
`lab/run.py` cannot share them and `print()` instead — which is why experiment output has been
unparseable.

- [ ] `git mv src/argotech/config.py src/argotech/core/config.py`; add `core/__init__.py`.
- [ ] `core/logging.py`: the setup currently inline in `serving/main.py`, plus `get_logger(name)`.
      **Keep the `argotech.access` logger's exact name and format** — log parsers and the OTEL
      exporter key on them.
- [ ] Repoint ~10 importers of `argotech.config`. Grep; do not guess.
- [ ] Extend `test_import_boundary.py`: importing `core.*` must pull in **no** other `argotech`
      module. That is what makes `core` the bottom of the graph rather than a second `utils`.
      **Prove it bites** — add `from argotech.data import db` to `core/config.py`, watch it fail,
      remove it, report both outputs.
- [ ] Suite, ruff, gate, commit.

### Task 2 — move shared feature assembly out of `serving/`
Measured: `jobs/precompute.py:23-24` imports `serving.container` and `serving.pipeline.gather_upstream`.
The batch job depending on the serving module is backwards — `gather_upstream` is feature assembly,
which is `features/`' job. Sharing it is deliberate and must be preserved: it is what makes a
precomputed prediction and a live one the same computation, verified by an existing parity check.

- [ ] Move `gather_upstream` (and anything it needs that is not serving-specific) into `features/`.
- [ ] `jobs/precompute.py` and `serving/` both import it from there. `jobs/` should then import
      nothing from `serving/` except the model-manager singleton — if that too can move to
      `models/`, do it; if not, say why.
- [ ] The parity check must still pass. Name it in your report.
- [ ] Suite, ruff, commit.

### Task 3 — `serving/services/`
- [ ] `git mv src/argotech/serving/pipeline.py src/argotech/serving/services/prediction.py`
      (add `services/__init__.py`). 516 lines; **12 files import it** — repoint every one.
- [ ] Routes in `serving/api/` should read as thin delegation to the service. If the move exposes
      logic sitting in a route that belongs in the service, note it — **do not fix it in this
      commit**; a rename and a refactor in one diff is unreviewable.
- [ ] Suite, ruff, gate, commit.

### Task 4 — write it down
- [ ] `README.md` and `docs/model-design.md` §7 carry the layout table above, including why
      `domain/` and `lab/` exist and why `utils/` does not.
- [ ] Commit.

## Explicitly not in this plan
- No behaviour change. Every task is a move plus repointed imports.
- No `services/` at the top level: `model-design.md` §7 records this branch moving *away* from
  `src/services/inferenceService/app/...` because it took five levels to reach a module. The service
  layer belongs **inside** `serving/`, where it is one level deep.
