# forward_z Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve a model that regresses `forward_z` directly instead of classifying its three-way discretisation, closing the ranking gap to the persistence baseline.

**Architecture:** `HistGradientBoostingRegressor` replaces `CalibratedClassifierCV` as the production arm. The predicted z feeds the *existing* `risk.vegetation_hazard_from_anomaly`, so both hazard sources use identical arithmetic and differ only in where z comes from. An explicit `target` key in the artifact bundle makes a classifier/regressor mismatch fail loudly instead of silently returning near-zero hazard for every field.

**Tech Stack:** Python 3.12, scikit-learn 1.6.1 (pinned — the artifact is a joblib pickle), pandas, FastAPI, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-21-forward-z-regression-design.md`

## Global Constraints

- `SEVERE_Z = -1.0`, `ELEVATED_Z = -0.35` (`training/dataset.py:66`) — the label cuts. Do not change.
- `SEVERE_ANOMALY_Z = -1.0`, `ANOMALY_SOFTNESS = 0.5` (`domain/risk.py:38,44`) — the hazard logistic. Do not change; the whole design rests on it already matching `SEVERE_Z`.
- The response schema does not change. `probabilities` stays `Map<String, Double>` — a string value inside that object breaks the Kotlin client's deserialisation and sends every prediction down its FALLBACK path.
- scikit-learn stays pinned at 1.6.1. Retraining and unpinning are one change, never one alone.
- `ruff check src tests` must pass clean; the repo is at zero findings.
- Run tests with `./venv/bin/python -m pytest`, not bare `pytest`.

### Deviation from the spec, accepted

The spec sketches `ARMS` as (factory, target, risk-lambda) triples. This plan uses **2-tuples** `(factory, target)` and branches on `target` inside `evaluate_fold`, because the fit call already has to branch (different target column, different NaN filtering) and a lambda in a module-level dict would hide that. Same behaviour, less indirection.

### Task ordering note

Task 2 is deliberately larger than Tasks 1 and 3. The artifact and the serving code **cannot** move independently: a regressor artifact with `predict_proba` serving code raises `AttributeError`, and a classifier artifact with `predict` serving code returns class labels read as z values. Task 2 is the atomic switch. Tasks 1 and 3 leave the tree green on their own.

---

### Task 1: Add the regressor arm to training

**Files:**
- Modify: `src/argotech/training/train.py` (imports, `make_regressor`, `ARMS`, `evaluate_fold`, final fit, bundle)
- Test: `tests/test_train_arms.py` (create)

**Interfaces:**
- Consumes: `MODEL_FEATURES`, `SEED`, `expected_severity`, `decide`, `_score` — all existing in `train.py`.
- Produces: `make_regressor(seed: int = 42) -> HistGradientBoostingRegressor`; `ARMS: dict[str, tuple[Callable, str]]` mapping arm name to (factory, target column); `PRODUCTION_ARM: str = "hgbr"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_arms.py`:

```python
"""The regressor arm's wiring: hyperparameters matched to the classifier's base estimator, and
the arm table shaped so `evaluate_fold` knows which target each arm is fitted against."""

from __future__ import annotations

import numpy as np


def test_make_regressor_matches_the_classifier_base_hyperparameters():
    """Both arms must differ only in what they predict, or the comparison measures the wrong thing."""
    from argotech.training.train import make_model, make_regressor

    reg = make_regressor(42)
    base = make_model(42).estimator

    for param in ("max_iter", "learning_rate", "max_depth", "min_samples_leaf",
                  "l2_regularization", "early_stopping", "validation_fraction", "random_state"):
        assert getattr(reg, param) == getattr(base, param), f"{param} differs between arms"


def test_arms_declare_which_target_each_is_fitted_against():
    from argotech.training.train import ARMS, PRODUCTION_ARM

    assert PRODUCTION_ARM == "hgbr"
    assert ARMS[PRODUCTION_ARM][1] == "forward_z"
    assert ARMS["hgb"][1] == "label"
    assert ARMS["linear"][1] == "label"
    assert set(ARMS) == {"hgbr", "hgb", "linear"}, "the classifier stays as a control arm"


def test_regressor_ranks_a_worse_field_higher():
    """risk = -predicted_z, so a field predicted to fall further behind must rank above one that
    does not. Gets the sign right, which is invisible in aggregate metrics until the queue inverts."""
    from argotech.training.train import make_regressor

    x = np.arange(200, dtype=float).reshape(-1, 1)
    z = -x[:, 0] / 100.0                      # higher x -> lower z -> worse field
    fitted = make_regressor(42).fit(x, z)
    risk = -fitted.predict(np.array([[10.0], [190.0]]))
    assert risk[1] > risk[0], "the field with the lower predicted z must carry the higher risk"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_train_arms.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_regressor'`.

- [ ] **Step 3: Add `make_regressor` and reshape `ARMS`**

In `src/argotech/training/train.py`, add to the sklearn imports:

```python
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
```

Add after `make_linear` (around line 132):

```python
def make_regressor(seed: int = 42) -> HistGradientBoostingRegressor:
    """The production arm: predict `forward_z` itself rather than its three-way discretisation.

    Same hyperparameters as `make_model`'s base estimator — the two arms must differ only in what
    they are asked to predict — minus the `CalibratedClassifierCV` wrapper, because a regressor
    emits a number and there is no posterior to calibrate.

    Why this is the production arm: over 5 seeds x 6 leave-one-cluster-out folds, regressing the
    continuous target lifts Spearman rho from 0.196 to 0.342 — an improvement in *every one of the
    30 folds* — and P@25 from 0.669 to 0.741, while macro F1 is unchanged within seed noise.
    Discretising before fitting told the model that z = -0.34 and z = +6.9 were the same outcome.
    """
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.06,
        max_depth=5,
        min_samples_leaf=25,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )
```

Replace the `ARMS` definition at line 137:

```python
# (factory, target column). The target decides which column is fitted and how a risk score is
# derived; `evaluate_fold` branches on it. `hgbr` is production; the other two are controls kept
# so the metrics file keeps reporting what the change gave up rather than deleting the evidence.
ARMS = {"hgbr": (make_regressor, "forward_z"),
        "hgb": (make_model, "label"),
        "linear": (make_linear, "label")}
PRODUCTION_ARM = "hgbr"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/test_train_arms.py -v`
Expected: 3 passed.

- [ ] **Step 5: Teach `evaluate_fold` the two arm shapes**

In `evaluate_fold`, replace the arm loop:

```python
    ztr = train.forward_z.to_numpy(dtype=float)
    for arm, (factory, target) in ARMS.items():
        if target == "forward_z":
            # Fit only where the target is observed; never drop a row from *evaluation*.
            ok = np.isfinite(ztr)
            fitted = factory(SEED).fit(Xtr[ok], ztr[ok])
            proba, risk = None, -fitted.predict(Xte)
        else:
            fitted = factory(SEED).fit(Xtr, ytr)
            proba = fitted.predict_proba(Xte)
            risk = expected_severity(proba)
        scored = _score(name, yte, decide(risk, ytr), proba, risk, zte)
        for key, value in scored.items():
            if key not in ("split", "n"):
                result[f"{key}_{arm}" if arm != PRODUCTION_ARM else key] = value
        if oof is not None and arm == PRODUCTION_ARM:
            oof.append((yte, risk, persistence_risk(test), climatology_risk(test), ytr))
```

`_score` already takes `proba` optionally and omits ECE when it is `None` — correct, because calibration is undefined without a posterior.

- [ ] **Step 6: Verify the three-arm evaluation runs**

Run: `./venv/bin/python -m argotech.training.train --folds-only --seed 42`
Expected: completes; the blocked section prints `macroF1=` for the headline arm plus `_hgb` and `_linear` suffixed control columns. Headline `rho` should be about **+0.35**, not +0.19 — that is the whole point of the change, visible immediately.

- [ ] **Step 7: Commit**

```bash
git add src/argotech/training/train.py tests/test_train_arms.py
git commit -m "Add a forward_z regressor arm alongside the classifier"
```

---

### Task 2: The atomic switch — artifact, contract guard, serving

**Files:**
- Modify: `src/argotech/training/train.py` (final fit + bundle `target` key)
- Modify: `src/argotech/models/registry.py:32-43`
- Modify: `src/argotech/serving/pipeline.py:261-283` (the vegetation-hazard branch)
- Test: `tests/test_artifact_contract.py` (create)
- Regenerate: `artifacts/agronomic_risk.joblib`, `artifacts/metrics.json`

**Interfaces:**
- Consumes: `make_regressor`, `PRODUCTION_ARM` from Task 1.
- Produces: bundle key `target: "forward_z"`; `ModelManager.agronomic_model()` unchanged in return shape — `(model, feature_columns, version, cluster_bounds, cluster_stats)` — but now raises `ValueError` on a bundle whose `target` is not `"forward_z"`.

- [ ] **Step 1: Write the failing contract test**

Create `tests/test_artifact_contract.py`:

```python
"""The guard between a rollback and a silent nationwide zero.

A classifier's `.predict()` returns class labels 0/1/2. Fed to regressor-shaped serving code those
labels are read as z values, and `vegetation_hazard_from_anomaly(0..2)` returns 0.002-0.119 — so
every field in the country reads "almost no canopy hazard" with no exception and no log line.
`artifact_path`'s servability guard cannot catch it: the feature set is identical. Only an explicit
contract key can.
"""

from __future__ import annotations

import joblib
import pytest


def _bundle(**overrides) -> dict:
    bundle = {
        "model": object(),
        "feature_columns": ["a", "b"],
        "version": "agro-test",
        "target": "forward_z",
        "n_samples": 10,
        "trained_on": "2024-01-01..2024-02-01",
    }
    bundle.update(overrides)
    return bundle


@pytest.mark.parametrize("bad", [{"target": "label"}, {"target": None}])
def test_an_artifact_with_the_wrong_target_is_refused(tmp_path, monkeypatch, bad):
    from argotech.config import settings
    from argotech.models.registry import ModelManager

    path = tmp_path / "wrong.joblib"
    joblib.dump(_bundle(**bad), path)
    monkeypatch.setattr(settings, "AGRONOMIC_MODEL_PATH", str(path))

    with pytest.raises(ValueError, match="forward_z"):
        ModelManager().agronomic_model()


def test_an_artifact_missing_the_target_key_is_refused(tmp_path, monkeypatch):
    """The pre-change artifact shape. It must fail loudly, not be assumed to be a regressor."""
    from argotech.config import settings
    from argotech.models.registry import ModelManager

    bundle = _bundle()
    del bundle["target"]
    path = tmp_path / "legacy.joblib"
    joblib.dump(bundle, path)
    monkeypatch.setattr(settings, "AGRONOMIC_MODEL_PATH", str(path))

    with pytest.raises(ValueError, match="forward_z"):
        ModelManager().agronomic_model()


def test_the_production_artifact_declares_the_regression_target():
    """The other half of the contract: what training writes is what serving demands."""
    import joblib as jl

    from argotech.config import settings

    bundle = jl.load(settings.AGRONOMIC_MODEL_PATH)
    assert bundle["target"] == "forward_z"
    assert hasattr(bundle["model"], "predict")
    assert not hasattr(bundle["model"], "predict_proba"), "a regressor has no posterior"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_artifact_contract.py -v`
Expected: all four FAIL — the first three because no `ValueError` is raised, the last with `KeyError: 'target'`.

- [ ] **Step 3: Write the regression target into the bundle and fit the regressor**

In `src/argotech/training/train.py`, in the final-fit block (around line 566), replace the classifier fit:

```python
    y_all = df.label.to_numpy()
    z_all = df.forward_z.to_numpy(dtype=float)
    ok = np.isfinite(z_all)
    final = make_regressor().fit(df[MODEL_FEATURES][ok], z_all[ok])
    holdout = df[df.cluster == max(df.cluster.unique())]
    held_risk = -final.predict(holdout[MODEL_FEATURES])
```

In the `joblib.dump({...})` bundle (around line 585), add one key next to `"label"`:

```python
        # The serving contract. `label` below is prose for a human; this is the machine-checked
        # key that stops a classifier artifact being read as a regressor. See registry.py.
        "target": "forward_z",
```

- [ ] **Step 4: Add the contract guard to the registry**

In `src/argotech/models/registry.py`, inside `agronomic_model()` after `bundle = joblib.load(path)`:

```python
            target = bundle.get("target")
            if target != "forward_z":
                raise ValueError(
                    f"Artifact '{path}' declares target {target!r}; serving requires 'forward_z'. "
                    "This is refused rather than adapted to: a classifier's `.predict()` returns "
                    "class labels 0/1/2, which the hazard map reads as near-zero anomalies, so "
                    "every field would silently report almost no canopy hazard. Retrain with "
                    "`python -m argotech.training.dataset` then `python -m argotech.training.train`."
                )
```

Update the `logger.info` on the following lines to report the target:

```python
            logger.info("Loaded agronomic model %s from %s: %s on %s, %d features, %d clusters, "
                        "trained on %s", version, os.path.abspath(path),
                        type(bundle["model"]).__name__, target, len(bundle["feature_columns"]),
                        len(self._agronomic[3]), bundle.get("trained_on"))
```

- [ ] **Step 5: Collapse the serving branch**

In `src/argotech/serving/pipeline.py`, replace the whole vegetation-hazard block (from `if index_source == "sentinel-2" and settings.VEGETATION_HAZARD_SOURCE == "model":` through the end of the `elif` branch) with:

```python
        if index_source == "sentinel-2":
            if settings.VEGETATION_HAZARD_SOURCE == "model":
                model, columns, model_version, bounds, stats = self.model_manager.agronomic_model()
                # Derived here rather than in `gather_upstream` so a `field_features` row stored by
                # an older precompute run still gets its twins — arithmetic over the raw row and the
                # artifact's snapshot, with no upstream call to pay for.
                cluster = assign_cluster(lat, lon, bounds)
                row = {**row, **cluster_relative_row(row, stats.get(cluster))}
                zhat = await run_in_threadpool(model.predict, pd.DataFrame([row])[columns])
                forecast_z = float(zhat[0])
                logger.info("field=%s forecast_z=%+.3f from model %s (cluster %s)",
                            field_id, forecast_z, model_version, cluster)
            else:
                # Persistence: the field's *observed* anomaly, carried forward unchanged.
                forecast_z, model_version = row.get("ndvi_z_peer"), PERSISTENCE_VERSION

            # One mapping for both sources. The paths differ only in where z comes from — observed
            # today, or predicted 30 days out — so the A/B compares exactly one thing.
            vegetation_hazard = risk.vegetation_hazard_from_anomaly(forecast_z)
            probabilities = _severity_to_probabilities(vegetation_hazard)
            probabilities_of = ("vegetation hazard (peer-relative canopy stress, 30-day horizon) — "
                                "encoded from a scalar, not a fitted posterior")
```

- [ ] **Step 6: Retrain to produce the regressor artifact**

Run: `./venv/bin/python -m argotech.training.train`
Expected: prints `Saved artifacts/agronomic_risk.joblib (agro-<timestamp>) and artifacts/metrics.json`. It must take the **production** path, not `artifacts/experimental/` — the feature set is unchanged, so `artifact_path` returns `PRODUCTION_ARTIFACT`. If it writes to `experimental/`, stop: something changed `MODEL_FEATURES` and the plan's assumption is broken.

- [ ] **Step 7: Run the contract tests and the full suite**

Run: `./venv/bin/python -m pytest tests/test_artifact_contract.py -v`
Expected: 4 passed.

Run: `./venv/bin/python -m pytest tests -q`
Expected: all pass. `test_a_vegetation_hazard_term_actually_contributed` already adapts to the configured source and asserts the `agro-` version prefix.

- [ ] **Step 8: Verify a live prediction end to end**

Run:

```bash
./venv/bin/python - <<'PY'
import json, sys, asyncio
sys.path.insert(0, "src")
from pathlib import Path
from argotech.data import meteo
from argotech.data.sentinel import sentinel_client
F = Path("tests/fixtures")
meteo.fetch_recent = lambda *a, **k: json.loads((F / "meteo_recent.json").read_text())
meteo.climatological_rain_30 = lambda *a, **k: 95.0
sentinel_client.fetch_history = lambda *a, **k: json.loads((F / "sentinel_history.json").read_text())
sentinel_client.fetch_sar_history = lambda *a, **k: json.loads((F / "sentinel_sar_history.json").read_text())
from fastapi.testclient import TestClient
from argotech.data.db import get_db
from argotech.serving.main import app
app.dependency_overrides[get_db] = lambda: None
r = TestClient(app).post("/predict/coldstart", json={"latitude": 10.8, "longitude": 7.9,
                                                    "crop_type": "Maize", "farm_size": 2.0})
b = r.json()
print(r.status_code, b["model_version"], b["risk_assessment"]["hazard"]["vegetation"], b["probabilities"])
PY
```

Expected: `200`, a `agro-<timestamp>` version, a vegetation hazard in [0,1], and three probabilities summing to ~1. The log line must show `forecast_z=` and the registry line must show `HistGradientBoostingRegressor on forward_z`.

- [ ] **Step 9: Lint and commit**

```bash
./venv/bin/python -m ruff check src tests
git add src/argotech/training/train.py src/argotech/models/registry.py \
        src/argotech/serving/pipeline.py tests/test_artifact_contract.py \
        artifacts/agronomic_risk.joblib artifacts/metrics.json
git commit -m "Serve a forward_z regressor, guarded by an explicit artifact target"
```

---

### Task 3: Flip the default and correct the copy

**Files:**
- Modify: `src/argotech/config.py:8-14`
- Modify: `src/argotech/serving/pipeline.py` (`_severity_to_probabilities` docstring)
- Modify: `README.md`

**Interfaces:**
- Consumes: the regressor artifact and serving path from Task 2.
- Produces: no new symbols. `settings.VEGETATION_HAZARD_SOURCE` defaults to `"model"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vegetation_hazard.py`:

```python
def test_the_default_hazard_source_is_the_model():
    """The classifier lost to persistence on rho (0.196 vs 0.351), which is why the default was
    `persistence`. The regressor draws level (0.342) and beats it on P@25 (0.741 vs 0.51), so the
    default flips. Pinned because the justifying comment in config.py must not outlive its numbers.
    """
    from argotech.config import Settings

    assert Settings.model_fields["VEGETATION_HAZARD_SOURCE"].default == "model"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_vegetation_hazard.py::test_the_default_hazard_source_is_the_model -v`
Expected: FAIL, `assert 'persistence' == 'model'`.

Note: `.env` in this working copy already sets `VEGETATION_HAZARD_SOURCE=model`, so `settings.VEGETATION_HAZARD_SOURCE` reads `"model"` regardless. The test asserts on `Settings.model_fields[...].default` precisely so it tests the committed default and not the local environment.

- [ ] **Step 3: Flip the default and replace the justification**

In `src/argotech/config.py`, replace the `VEGETATION_HAZARD_SOURCE` block:

```python
    # Where the vegetation hazard term comes from: "model" or "persistence".
    #
    # Both map a peer-standardised NDVI anomaly through `risk.vegetation_hazard_from_anomaly`; they
    # differ only in where the anomaly comes from — predicted 30 days out, or the field's own
    # observed value carried forward.
    #
    # Defaults to the model since 2026-08-21. The previous default was persistence because the
    # *classifier* ranked worse than it: Spearman 0.196 against 0.351. Regressing `forward_z`
    # instead lifts the model to 0.342 — level with persistence, improving in all 30 of 5 seeds x 6
    # blocked folds — and P@25 to 0.741 against persistence's 0.51. The case for this default rests
    # on P@25, not on rho, where the two are level.
    VEGETATION_HAZARD_SOURCE: str = "model"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./venv/bin/python -m pytest tests/test_vegetation_hazard.py -v`
Expected: all pass, including the new one.

- [ ] **Step 5: Correct the encoding docstring**

In `src/argotech/serving/pipeline.py`, `_severity_to_probabilities` says the encoding "reproduces the model path's own summary statistic". There is no model posterior any more — both paths encode a scalar. Replace that paragraph:

```python
    """Encode a scalar vegetation hazard as the 3-class vector the Kotlin client requires.

    `probabilities` is typed `Map<String, Double>` downstream and is not optional, so a number has
    to become three. Rather than invent a posterior — neither hazard source has one, since both
    produce a single scalar — this puts all the mass on the two classes adjacent to the hazard
    value, the unique distribution satisfying

        h = 0.5 * p_medium + 1.0 * p_high

    so a consumer computing expected severity recovers the hazard exactly. It is an encoding of one
    scalar, not a confidence, and `probabilities_of` says so on the wire.
    """
```

- [ ] **Step 6: Update the README**

Exactly one line describes the served model. `README.md:290` currently reads:

```
| `VEGETATION_HAZARD_SOURCE` | `persistence` | Where the vegetation hazard comes from. `persistence` carries the field's own peer anomaly forward; `model` runs the trained classifier. See [The model](#the-model) |
```

Replace it with:

```
| `VEGETATION_HAZARD_SOURCE` | `model` | Where the peer anomaly fed to the hazard map comes from. `model` predicts it 30 days ahead (`HistGradientBoostingRegressor` on `forward_z`); `persistence` carries the field's own observed value forward. Same mapping either way. See [The model](#the-model) |
```

Verify no other stale description survives:

```bash
grep -n "classifier\|predict_proba\|posterior" README.md
```

Expected: no hits, or hits that describe the control arm or project history rather than the served model.

- [ ] **Step 7: Full verification**

Run: `./venv/bin/python -m ruff check src tests`
Expected: `All checks passed!`

Run: `./venv/bin/python -m pytest tests -q`
Expected: all pass, no new warnings.

- [ ] **Step 8: Commit**

```bash
git add src/argotech/config.py src/argotech/serving/pipeline.py \
        tests/test_vegetation_hazard.py README.md
git commit -m "Default the vegetation hazard to the regressor, and correct the copy"
```

---

## Out of scope

Recorded in the spec, not implemented here: `loss="absolute_error"` for the heavy left tail of `forward_z`; the optical staleness check on `history[-1]`; radar-only inference; Presto embeddings (evaluated and rejected).
