# Regress `forward_z` instead of classifying its discretisation

**Date:** 2026-08-21
**Status:** approved, not yet implemented
**Affects:** `training/train.py`, `serving/pipeline.py`, `models/registry.py`, `config.py`

## Problem

The served model predicts `label` — a three-way discretisation of `forward_z`, the peer-standardised
NDVI anomaly one 30-day bucket ahead, cut at `SEVERE_Z = -1.0` and `ELEVATED_Z = -0.35`.

Discretising the target before fitting throws away most of the signal. `forward_z` spans -11.35 to
+6.93; a field at z = -0.34 and one at z = +6.9 are both "class 0", and cross-entropy treats them as
identical examples. The model is asked to unlearn an ordering that is already in the data.

The cost is measurable. `config.py` currently defaults `VEGETATION_HAZARD_SOURCE` to `persistence`
because the classifier ranks worse than carrying the field's own anomaly forward: Spearman 0.191
against 0.351 on leave-one-cluster-out.

## Evidence

Five seeds x six leave-one-cluster-out folds, both arms on identical folds, same features, same
`decide()` rule so macro F1 compares estimators rather than thresholding schemes.

All figures are means over the 5 seeds of the per-fold mean.

| metric | classifier | regressor | delta | sd across seeds | folds won |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spearman rho | +0.196 | +0.342 | **+0.147** | 0.020 | **30/30** |
| P@25 | 0.669 | 0.741 | **+0.072** | 0.033 | 21/30 |
| macro F1 | 0.415 | 0.421 | +0.006 | 0.010 | 19/30 |

Rho improves in every fold of every seed — a 7:1 signal-to-noise ratio, the strongest effect
measured on this dataset. P@25 improves in 5/5 seeds. Macro F1 is inside seed noise and should be
reported as unchanged.

The regressor **draws level with persistence** on rho (0.342 vs 0.351, winning 11/30 folds); it does
not surpass it. It does beat persistence decisively on P@25 (0.741 vs 0.51).

Reproduced by a throwaway spike, not kept in the repo. The recipe: for each held-out cluster, fit
`HistGradientBoostingRegressor` on `forward_z` and take `risk = -predict(X)`, fit the current
`make_model` on `label` and take `risk = expected_severity(predict_proba(X))`, then score both with
`rank_correlation`, `precision_at_k` and `_macro(yte, decide(risk, ytr))` from `train.py`. Using
`decide()` on both arms matters: thresholding raw predicted z at the label cuts instead makes the
regressor look 0.026 *worse* on macro F1, which is an artifact of the decision rule, not the
estimator.

## Design

### 1. Hazard mapping — reuse, do not write

`risk.vegetation_hazard_from_anomaly` already maps a peer-standardised NDVI anomaly onto [0,1] with
a logistic centred on `SEVERE_ANOMALY_Z = -1.0`, which is exactly `dataset.SEVERE_Z`. Apply it to the
predicted z.

Both hazard sources then use identical arithmetic, and the only difference between
`VEGETATION_HAZARD_SOURCE=persistence` and `=model` becomes **where z comes from** — observed today,
or predicted 30 days out. No new constants, NaN handling inherited, already covered by
`tests/test_vegetation_hazard.py`.

Inherited quirk, deliberately not fixed here: at z = 0 the hazard is 0.119, so an average field
carries a small baseline vegetation term. That is current shipped behaviour on the persistence path;
changing it would move both paths and belongs in its own change.

### 2. Artifact contract

The bundle gains one key: `target: "forward_z"`. `ModelManager.agronomic_model()` validates it and
raises `ValueError` when it is missing or unrecognised.

This guard is not defensive boilerplate. A classifier's `.predict()` returns class labels 0/1/2. If a
stale classifier artifact reached regressor-shaped serving code — after a rollback, or a
`git checkout` of `artifacts/` — those labels would be read as z values, and
`vegetation_hazard_from_anomaly(0..2)` returns 0.002-0.119 for **every field**. No exception, no log
line, every prediction silently physics-only. `artifact_path`'s servability guard does not catch it
because the feature set is unchanged. Only an explicit contract key does.

### 3. Training (`train.py`)

`ARMS` becomes (factory, target column, risk function) triples:

```python
ARMS = {
    "hgbr":   (make_regressor, "forward_z", lambda m, X: -m.predict(X)),
    "hgb":    (make_model,     "label",     lambda m, X: expected_severity(m.predict_proba(X))),
    "linear": (make_linear,    "label",     lambda m, X: expected_severity(m.predict_proba(X))),
}
```

`make_regressor` is `HistGradientBoostingRegressor` with the same hyperparameters as `make_model`'s
base estimator, minus the `CalibratedClassifierCV` wrapper — a regressor emits a number, so there is
no posterior to calibrate.

`hgbr` becomes the production arm. The classifier stays as a control arm exactly as `linear` does, so
`metrics.json` keeps reporting what was given up rather than deleting the evidence.

Unchanged: `decide()` takes quantiles of the risk score, `precision_at_k` and `rank_correlation` are
rank-based, so an unbounded `-z` needs no rescaling. `_score` already takes `proba` optionally and
omits ECE without it, which is correct — calibration is undefined for the regressor arm.

Rows with non-finite `forward_z` are dropped from the regressor's fit only, never from evaluation.

### 4. Serving (`serving/pipeline.py`)

The vegetation branch gets smaller. The two divergent computations collapse into one:

```python
if index_source == "sentinel-2":
    if settings.VEGETATION_HAZARD_SOURCE == "model":
        model, columns, model_version, bounds, stats = self.model_manager.agronomic_model()
        cluster = assign_cluster(lat, lon, bounds)
        row = {**row, **cluster_relative_row(row, stats.get(cluster))}
        zhat = await run_in_threadpool(model.predict, pd.DataFrame([row])[columns])
        forecast_z = float(zhat[0])
    else:
        forecast_z, model_version = row.get("ndvi_z_peer"), PERSISTENCE_VERSION

    vegetation_hazard = risk.vegetation_hazard_from_anomaly(forecast_z)
    probabilities = _severity_to_probabilities(vegetation_hazard)
```

The separate `elif index_source == "sentinel-2"` persistence branch is deleted.

### 5. Response semantics

No schema change. `probabilities` remains `Map<String, Double>`, so the Kotlin client is untouched.

What changes is meaning: on the model path `probabilities` becomes an encoding of the scalar hazard
rather than a fitted posterior — which is what the persistence path has always emitted. The
`probabilities_of` copy collapses to one sentence covering both paths:

> vegetation hazard (peer-relative canopy stress, 30-day horizon) — encoded from a scalar, not a
> fitted posterior

`forecast_z` is deliberately **not** added to the response. Nothing has asked for it and the pipeline
log line already carries it.

### 6. Testing

| test | what it pins |
| --- | --- |
| `ModelManager` raises on a bundle with missing/unrecognised `target` | the silent-nationwide-zero failure in section 2 |
| `train.py` writes `target: "forward_z"` into the bundle | the other half of that contract |
| e2e model path: hazard in [0,1], `model_version` starts `agro-`, probabilities sum to ~1 | the wire contract |
| `test_every_feature_the_artifact_declares_is_actually_fed` (existing) | feature set unchanged by this work |
| `tests/test_vegetation_hazard.py` (existing) | the mapping, which is the same function |

The contract-guard test is written first and must fail before `ModelManager` is changed.

### 7. Rollout

Deploy the artifact and the code together, or the artifact first. Old artifact plus new code fails
loudly on the `target` key rather than silently — the point of section 2. Rollback reverts both; the
timestamped copies in `artifacts/` make that possible.

`config.py`'s default flips to `"model"`, and the comment justifying persistence is rewritten with
the current numbers.

**Caveat to record honestly:** the case for flipping the default rests on P@25 (0.741 vs 0.51), not
on rho, where the regressor only draws level (11/30 folds). Leaving the default at `persistence`
until rho actually wins is a defensible reading of the same evidence.

## Out of scope

- **`loss="absolute_error"`** — `forward_z` has a heavy left tail (sd 1.18, but 1.67% of rows below
  -3, minimum -11.35), so squared error will chase outliers. A one-word experiment that may beat the
  +0.147 measured here. Worth running before or after, but separately.
- **Presto embeddings** — evaluated 2026-08-21 and rejected: +0.089 P@25 at 2.1 SE, costing 0.012
  macro F1 and 0.055 rho, for 200 MB of torch on the serving path. See
  `artifacts/experimental/agronomic_risk_clusterrel_metrics.json`.
- **Optical staleness** — `pipeline.py` takes `history[-1]` as current regardless of age. A real
  defect, unrelated to this change.
- **Radar-only inference** — needs `dataset.py` to emit radar-only rows and a retrain.
