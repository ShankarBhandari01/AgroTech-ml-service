# Close-out: provenance, replication, sweep, and retiring the incumbent

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Put E02's results under the provenance rule, replicate the one positive finding properly, measure the transfer-versus-specificity curve, and retire `argotech.training` behind a lab artifact-export path.

**Spec:** `docs/superpowers/specs/2026-08-26-two-way-demeaned-estimand-design.md`

## Global Constraints

- Python >=3.11. No new dependency. Ruff clean, `line-length = 110`.
- `venv/bin/python3 -m pytest` (`venv/bin/pytest` has a stale shebang). Baseline: 171 passed, 1 skipped.
- Reproduction gate must keep printing `0.345 [0.2357, 0.4353] 0.0`.
- `src/argotech/domain/` and `src/argotech/serving/` must not be modified.
- **Every committed number must name the run that produced it** — data manifest hash, git SHA, seed. A number that cannot be traced does not get committed.
- Only `level_z` supports cross-`peer_key` comparison; `within_*` and `delta_z` targets are defined *through* `ndvi_z_peer`. Any table crossing keys must say so.

## Measured context (do not re-derive)

Authoritative E02 matrix on `97751bc`, 12/12 ran. Every **spatial** `cluster_month` cell skips — a held-out cluster has no peer bucket, so the feature cannot be formed. `level_z` boosted: spatial leaky +0.0064 / geo_month +0.0066; temporal leaky +0.0194 / cluster_month +0.0179 / geo_month +0.0172. Best result anywhere: spatial, `geo_month`, `within_xy`, linear **+0.0404** (boosted +0.0218) against zero +0.0000.

---

### Task G: artifact export in the lab, then retire `argotech.training`

`src/argotech/training/train.py` is the only producer of `artifacts/agronomic_risk.joblib`, the bundle `models/registry.py` loads and `serving/pipeline.py` runs. It must not be deleted until the lab can produce that bundle.

**Files:** Create `src/argotech/lab/export.py`; modify `src/argotech/lab/run.py`, `src/argotech/models/registry.py` (error-message paths only); delete `src/argotech/training/train.py`, `src/argotech/training/__init__.py`; move `src/argotech/training/embed.py` → `src/argotech/lab/embed.py`; modify `tests/test_features.py`, delete `tests/test_train_arms.py`; test `tests/test_export.py`; docs `README.md`, `docs/model-design.md`.

- [ ] **Step 1: `lab/export.py`.** `export_artifact(panel, cfg, path) -> Path` fits on **all** data — deliberately, and this is the one place a whole-frame fit is correct: at inference there is no future, and `peer_stats`' own docstring already says "over the full frame for the shipped artifact". It must fit peer stats on the full panel, build the target on the full panel, train the configured arm on every row, and write a bundle whose shape `models/registry.py` already validates (`feature_columns`, `peer_stats`, `cluster_stats`, the estimator). Read `registry.py` for the exact contract; do not guess it.
- [ ] **Step 2: tests.** `tests/test_export.py` must assert the written bundle loads through `registry.py`'s own loader and passes its contract check — not merely that a file appeared. Add a test that an artifact missing a required key is rejected.
- [ ] **Step 3: wire it.** `python -m argotech.lab.export <config>` writes the production artifact. Record provenance beside it (manifest hash, git SHA, seed) the way `run.py` does.
- [ ] **Step 4: retire.** `git rm src/argotech/training/train.py src/argotech/training/__init__.py`; `git mv src/argotech/training/embed.py src/argotech/lab/embed.py`; repoint every import. `tests/test_features.py` imports `rank_correlation` from the incumbent — repoint at `argotech.lab.evaluate.spearman` and **check the semantics actually match** before assuming they do. Delete `tests/test_train_arms.py`: it pins the classifier arms and the production-artifact guard, both retired; replacement coverage is `tests/test_arms.py`.
- [ ] **Step 5: docs.** `README.md`'s Training section and `docs/model-design.md` §7 must name the lab, not `argotech.training`. Fix `registry.py`'s two error strings.
- [ ] **Step 6:** full suite, ruff, gate, commit.

---

### Task D: put E02's results under the provenance rule

The authoritative matrix lives in a scratchpad. The spec's own rule is that a cited number names its run.

- [ ] **Step 1:** Re-run the 12-config matrix with results written to `experiments/E02/`, one config file and one `.result.json` per cell, each carrying its provenance block.
- [ ] **Step 2:** Write `experiments/E02/README.md` summarising the matrix, carrying the cross-key caveat verbatim, and stating plainly that every spatial `cluster_month` cell skipped and why.
- [ ] **Step 3:** commit.

---

### Task E: replicate the one positive result

Spatial / `geo_month` / `within_xy` / linear at **+0.0404** is the strongest result in the investigation and rests on one run over four folds. §9.1 G of `docs/model-design.md` records this repo previously reporting a precision@25 whose seed spread was twice the effect — do not repeat it.

- [ ] **Step 1:** Run that configuration across at least five seeds. Report per-seed values, the mean, and a bootstrap CI over folds.
- [ ] **Step 2:** Note which arms are deterministic. The linear arm has no stochastic component, so a seed sweep measures the boosted arm's variance and the *fold* structure's, not the data's — say so rather than implying five independent replicates.
- [ ] **Step 3:** State whether the result survives. If the interval crosses zero, say so plainly; a negative replication is a result.
- [ ] **Step 4:** write to `experiments/E03-replication/`, commit.

---

### Task F: the transfer-versus-specificity curve

Band width was set at 20°/1000 m because it was the coarsest option reaching every held-out cluster. The curve is the actual contribution and is unmeasured.

- [ ] **Step 1:** Sweep `lat_band` × `elev_band` over at least `{5, 10, 20, 90} × {500, 1000, 2000}`, on `level_z` (the only key-comparable target) and on `within_xy` (where the positive result lives), spatial folds.
- [ ] **Step 2:** For every cell report `peer_coverage` per held-out cluster **and** net benefit. Coverage without skill, or skill without coverage, are both failures — the point is where they trade.
- [ ] **Step 3:** Plot or tabulate the trade-off and state where it turns. Known anchors: donor rows per held-out cluster are `[0, 282, 1098, 0]` at 5°/500 m, `[36, 1353, 1098, 647]` at 10°/1000 m, `[2175, 2799, 1792, 3139]` at 20°/1000 m.
- [ ] **Step 4:** write to `experiments/E04-band-sweep/`, add findings to spec §7a, commit.
