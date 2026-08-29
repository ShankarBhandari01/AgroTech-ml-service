# Peer Anomaly Skew Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ndvi_z_peer` and `rvi_z_peer` mean the same thing in training and in serving, by standardising both against one (cluster, calendar-month) reference snapshot carried in the artifact.

**Architecture:** `satellite_block` and `radar_block` stop receiving a peer *list* and receive a `[mu, sigma]` pair. `dataset.py` and `pipeline.py` both look that pair up from a `peer_stats` table — computed per fold during evaluation, snapshotted into the bundle for serving. This is a second instance of the `cluster_stats` / `cluster_relative_row` mechanism that already crosses this boundary correctly.

**Tech Stack:** Python 3.12, scikit-learn 1.6.1 (pinned — the artifact is a joblib pickle), pandas, FastAPI, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-21-peer-anomaly-skew-design.md`

## Global Constraints

- `SEVERE_Z = -1.0`, `ELEVATED_Z = -0.35` (`training/dataset.py`), `SEVERE_ANOMALY_Z = -1.0`, `ANOMALY_SOFTNESS = 0.5` (`domain/risk.py`), `MAX_LABEL_GAP_DAYS = 45` (`training/dataset.py`) — all fixed. Do not change any of them.
- scikit-learn stays pinned at 1.6.1. Retraining and unpinning are one change, never one alone.
- The response schema does not change. `probabilities` stays `Map<String, Double>`.
- `./venv/bin/python -m ruff check src tests` must pass clean. Line length 110.
- Run tests with `./venv/bin/python -m pytest`, never bare `pytest`.
- **The label is not touched.** `forward_z` keeps its per-date cohort. Only *features* move to the snapshot.
- Missing bucket, unknown cluster, unseen month → **NaN**, never 0.0. Mirror `cluster_relative_row` exactly rather than inventing a third convention.

### Known-hostile detail

`data/training_set.parquet` was built against a frozen `date.today() == 2026-08-11` because Open-Meteo's archive quota was exhausted. A plain `python -m argotech.training.dataset` today will NOT reproduce it. Task 2 requires a dataset rebuild; if the quota is still exhausted, say so in the report rather than silently shipping a partial build — a rebuild that resolves only a subset of sites changes the cohort buckets and therefore every feature.

### Task ordering note

Task 2 is deliberately large. Changing the block signatures breaks both callers at once, and serving cannot look up a bucket that no artifact carries yet. The signature, both callers, the bundle key and the retrain are one atomic change. Tasks 1 and 3 leave the tree green on their own.

---

### Task 1: `peer_stats` as a pure function

**Files:**
- Modify: `src/argotech/features/agronomic.py`
- Test: `tests/test_peer_stats.py` (create)

**Interfaces:**
- Consumes: `CLUSTER_RELATIVE`, `CZ_SUFFIX` conventions already in the module.
- Produces: `PEER_RELATIVE: list[str]`; `peer_bucket(cluster: str | None, sensing_date: str) -> str | None`; `peer_stats(df, group="cluster") -> dict[str, dict[str, list[float]]]`; `peer_reference(stats: dict | None, bucket: str | None, column: str) -> list[float] | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_peer_stats.py`:

```python
"""The reference distribution both paths standardise against.

`ndvi_z_peer` meant two different things — a concurrent cross-site cohort in training, the site's
own 12-month history at serving, correlated only 0.626 with 25.5% sign flips. These pin the shape
of the single replacement so the two callers cannot drift again.
"""

from __future__ import annotations

import math

import pandas as pd

from argotech.features.agronomic import (
    PEER_RELATIVE,
    peer_bucket,
    peer_reference,
    peer_stats,
)


def _frame() -> pd.DataFrame:
    rows = []
    for cluster, base in (("A", 0.5), ("B", 0.2)):
        for month in ("03", "09"):
            for i in range(10):
                rows.append({"cluster": cluster,
                             "obs_date": f"2025-{month}-1{i % 9}",
                             "ndvi": base + i * 0.01,
                             "rvi": base + i * 0.02})
    return pd.DataFrame(rows)


def test_the_bucket_is_cluster_and_calendar_month():
    assert peer_bucket("Kaduna_Grain_Belt", "2026-09-14") == "Kaduna_Grain_Belt|09"
    assert peer_bucket(None, "2026-09-14") is None, "no cluster means no reference"


def test_stats_are_keyed_per_cluster_and_month_for_every_peer_column():
    stats = peer_stats(_frame())
    assert set(stats) == {"A|03", "A|09", "B|03", "B|09"}
    for bucket in stats:
        assert set(stats[bucket]) == set(PEER_RELATIVE)
        for mu, sigma in stats[bucket].values():
            assert math.isfinite(mu) and sigma > 0.0


def test_a_cluster_offset_does_not_leak_across_buckets():
    """A's mean must not move when B's values change — the property that makes per-fold stats safe."""
    df = _frame()
    a_only = peer_stats(df[df.cluster == "A"])
    both = peer_stats(df)
    assert a_only["A|03"]["ndvi"] == both["A|03"]["ndvi"]


def test_an_unknown_bucket_yields_no_reference_rather_than_a_default():
    """None, not [0.0, 1.0]. A fabricated reference asserts 'average for its region' about a region
    that was never measured — the same fabrication `cluster_relative_row` refuses to make."""
    stats = peer_stats(_frame())
    assert peer_reference(stats, "A|03", "ndvi") is not None
    assert peer_reference(stats, "Z|03", "ndvi") is None
    assert peer_reference(stats, None, "ndvi") is None
    assert peer_reference(None, "A|03", "ndvi") is None
    assert peer_reference(stats, "A|03", "not_a_column") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_peer_stats.py -v`
Expected: FAIL with `ImportError: cannot import name 'PEER_RELATIVE'`.

- [ ] **Step 3: Implement**

In `src/argotech/features/agronomic.py`, add beside `cluster_stats`:

```python
# The raw columns whose peer anomaly is standardised. Their `_z_peer` outputs are model features;
# `ndvi_z_peer` is also the entire input to the persistence hazard path.
PEER_RELATIVE = ["ndvi", "rvi"]


def peer_bucket(cluster: str | None, sensing_date: str) -> str | None:
    """The reference bucket for one observation: `"<cluster>|<MM>"`, or None without a cluster.

    Calendar month, not date: the cohort a field is compared against has to be knowable at serving
    time, and a specific date's cohort is not. Month is the coarsest key that still separates the
    growing season from the dry season, which is the variation that would otherwise dominate.
    """
    return f"{cluster}|{sensing_date[5:7]}" if cluster else None


def peer_stats(df, group: str = "cluster") -> dict:
    """`{"<cluster>|<MM>": {column: [mu, sigma]}}` — the reference both paths standardise against.

    Snapshotted into the artifact so serving reproduces training's arithmetic from constants rather
    than from whatever sample it happens to hold. Plain lists rather than a DataFrame: this crosses
    a joblib boundary and then a `predict` hot path, and must not drag pandas into either.

    Computed per fold during evaluation and over the full frame for the shipped artifact. Computing
    it over the full frame during evaluation would standardise a held-out cluster against itself.
    """
    keys = df[group].astype(str) + "|" + df["obs_date"].str.slice(5, 7)
    grouped = df.assign(_bucket=keys).groupby("_bucket")
    mean, sd = grouped[PEER_RELATIVE].mean(), grouped[PEER_RELATIVE].std()
    return {str(b): {col: [float(mean.at[b, col]), float(sd.at[b, col])] for col in PEER_RELATIVE}
            for b in mean.index}


def peer_reference(stats: dict | None, bucket: str | None, column: str) -> list[float] | None:
    """One `[mu, sigma]`, or None when the bucket was never measured.

    None rather than a default: `[0.0, 1.0]` would assert that an unmeasured region is exactly
    average, which is the fabrication `cluster_relative_row` refuses for the same reason.
    """
    if stats is None or bucket is None:
        return None
    entry = (stats.get(bucket) or {}).get(column)
    return entry if entry and math.isfinite(entry[0]) else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/test_peer_stats.py -v`
Expected: 4 passed.

- [ ] **Step 5: Full suite and lint**

Run: `./venv/bin/python -m pytest tests -q` then `./venv/bin/python -m ruff check src tests`
Expected: all pass, ruff clean. Nothing calls the new functions yet, so nothing else moves.

- [ ] **Step 6: Commit**

```bash
git add src/argotech/features/agronomic.py tests/test_peer_stats.py
git commit -m "Add the (cluster, month) peer reference both paths will share"
```

---

### Task 2: The atomic switch

**Files:**
- Modify: `src/argotech/features/agronomic.py` (`satellite_block`, `radar_block` signatures)
- Modify: `src/argotech/training/dataset.py` (feature cohort → bucket lookup; label cohort untouched)
- Modify: `src/argotech/training/train.py` (per-fold stats; bundle snapshot)
- Modify: `src/argotech/models/registry.py` (return and require `peer_stats`)
- Modify: `src/argotech/serving/pipeline.py` (bucket lookup; delete the self-history substitution)
- Test: `tests/test_peer_equivalence.py` (create), `tests/test_artifact_contract.py` (extend)
- Regenerate: `data/training_set.parquet`, `artifacts/agronomic_risk.joblib`, `artifacts/metrics.json`

**Interfaces:**
- Consumes: `PEER_RELATIVE`, `peer_bucket`, `peer_stats`, `peer_reference` from Task 1.
- Produces: `satellite_block(current, past_ndvi, peer_mu_sigma)` and `radar_block(current, peer_mu_sigma)` where `peer_mu_sigma: list[float] | None`; bundle key `peer_stats`; `ModelManager.agronomic_model()` returns a 6-tuple `(model, feature_columns, version, cluster_bounds, cluster_stats, peer_stats)`.

- [ ] **Step 1: Write the failing equivalence test**

This is the test whose absence let the defect live. Create `tests/test_peer_equivalence.py`:

```python
"""Training and serving must compute the SAME peer anomaly for the same input.

`tests/test_cluster_relative.py` pins exactly this property for the `_cz` twins, and those never
drifted. `ndvi_z_peer` had no such test and drifted into two definitions — a concurrent cross-site
cohort in training, the site's own history at serving, r = 0.626 with 25.5% sign flips.
"""

from __future__ import annotations

import math

from argotech.features.agronomic import (
    peer_bucket,
    peer_reference,
    radar_block,
    satellite_block,
)

STATS = {"Kaduna_Grain_Belt|09": {"ndvi": [0.50, 0.10], "rvi": [0.60, 0.20]}}
OBS = {"ndvi": 0.32, "ndwi": 0.10, "evi": 0.25, "sensing_date": "2026-09-14"}
SAR = {"vv": 0.20, "vh": 0.05, "rvi": 0.40}


def _blocks(cluster: str | None):
    bucket = peer_bucket(cluster, OBS["sensing_date"])
    sat = satellite_block(OBS, [0.4, 0.5, 0.6], peer_reference(STATS, bucket, "ndvi"))
    rad = radar_block(SAR, peer_reference(STATS, bucket, "rvi"))
    return sat, rad


def test_the_anomaly_is_the_standardisation_the_snapshot_describes():
    """(0.32 - 0.50) / 0.10 = -1.8, and (0.40 - 0.60) / 0.20 = -1.0. No other convention."""
    sat, rad = _blocks("Kaduna_Grain_Belt")
    assert abs(sat["ndvi_z_peer"] - (-1.8)) < 1e-9
    assert abs(rad["rvi_z_peer"] - (-1.0)) < 1e-9


def test_both_callers_agree_because_neither_computes_the_reference():
    """The training and serving call sites differ only in where `stats` came from. Passing the same
    reference must give the same number — the property that makes one column mean one thing."""
    training_side, _ = _blocks("Kaduna_Grain_Belt")
    serving_side, _ = _blocks("Kaduna_Grain_Belt")
    assert training_side["ndvi_z_peer"] == serving_side["ndvi_z_peer"]


def test_an_unmeasured_region_yields_nan_not_zero():
    """0.0 would assert 'exactly average for its district' about a district never measured."""
    sat, rad = _blocks(None)
    assert math.isnan(sat["ndvi_z_peer"])
    assert math.isnan(rad["rvi_z_peer"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_peer_equivalence.py -v`
Expected: FAIL — `satellite_block` still takes a peer list, so passing `[mu, sigma]` raises or produces a wrong number.

- [ ] **Step 3: Change the block signatures**

In `src/argotech/features/agronomic.py`, replace the peer handling in `satellite_block` and `radar_block`. Keep `past_ndvi` (VCI's own history) exactly as it is — only the peer argument changes:

```python
def _peer_z(value: float, peer_mu_sigma: list[float] | None) -> float:
    """Standardise against the shared reference, or NaN when the bucket was never measured.

    sigma > 1e-9 mirrors `cluster_relative_row`'s constant-column rule, so the two mechanisms
    cannot disagree about a degenerate reference.
    """
    if peer_mu_sigma is None or value is None or not math.isfinite(value):
        return float("nan")
    mu, sigma = peer_mu_sigma
    if not math.isfinite(sigma) or sigma <= 1e-9:
        return 0.0
    return (value - mu) / sigma
```

`satellite_block(current, past_ndvi, peer_mu_sigma)` sets `"ndvi_z_peer": _peer_z(ndvi, peer_mu_sigma)`; `radar_block(current, peer_mu_sigma)` sets `"rvi_z_peer": _peer_z(rvi, peer_mu_sigma)`. Update both docstrings — they currently say "the concurrent cohort, excluding this site", which is the claim that stopped being true.

- [ ] **Step 4: Run the equivalence test to verify it passes**

Run: `./venv/bin/python -m pytest tests/test_peer_equivalence.py -v`
Expected: 3 passed.

- [ ] **Step 5: Point `dataset.py` at the buckets**

In `src/argotech/training/dataset.py`, compute `peer_stats` over the collected observations *before* the row loop, then replace the per-date cohort lookups that feed `satellite_block`/`radar_block` with `peer_reference(stats, peer_bucket(cluster, obs["sensing_date"]), "ndvi" | "rvi")`.

**Leave the label's cohort exactly as it is** — `peers_future`, the `len(peers_future) >= 5` guard and the `forward_z` computation are untouched. Only the feature side moves.

- [ ] **Step 6: Per-fold stats and the bundle snapshot in `train.py`**

Evaluation must compute `peer_stats` from the training fold only, then apply it to both fold halves. Computing over the full frame standardises a held-out cluster against itself.

Because the parquet stores `ndvi_z_peer` already computed, recomputing per fold means re-deriving the column inside `evaluate_fold` from the raw `ndvi`/`rvi` columns and the fold's stats. Expect a held-out cluster to have no bucket at all, so its peer columns become NaN — that is the honest result, and the numbers will drop.

The final fit snapshots stats over the full frame into the bundle:

```python
        "peer_stats": peer_stats(df),
```

- [ ] **Step 7: Registry contract**

In `src/argotech/models/registry.py`, return `peer_stats` as a sixth tuple element and refuse a bundle without it:

```python
            if not bundle.get("peer_stats"):
                raise ValueError(
                    f"Artifact '{path}' carries no peer_stats; serving cannot standardise the peer "
                    "anomaly without the reference the model was fitted against. An artifact from "
                    "before this contract must be retrained.")
```

- [ ] **Step 8: Serving looks up the bucket**

In `src/argotech/serving/pipeline.py`, `gather_upstream` currently passes `series[:-1]` as the peer list for optical and the site's own past RVI for radar. Both become bucket lookups. `gather_upstream` needs the artifact's `peer_stats`, so pass it in as an argument rather than reaching for the model manager inside a function the nightly job also calls.

**Delete the comment claiming the self-history substitution mirrors the optical block.** It documented the defect as a decision.

- [ ] **Step 9: Extend the artifact contract test**

In `tests/test_artifact_contract.py`, add `peer_stats` to the synthetic bundle helper, and add a test that a bundle without it is refused. Assert the production artifact carries a non-empty `peer_stats` whose keys match the `"<cluster>|<MM>"` shape.

- [ ] **Step 10: Rebuild and retrain**

Run `./venv/bin/python -m argotech.training.dataset` then `./venv/bin/python -m argotech.training.train`.

If Open-Meteo's archive quota blocks a full rebuild, STOP and report rather than shipping a partial dataset — fewer sites means different cohort buckets and therefore different features everywhere. Confirm the retrain writes to the PRODUCTION path.

Record: new sample count, new artifact version, and the spatial/temporal numbers for the regressor and all three baselines. **Expect them to be worse than the current figures (regressor spatial P@25 0.760, rho 0.377, macro F1 0.441) because the held-out cluster now loses its top feature. Report the drop; do not treat it as a failure.**

- [ ] **Step 11: Full verification**

Run: `./venv/bin/python -m pytest tests -q` and `./venv/bin/python -m ruff check src tests`
Expected: all pass, ruff clean.

Then confirm end to end that a live prediction still works and the peer anomaly is finite:

```bash
./venv/bin/python - <<'PY'
import json, sys, asyncio; sys.path.insert(0, "src")
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
b = TestClient(app).post("/predict/coldstart", json={"latitude": 10.8, "longitude": 7.9,
                                                    "crop_type": "Maize", "farm_size": 2.0}).json()
print(b["model_version"], b["risk_assessment"]["hazard"]["vegetation"], b["probabilities"])
PY
```

- [ ] **Step 12: Commit**

```bash
git add src/argotech/features/agronomic.py src/argotech/training/dataset.py \
        src/argotech/training/train.py src/argotech/models/registry.py \
        src/argotech/serving/pipeline.py tests/test_peer_equivalence.py \
        tests/test_artifact_contract.py data/training_set.parquet \
        artifacts/agronomic_risk.joblib artifacts/metrics.json
git commit -m "Standardise the peer anomaly against one shared reference"
```

---

### Task 3: Make the documentation cite the retrain

**Files:**
- Modify: `README.md`, `src/argotech/config.py`, `docs/model-design.md`

**Interfaces:**
- Consumes: the new `artifacts/metrics.json` from Task 2. **Every number below must come from that file, not from this plan.**

- [ ] **Step 1: Reconcile every published figure**

`README.md`, `config.py`'s `VEGETATION_HAZARD_SOURCE` comment and `domain/risk.py`'s hazard docstring all carry the three-way comparison against persistence and climatology. Recompute each from the new metrics and update. Report any claim that flips — in particular whether the model still wins spatial P@25, which is the sole remaining basis for the default.

- [ ] **Step 2: Record what changed and why the numbers dropped**

Add a short section to `docs/model-design.md` stating that `ndvi_z_peer`/`rvi_z_peer` were previously two different quantities, what the corrected feature means now (district-typical-for-month, not concurrent-peer), and that leave-one-cluster-out figures fell because a held-out cluster legitimately has no reference bucket. A reader comparing this run to the previous one must find the explanation here rather than assuming a regression.

- [ ] **Step 3: Fix the pre-existing contradiction in `docs/model-design.md` §9**

§9 still says "7,527 samples" and "the weather block is no longer inert" with `ndmi`/`evi`-led importances, while the README says the opposite and cites §9 as its source. Reconcile both against the new metrics.

- [ ] **Step 4: Verify and commit**

Run: `./venv/bin/python -m pytest tests -q` and `./venv/bin/python -m ruff check src tests`

```bash
git add README.md src/argotech/config.py docs/model-design.md
git commit -m "Cite the corrected peer anomaly retrain"
```

---

## Out of scope

Recorded in the spec, not implemented here: the concurrent-cohort upgrade (approach B); `cluster_stats` computed within-fold; `decide()` taking quantiles from the test fold; forward-chaining without a label embargo; the `climatological_rain_30` NaN escape route.
