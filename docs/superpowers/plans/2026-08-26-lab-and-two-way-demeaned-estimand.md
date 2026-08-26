# Lab and Two-Way Demeaned Estimand — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `argotech.lab` — a config-driven experiment harness — and answer E02: does a two-way demeaned target carry signal that the level target did not?

**Architecture:** Six focused modules replace one 684-line `training/train.py`. `targets.py` holds the estimand family and the strictly-prior field-effect estimator; `panel.py` is the existing dataset builder plus a content manifest; `splits.py`, `arms.py` and `evaluate.py` are extracted from `train.py`; `run.py` executes a YAML config and logs provenance to a local MLflow store. `lab/` is added *alongside* `training/`, which is retired only after E02 has been run against it as a comparator.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn 1.6.1, scipy, pytest. New in the `train` extra only: `mlflow`, `pyyaml`.

**Spec:** `docs/superpowers/specs/2026-08-26-two-way-demeaned-estimand-design.md`

## Global Constraints

- Python `>=3.11`. Copy no new runtime dependency into `[project.dependencies]` — the serving image must not grow. `mlflow` and `pyyaml` go in `[project.optional-dependencies].train` only.
- Ruff: `line-length = 110`, rule set as configured in `pyproject.toml`. Run `ruff check` before every commit; it currently passes at zero and must continue to.
- **`src/argotech/domain/` is not modified by any task in this plan.** It is the incumbent the learned layer must beat.
- **Leakage rule:** every quantity derived from a field's own history uses observations *strictly before* the current row. Any function that violates this fails its test.
- **Provenance rule:** every reported number carries a data manifest hash, a git SHA and a seed. A number that cannot be traced to a run does not enter a document.
- Existing tests must keep passing at every commit: `pytest -q`.
- Target column names are fixed across tasks: the panel's raw target is `forward_z`; the field-effect estimate is `alpha_hat`; the reformulated target is `ztilde`.

---

### Task 1: `lab/targets.py` — the field-effect estimator and the target family

The heart of the spec. Pure functions over a DataFrame, no I/O.

**Files:**
- Create: `src/argotech/lab/__init__.py` (empty)
- Create: `src/argotech/lab/targets.py`
- Test: `tests/test_targets.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Reads columns `site_id`, `cluster`, `obs_date`, `forward_z`, `ndvi_z_peer` from the panel frame.
- Produces:
  - `alpha_hat(df, column="ndvi_z_peer", unit="site_id", min_history=0, shrink=0.0) -> pd.Series`
  - `TARGET_KINDS: tuple[str, ...]` = `("level_z", "within_y", "within_xy", "delta_z")`
  - `build_target(df, kind, features, min_history=0, shrink=0.0) -> tuple[pd.DataFrame, list[str]]` returning the frame with `alpha_hat` and `ztilde` columns added and rows lacking a reference dropped, plus the feature-name list the arm should use.

**Design note for the implementer.** Shrinkage pulls the field mean toward **zero**, not toward a computed cluster mean. Two reasons: `forward_z` is already standardised within cluster and date, so the cluster mean is ~0 by construction; and E01 measured the cluster ICC at exactly 0.0000 (`experiments/E01_variance_decomposition.out`). Shrinking toward a mean computed over the whole frame would also leak future observations into a prior-only quantity. Zero is both correct and one term shorter.

- [ ] **Step 1: Write the failing test**

Create `tests/test_targets.py`:

```python
"""The estimand family, and the one property everything else rests on: alpha_hat sees no future.

The spec's whole argument is that `forward_z` retains a field effect the climatology baseline
collects for free. These pin the transform that removes it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from argotech.lab.targets import TARGET_KINDS, alpha_hat, build_target


def _panel(n_sites: int = 4, n_obs: int = 10) -> pd.DataFrame:
    """A synthetic panel with a real field effect: site k sits at level k - 1.5."""
    rows = []
    for k in range(n_sites):
        for j in range(n_obs):
            rows.append({"site_id": f"S{k}", "cluster": "C0",
                         "obs_date": f"2025-01-{j + 1:02d}",
                         "ndvi_z_peer": (k - 1.5) + 0.1 * ((j % 3) - 1),
                         "forward_z": (k - 1.5) + 0.1 * ((j % 3) - 1)})
    return pd.DataFrame(rows)


def test_alpha_hat_never_sees_the_future():
    df = _panel()
    before = alpha_hat(df)
    spiked = df.copy()
    # Plant an enormous value in every site's LAST observation.
    last = spiked.groupby("site_id").tail(1).index
    spiked.loc[last, "ndvi_z_peer"] = 999.0
    after = alpha_hat(spiked)
    # Every estimate except the ones at or after the spike must be untouched.
    assert np.allclose(before.drop(last), after.drop(last), equal_nan=True), \
        "a future observation moved an earlier alpha_hat: the estimator leaks"


def test_first_observation_has_no_reference():
    df = _panel()
    first = df.groupby("site_id").head(1).index
    a = alpha_hat(df, min_history=1)
    assert a.loc[first].isna().all(), "a site's first row has no prior history and must be NaN"


def test_min_history_drops_rows_rather_than_defaulting_them():
    df = _panel(n_obs=10)
    out, _ = build_target(df, "within_y", features=["ndvi_z_peer"], min_history=3)
    # Each site loses its first 3 rows; nothing is silently filled with 0.
    assert len(out) == 4 * (10 - 3)
    assert out["alpha_hat"].notna().all()


def test_alpha_hat_recovers_the_planted_field_effect():
    df = _panel()
    a = alpha_hat(df, min_history=5)
    got = df.assign(a=a).dropna(subset=["a"]).groupby("site_id")["a"].mean()
    assert np.allclose(got.to_numpy(), [-1.5, -0.5, 0.5, 1.5], atol=0.05)


def test_within_y_removes_the_field_effect():
    df = _panel()
    out, _ = build_target(df, "within_y", features=["ndvi_z_peer"], min_history=3)
    per_site = out.groupby("site_id")["ztilde"].mean().abs()
    assert (per_site < 0.15).all(), f"field effect survived demeaning: {per_site.to_dict()}"
    assert out["ztilde"].var() < out["forward_z"].var()


def test_level_z_is_the_untransformed_target():
    df = _panel()
    out, _ = build_target(df, "level_z", features=["ndvi_z_peer"], min_history=3)
    assert np.allclose(out["ztilde"], out["forward_z"])


def test_shrink_zero_equals_the_raw_field_mean():
    df = _panel()
    assert np.allclose(alpha_hat(df, shrink=0.0, min_history=1).dropna(),
                       alpha_hat(df, min_history=1).dropna())


def test_shrink_pulls_short_histories_toward_zero():
    df = _panel()
    raw = alpha_hat(df, min_history=1)
    shrunk = alpha_hat(df, shrink=5.0, min_history=1)
    both = df.assign(raw=raw, shrunk=shrunk).dropna(subset=["raw"])
    assert (both["shrunk"].abs() <= both["raw"].abs() + 1e-9).all(), \
        "shrinkage must never move an estimate away from zero"
    # The effect must be strongest where history is shortest.
    early = both.groupby("site_id").head(1)
    late = both.groupby("site_id").tail(1)
    assert (early["shrunk"].abs() / early["raw"].abs()).mean() < \
           (late["shrunk"].abs() / late["raw"].abs()).mean()


def test_within_xy_demeans_the_features_too():
    df = _panel()
    out, feats = build_target(df, "within_xy", features=["ndvi_z_peer"], min_history=3)
    assert feats == ["ndvi_z_peer_w"], "within_xy must hand the arm the demeaned feature names"
    assert abs(out["ndvi_z_peer_w"].mean()) < 0.15
    assert "ndvi_z_peer" in out.columns, "the level column stays available for the baselines"


def test_delta_z_is_the_first_difference():
    df = _panel()
    out, _ = build_target(df, "delta_z", features=["ndvi_z_peer"], min_history=0)
    assert np.allclose(out["ztilde"], out["forward_z"] - out["ndvi_z_peer"])


def test_every_declared_kind_builds():
    df = _panel()
    for kind in TARGET_KINDS:
        out, feats = build_target(df, kind, features=["ndvi_z_peer"], min_history=2)
        assert "ztilde" in out.columns and len(out) > 0 and feats
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_targets.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'argotech.lab'`

- [ ] **Step 3: Write the implementation**

Create `src/argotech/lab/__init__.py` as an empty file. Create `src/argotech/lab/targets.py`:

```python
"""The estimand family.

`forward_z` standardises each observation against its cluster peers at the same date, which removes
the cohort-date effect. It never standardises against the field's own history, so the field effect
survives in the target — and `add_site_climatology`, which estimates exactly that effect, beats the
fitted model by supplying the half the target left in (docs/model-design.md, artifacts/metrics.json).

E01 measured the field effect at 34.5% of `forward_z` variance, CI [0.236, 0.435]. This module
removes it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_KINDS = ("level_z", "within_y", "within_xy", "delta_z")

WITHIN_SUFFIX = "_w"


def alpha_hat(df: pd.DataFrame, column: str = "ndvi_z_peer", unit: str = "site_id",
              min_history: int = 0, shrink: float = 0.0) -> pd.Series:
    """The field effect, estimated from a field's strictly earlier observations.

        n_j       = number of observations of this field before the j-th
        raw_j     = (1 / n_j) * SUM_{m < j} column_m
        alpha_j   = (n_j / (n_j + shrink)) * raw_j

    Shrinkage pulls toward **zero**, not toward a computed cluster mean. `forward_z` is already
    standardised within cluster and date, so the cluster mean is ~0 by construction, and E01
    measured the cluster ICC at exactly 0.0000. A cluster mean computed over the whole frame would
    also leak future observations into a quantity defined as prior-only.

    E01 also gives shrinkage a number to beat: the field effect is 34.5% of variance, but an
    unshrunk `alpha_hat` removes only 19.9% of it. The ~15-point gap is this estimator's own noise.

    Rows with fewer than `min_history` prior observations get NaN — they have no reference, and a
    default of 0.0 would assert "exactly average" about a field nothing is known of.
    """
    order = df.sort_values([unit, "obs_date"]).index
    d = df.loc[order]
    grouped = d.groupby(unit, sort=False)[column]

    prior_mean = grouped.transform(lambda s: s.expanding().mean().shift(1))
    prior_n = grouped.cumcount().astype(float)

    weight = np.where(prior_n > 0, prior_n / (prior_n + shrink), 0.0)
    out = pd.Series(weight * prior_mean.fillna(0.0).to_numpy(), index=d.index)
    out[prior_n < max(min_history, 1)] = np.nan

    return out.reindex(df.index)


def _within(df: pd.DataFrame, columns: list[str], unit: str) -> pd.DataFrame:
    """Each column minus its own strictly-prior field mean. The Frisch-Waugh-Lovell other half."""
    out = {}
    for c in columns:
        out[c + WITHIN_SUFFIX] = df[c] - alpha_hat(df, column=c, unit=unit, min_history=1)
    return pd.DataFrame(out, index=df.index)


def build_target(df: pd.DataFrame, kind: str, features: list[str], unit: str = "site_id",
                 min_history: int = 0, shrink: float = 0.0) -> tuple[pd.DataFrame, list[str]]:
    """Attach `alpha_hat` and `ztilde`, drop rows with no reference, and name the arm's features.

    Returns (frame, feature_names). `within_xy` hands back demeaned feature names; every other kind
    hands back `features` unchanged. Residualising the target alone is *not* the within estimator —
    FWL requires demeaning both sides — so the two are separate kinds and E02 runs them head to head.
    """
    if kind not in TARGET_KINDS:
        raise ValueError(f"unknown target kind {kind!r}; expected one of {TARGET_KINDS}")

    out = df.copy()
    out["alpha_hat"] = alpha_hat(out, unit=unit, min_history=min_history, shrink=shrink)

    if kind == "level_z":
        out["ztilde"] = out["forward_z"]
        names = list(features)
    elif kind == "within_y":
        out["ztilde"] = out["forward_z"] - out["alpha_hat"]
        names = list(features)
    elif kind == "within_xy":
        out = pd.concat([out, _within(out, features, unit)], axis=1)
        out["ztilde"] = out["forward_z"] - out["alpha_hat"]
        names = [c + WITHIN_SUFFIX for c in features]
    else:  # delta_z — the first difference; persistence collapses to the zero predictor
        out["ztilde"] = out["forward_z"] - out["ndvi_z_peer"]
        names = list(features)

    required = ["ztilde"] + ([] if kind == "delta_z" else ["alpha_hat"])
    return out.dropna(subset=required).reset_index(drop=True), names
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_targets.py -q && ruff check src/argotech/lab tests/test_targets.py`
Expected: 10 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/argotech/lab/__init__.py src/argotech/lab/targets.py tests/test_targets.py
git commit -m "Remove the half of the demeaning forward_z left in

alpha_hat estimates a field's effect from its strictly earlier
observations and shrinks toward zero, which is the cluster-neutral
anomaly by construction and, per E01, the cluster mean to five decimal
places. The planted-spike test is the load-bearing one: a future
observation that moves an earlier estimate is the failure mode the whole
argument rests on not having."
```

---

### Task 2: `lab/panel.py` — the panel builder plus a content manifest

`training/dataset.py` moves wholesale. It carries the Sentinel and Open-Meteo backfill, the 429 retry, the refuse-to-memoise-an-empty-result fix and the band cache — none of which is being rewritten.

**Files:**
- Move: `src/argotech/training/dataset.py` → `src/argotech/lab/panel.py` (use `git mv` to keep history)
- Modify: `tests/test_embeddings.py` (import path), `tests/test_vegetation_hazard.py` (import path)
- Test: `tests/test_panel.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `manifest(df) -> dict` with keys `content_hash`, `rows`, `sites`, `clusters`, `date_min`, `date_max`; `write_panel(df, path) -> Path` writing `<path>` and `<path>.manifest.json`. Everything already exported by `dataset.py` keeps its name, including `CLUSTERS`, `SEVERE_Z`, `ELEVATED_Z`, `_cached_fetch`, `build_dataset`.

- [ ] **Step 1: Move the module and fix imports**

```bash
git mv src/argotech/training/dataset.py src/argotech/lab/panel.py
grep -rl "argotech.training.dataset" src tests | xargs sed -i '' 's/argotech\.training\.dataset/argotech.lab.panel/g'
pytest -q
```

Expected: the full suite still passes. Four files import from `argotech.training`; this step fixes the two that import `dataset` (`tests/test_embeddings.py`, `tests/test_vegetation_hazard.py`) plus `training/train.py` and `training/embed.py`. `training/train.py` keeps working — it is retired in Task 9, not now.

- [ ] **Step 2: Write the failing test**

Create `tests/test_panel.py`:

```python
"""The manifest is what makes a reported number traceable to the data that produced it.

docs/RESEARCH_SUMMARY.md records a metrics table cited by the README and two source files that
exists in no committed file. A content hash beside every panel is how that stops happening.
"""

from __future__ import annotations

import json

import pandas as pd

from argotech.lab.panel import manifest, write_panel


def _df() -> pd.DataFrame:
    return pd.DataFrame({"site_id": ["A", "A", "B"], "cluster": ["C0", "C0", "C1"],
                         "obs_date": ["2025-01-01", "2025-02-01", "2025-01-01"],
                         "forward_z": [0.1, -0.2, 0.3]})


def test_manifest_describes_the_panel():
    m = manifest(_df())
    assert m["rows"] == 3 and m["sites"] == 2 and m["clusters"] == ["C0", "C1"]
    assert m["date_min"] == "2025-01-01" and m["date_max"] == "2025-02-01"
    assert len(m["content_hash"]) == 64


def test_the_hash_is_stable_across_identical_frames():
    assert manifest(_df())["content_hash"] == manifest(_df())["content_hash"]


def test_the_hash_is_stable_across_row_order():
    shuffled = _df().iloc[::-1].reset_index(drop=True)
    assert manifest(_df())["content_hash"] == manifest(shuffled)["content_hash"], \
        "row order is not data; a re-sorted panel is the same panel"


def test_the_hash_changes_when_a_value_changes():
    changed = _df()
    changed.loc[0, "forward_z"] = 0.10001
    assert manifest(_df())["content_hash"] != manifest(changed)["content_hash"]


def test_write_panel_emits_a_sidecar(tmp_path):
    p = write_panel(_df(), tmp_path / "panel.parquet")
    side = json.loads(p.with_suffix(".parquet.manifest.json").read_text())
    assert side["content_hash"] == manifest(_df())["content_hash"]
    assert pd.read_parquet(p).shape == (3, 4)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_panel.py -q`
Expected: FAIL — `ImportError: cannot import name 'manifest'`

- [ ] **Step 4: Add the manifest functions to `lab/panel.py`**

Append to `src/argotech/lab/panel.py` (and add `import hashlib`, `import json` to the imports):

```python
def manifest(df: pd.DataFrame) -> dict:
    """A content fingerprint for a panel, so a metric can name the data it was measured on.

    Hashed over row-order-independent bytes: `build_samples` parallelises over sites, so two runs
    of the same builder can emit the same rows in a different order. Order is not data.
    """
    canon = df.sort_values(list(df.columns)).reset_index(drop=True)
    digest = hashlib.sha256(pd.util.hash_pandas_object(canon, index=False).values.tobytes())
    return {
        "content_hash": digest.hexdigest(),
        "rows": int(len(df)),
        "sites": int(df["site_id"].nunique()),
        "clusters": sorted(df["cluster"].dropna().unique().tolist()),
        "date_min": str(df["obs_date"].min()),
        "date_max": str(df["obs_date"].max()),
    }


def write_panel(df: pd.DataFrame, path) -> Path:
    """Write the panel and its manifest side by side. Neither is useful without the other."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(manifest(df), indent=2))
    return path
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_panel.py tests/test_embeddings.py tests/test_vegetation_hazard.py -q && ruff check src/argotech/lab`
Expected: all pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add -A src/argotech tests
git commit -m "Move the panel builder into the lab and fingerprint what it emits

git mv, so the Sentinel backfill, the 429 retry and the
refuse-to-memoise-an-empty-result fix keep their history rather than
arriving as new code. The manifest is the addition: hashed over sorted
rows because build_samples parallelises over sites and row order is not
data."
```

---

### Task 3: `lab/splits.py` — the two validation protocols

Extracted from `train.py:424-451` so a split can be tested without training anything.

**Files:**
- Create: `src/argotech/lab/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `leave_one_cluster_out(df) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]` and `forward_chaining(df, n_folds=3) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]`, both yielding `(fold_name, train, test)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_splits.py`:

```python
"""What the two protocols do, and the one thing E01 says a blocked score is not evidence of.

Cluster ICC of forward_z is exactly 0.0000, so leave-one-cluster-out blocks against feature
distribution shift and never against target structure. These pin the mechanics; the interpretation
lives in the spec's section 7.
"""

from __future__ import annotations

import pandas as pd

from argotech.lab.splits import forward_chaining, leave_one_cluster_out


def _df() -> pd.DataFrame:
    rows = []
    for c in ("C0", "C1", "C2"):
        for s in range(4):
            for m in range(1, 13):
                rows.append({"cluster": c, "site_id": f"{c}-{s}",
                             "obs_date": f"2025-{m:02d}-01", "forward_z": 0.1 * m})
    return pd.DataFrame(rows)


def test_every_cluster_is_held_out_exactly_once():
    folds = list(leave_one_cluster_out(_df()))
    assert [name for name, _, _ in folds] == ["C0", "C1", "C2"]


def test_no_site_appears_on_both_sides_of_a_spatial_fold():
    for name, train, test in leave_one_cluster_out(_df()):
        assert set(train.site_id) & set(test.site_id) == set(), f"{name} leaks a site"
        assert set(test.cluster) == {name}


def test_forward_chaining_never_trains_on_the_future():
    for name, train, test in forward_chaining(_df(), n_folds=3):
        assert train.obs_date.max() < test.obs_date.min(), f"{name} trains on the future"


def test_forward_chaining_training_sets_grow():
    sizes = [len(train) for _, train, _ in forward_chaining(_df(), n_folds=3)]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3


def test_a_fold_with_an_empty_side_is_not_yielded():
    single = _df()[lambda d: d.cluster == "C0"]
    assert list(leave_one_cluster_out(single)) == [], \
        "one cluster cannot be blocked against itself"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_splits.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'argotech.lab.splits'`

- [ ] **Step 3: Write the implementation**

Create `src/argotech/lab/splits.py`:

```python
"""The two validation protocols, separated from anything that fits a model.

Leave-one-cluster-out answers "does this work in a district we have never seen"; forward chaining
answers "does it work next month". Both are retained from docs/model-design.md section 6.

One measured caveat belongs with the first, from E01: the cluster ICC of `forward_z` is exactly
0.0000, because `z` is standardised within cluster and date. Blocking by cluster therefore controls
feature distribution shift and nothing about the target's structure. That narrows what a blocked
score is evidence of; it does not make the protocol wrong.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

Fold = tuple[str, pd.DataFrame, pd.DataFrame]


def leave_one_cluster_out(df: pd.DataFrame) -> Iterator[Fold]:
    """One fold per cluster, that cluster held out entirely."""
    for cluster in sorted(df["cluster"].dropna().unique()):
        test = df[df["cluster"] == cluster]
        train = df[df["cluster"] != cluster]
        if len(train) and len(test):
            yield str(cluster), train, test


def forward_chaining(df: pd.DataFrame, n_folds: int = 3) -> Iterator[Fold]:
    """Expanding-window temporal folds, cut on the observation date.

    Cut on `obs_date`, the date a prediction would have been made, rather than on `label_date`.
    Cutting on the label date would let the last month of training outcomes become known after the
    first test prediction was made — the defect docs/RESEARCH_SUMMARY.md section 5 records as
    uncorrected in the incumbent.
    """
    dates = np.sort(df["obs_date"].unique())
    if len(dates) <= n_folds:
        return
    for cut in np.array_split(dates, n_folds + 1)[1:]:
        boundary = cut[0]
        train, test = df[df["obs_date"] < boundary], df[df["obs_date"] >= boundary]
        if len(train) and len(test):
            yield f"train < {boundary}", train, test
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_splits.py -q && ruff check src/argotech/lab/splits.py`
Expected: 5 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/argotech/lab/splits.py tests/test_splits.py
git commit -m "Separate the protocols from anything that fits a model

Also fixes what RESEARCH_SUMMARY flagged and nobody corrected: forward
chaining now cuts on obs_date, the date a prediction would have been
made, not on label_date, which let the last month of training outcomes
become known after the first test prediction."
```

---

### Task 4: `lab/arms.py` — baselines and estimators behind one interface

Under the reformulated target the quantity being predicted is continuous, so every arm is a regressor returning a score. The classification arms in `train.py` are not carried over; the 3-class discretisation existed to serve a control classifier.

**Files:**
- Create: `src/argotech/lab/arms.py`
- Test: `tests/test_arms.py`

**Interfaces:**
- Consumes: `argotech.lab.targets.alpha_hat`.
- Produces: `ARMS: dict[str, Callable[[int], Arm]]` keyed `"zero"`, `"persistence"`, `"climatology"`, `"linear"`, `"boosted"`. Each `Arm` has `.fit(train, features) -> Arm` and `.predict(test, features) -> np.ndarray`, predicting `ztilde`. Also `resid_sd(arm, train, features) -> float`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_arms.py`:

```python
"""Every arm behind one interface, including the ones that fit nothing.

The point of the reformulation is that the baselines stop winning by proxy. `zero` is the
climatology baseline under the new target *by construction*, so it is here as an arm rather than a
special case, and it is the one E02 has to beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from argotech.lab.arms import ARMS, resid_sd
from argotech.lab.targets import build_target

FEATS = ["ndvi_z_peer", "rain_30"]


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for k in range(8):
        for j in range(12):
            level = k - 3.5
            rows.append({"site_id": f"S{k}", "cluster": f"C{k % 2}",
                         "obs_date": f"2025-{j + 1:02d}-01",
                         "ndvi_z_peer": level + rng.normal(0, 0.2),
                         "rain_30": rng.normal(50, 10),
                         "forward_z": level + rng.normal(0, 0.2)})
    return pd.DataFrame(rows)


def _split():
    df, feats = build_target(_panel(), "within_y", features=FEATS, min_history=2)
    return df.iloc[:60], df.iloc[60:], feats


def test_every_declared_arm_fits_and_predicts_the_right_shape():
    train, test, feats = _split()
    for name, make in ARMS.items():
        pred = make(42).fit(train, feats).predict(test, feats)
        assert pred.shape == (len(test),), name
        assert np.isfinite(pred).all(), f"{name} emitted a non-finite prediction"


def test_the_zero_arm_predicts_zero():
    train, test, feats = _split()
    pred = ARMS["zero"](42).fit(train, feats).predict(test, feats)
    assert np.allclose(pred, 0.0), "under ztilde the climatology baseline IS the zero predictor"


def test_the_climatology_arm_is_the_zero_arm_under_the_reformulated_target():
    train, test, feats = _split()
    assert np.allclose(ARMS["climatology"](42).fit(train, feats).predict(test, feats),
                       ARMS["zero"](42).fit(train, feats).predict(test, feats)), \
        "if these differ, the target has not removed the field effect"


def test_persistence_carries_the_current_within_deviation_forward():
    train, test, feats = _split()
    pred = ARMS["persistence"](42).fit(train, feats).predict(test, feats)
    assert np.allclose(pred, test["ndvi_z_peer"] - test["alpha_hat"])


def test_the_fitted_arms_are_deterministic_given_a_seed():
    train, test, feats = _split()
    for name in ("linear", "boosted"):
        a = ARMS[name](42).fit(train, feats).predict(test, feats)
        b = ARMS[name](42).fit(train, feats).predict(test, feats)
        assert np.allclose(a, b), f"{name} is not reproducible at a fixed seed"


def test_the_fitted_arms_tolerate_a_missing_feature_value():
    train, test, feats = _split()
    holed = test.copy()
    holed.loc[holed.index[0], feats[0]] = np.nan
    for name in ("linear", "boosted"):
        pred = ARMS[name](42).fit(train, feats).predict(holed, feats)
        assert np.isfinite(pred).all(), f"{name} propagated a NaN into its prediction"


def test_resid_sd_is_positive_and_finite():
    train, _, feats = _split()
    sd = resid_sd(ARMS["boosted"](42).fit(train, feats), train, feats)
    assert np.isfinite(sd) and sd > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_arms.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'argotech.lab.arms'`

- [ ] **Step 3: Write the implementation**

Create `src/argotech/lab/arms.py`:

```python
"""Every arm behind one interface: fit on a training frame, predict `ztilde` on a test frame.

The baselines are arms rather than special cases because the whole argument turns on comparing
against them honestly. Under the reformulated target `climatology` collapses onto `zero` by
construction, which is the point: the baseline that beat the incumbent can no longer win by
supplying a field effect the target left in.

Regressors, not classifiers. `ztilde` is continuous; the three-class discretisation in the
incumbent existed to serve a control classifier and is not carried over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Arm:
    """Fit-and-predict. Subclasses override `_fit` and `_predict`."""

    def fit(self, train: pd.DataFrame, features: list[str]) -> "Arm":
        self._fit(train, features)
        return self

    def predict(self, test: pd.DataFrame, features: list[str]) -> np.ndarray:
        return np.asarray(self._predict(test, features), dtype=float)

    def _fit(self, train, features): ...

    def _predict(self, test, features): raise NotImplementedError


class Zero(Arm):
    """Predict no deviation from the field's own norm. Climatology, under this target."""

    def _predict(self, test, features):
        return np.zeros(len(test))


class Persistence(Arm):
    """Carry the field's current within-deviation forward."""

    def _predict(self, test, features):
        return (test["ndvi_z_peer"] - test["alpha_hat"]).fillna(0.0).to_numpy()


class Sklearn(Arm):
    """A scikit-learn estimator, with NaN handling stated rather than assumed."""

    def __init__(self, estimator, impute: bool):
        self.estimator = estimator
        self.impute = impute

    def _fit(self, train, features):
        X, y = train[features], train["ztilde"]
        ok = y.notna()
        self.estimator.fit(X[ok], y[ok])

    def _predict(self, test, features):
        return self.estimator.predict(test[features])


def _linear(seed: int) -> Arm:
    """Ridge over standardised features, median-imputed.

    A linear arm is here because the cross-region literature finds simpler models transfer better
    under shift, and because on the incumbent it did not merely match the boosted trees out of
    cluster — it beat them, 0.421 to 0.408 (docs/model-design.md section 9.1 C). It handicaps
    itself by needing imputation where the boosted arm routes NaN natively, which makes the result
    stronger rather than weaker.
    """
    return Sklearn(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 Ridge(alpha=1.0, random_state=seed)), impute=True)


def _boosted(seed: int) -> Arm:
    """One gradient-boosted tree. Not a stack: on tabular data this size a stack buys a fraction of
    a point and costs interpretability, latency and four times the retraining surface.

    NaN is routed down its own branch rather than filled. A substituted value would silently claim
    the field was observed.
    """
    return Sklearn(HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_depth=None, min_samples_leaf=25,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=seed), impute=False)


ARMS = {
    "zero": lambda seed: Zero(),
    "persistence": lambda seed: Persistence(),
    "climatology": lambda seed: Zero(),
    "linear": _linear,
    "boosted": _boosted,
}


def resid_sd(arm: Arm, train: pd.DataFrame, features: list[str]) -> float:
    """In-sample residual spread, used to turn a point prediction into an event probability.

    Deliberately in-sample and deliberately simple: it is a scale for the normal CDF in
    `evaluate.prob_event`, not an uncertainty claim. Distribution-free intervals with a coverage
    guarantee are conformal's job and are out of scope for this plan.
    """
    resid = train["ztilde"].to_numpy() - arm.predict(train, features)
    sd = float(np.nanstd(resid))
    return sd if np.isfinite(sd) and sd > 0 else 1.0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_arms.py -q && ruff check src/argotech/lab/arms.py`
Expected: 7 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/argotech/lab/arms.py tests/test_arms.py
git commit -m "Put the baselines behind the same interface as the models

climatology collapses onto zero under ztilde, and a test asserts it: if
those two arms ever diverge, the target has stopped removing the field
effect and every number downstream is measuring the old problem again."
```

---

### Task 5: `lab/evaluate.py` — net benefit, ranking, and intervals

**Files:**
- Create: `src/argotech/lab/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure numeric functions).
- Produces:
  - `prob_event(pred, resid_sd, tau) -> np.ndarray`
  - `net_benefit(y, prob, threshold) -> float`
  - `decision_curve(y, prob, thresholds) -> list[dict]` with keys `threshold`, `model`, `visit_all`, `visit_none`
  - `precision_at_k(y, score, k) -> float`
  - `spearman(score, truth) -> float`
  - `bootstrap_ci(values, alpha=0.05, seed=0) -> tuple[float, float]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate.py`:

```python
"""Net benefit, and the arithmetic it has to reproduce.

The incumbent reports macro F1 and precision@25 and lets them disagree, leaving the operating point
to a constant nobody derived (docs/model-design.md section 9.1 D). Net benefit replaces that
argument with a curve, so these tests pin it against the closed form rather than against itself.
"""

from __future__ import annotations

import numpy as np

from argotech.lab.evaluate import (
    bootstrap_ci,
    decision_curve,
    net_benefit,
    precision_at_k,
    prob_event,
    spearman,
)


def test_net_benefit_matches_the_closed_form():
    # 10 cases, 4 events. Flag the 5 highest probabilities: 3 true positives, 2 false positives.
    y = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0])
    prob = np.array([.9, .8, .7, .65, .6, .2, .1, .1, .1, .1])
    t = 0.5
    # NB = TP/n - (FP/n) * (t / (1 - t))
    assert np.isclose(net_benefit(y, prob, t), 3 / 10 - (2 / 10) * (t / (1 - t)))


def test_visit_none_scores_exactly_zero():
    y = np.array([1, 0, 1, 0])
    assert net_benefit(y, np.zeros(4), 0.5) == 0.0


def test_visit_all_matches_the_prevalence_formula():
    y = np.array([1, 1, 0, 0, 0])
    t = 0.25
    curve = decision_curve(y, np.full(5, 0.99), [t])[0]
    assert np.isclose(curve["visit_all"], 0.4 - 0.6 * (t / (1 - t)))
    assert curve["visit_none"] == 0.0


def test_a_perfect_ranker_beats_visit_all_at_every_threshold():
    y = np.array([1] * 5 + [0] * 15)
    prob = np.concatenate([np.full(5, 0.95), np.full(15, 0.02)])
    for row in decision_curve(y, prob, [0.1, 0.3, 0.5, 0.7]):
        assert row["model"] >= row["visit_all"] and row["model"] >= row["visit_none"]


def test_prob_event_is_a_normal_cdf_at_the_threshold():
    # A prediction sitting exactly on tau has probability 0.5 of falling at or below it.
    assert np.allclose(prob_event(np.array([-1.0]), 0.8, tau=-1.0), 0.5)
    # A worse prediction (further below tau) must carry higher probability.
    assert prob_event(np.array([-2.0]), 0.8, -1.0)[0] > prob_event(np.array([0.0]), 0.8, -1.0)[0]


def test_precision_at_k_counts_events_in_the_top_k():
    y = np.array([1, 0, 1, 0, 1])
    score = np.array([.9, .8, .7, .6, .5])   # top 3 contains 2 events
    assert np.isclose(precision_at_k(y, score, 3), 2 / 3)


def test_precision_at_k_clamps_k_to_the_sample():
    y = np.array([1, 0])
    assert np.isclose(precision_at_k(y, np.array([.9, .1]), 25), 0.5)


def test_spearman_is_signed_correctly():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(truth, truth) > 0.99
    assert spearman(-truth, truth) < -0.99


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    vals = [0.40, 0.42, 0.45, 0.39, 0.44]
    lo, hi = bootstrap_ci(vals, seed=0)
    assert lo < np.mean(vals) < hi
    assert bootstrap_ci(vals, seed=0) == bootstrap_ci(vals, seed=0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_evaluate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'argotech.lab.evaluate'`

- [ ] **Step 3: Write the implementation**

Create `src/argotech/lab/evaluate.py`:

```python
"""What "good" means, once ranking and classification stop agreeing.

The incumbent reports macro F1 and precision@25, they disagree, and docs/model-design.md section 6
leaves the choice as a decision "someone has to make consciously" that nobody has made. Net benefit
(Vickers & Elkin, Medical Decision Making 26:565-574, 2006) makes it unnecessary: a model is
evaluated by the consequences of acting on it, across the whole range of cost ratios, against
visiting every field and visiting none.

    NB(t) = TP/n - (FP/n) * (t / (1 - t))

The threshold t is the probability at which an extension visit becomes worthwhile, so t/(1-t) is
exactly the missed-outbreak-to-wasted-trip cost ratio. Reporting the curve rather than one point is
what turns section 9.1 D's alert-rate table into an answer.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr

# The incumbent's severe cut, reused so an event means the same thing across documents.
DEFAULT_TAU = -1.0
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def prob_event(pred: np.ndarray, resid_sd: float, tau: float = DEFAULT_TAU) -> np.ndarray:
    """P(ztilde <= tau | prediction), as a normal CDF over the training residual spread.

    Net benefit needs probabilities, and the arms emit point predictions on a continuous target.
    Assuming normal residuals is the cheapest defensible bridge; it is an assumption and it is
    stated here rather than buried. Distribution-free coverage is conformal's job, out of scope.
    """
    sd = resid_sd if resid_sd > 0 else 1.0
    z = (tau - np.asarray(pred, dtype=float)) / sd
    return np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in z])


def net_benefit(y: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    """Net benefit of acting on `prob` at `threshold`, in true-positives-per-case units."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0 or not 0.0 < threshold < 1.0:
        return 0.0
    flagged = np.asarray(prob, dtype=float) >= threshold
    tp = float(np.sum(flagged & (y == 1)))
    fp = float(np.sum(flagged & (y == 0)))
    return tp / n - (fp / n) * (threshold / (1.0 - threshold))


def decision_curve(y: np.ndarray, prob: np.ndarray,
                   thresholds=DEFAULT_THRESHOLDS) -> list[dict]:
    """The model against both default strategies, at every threshold.

    `visit_none` is 0 by definition. `visit_all` flags everything, so its net benefit is
    prevalence - (1 - prevalence) * odds(t). A model is worth deploying only where it clears both.
    """
    y = np.asarray(y, dtype=float)
    ones = np.ones(len(y))
    return [{"threshold": float(t),
             "model": net_benefit(y, prob, t),
             "visit_all": net_benefit(y, ones, t),
             "visit_none": 0.0}
            for t in thresholds]


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    """Share of the top-k ranked cases that were events — an agent visits k farms this week."""
    y = np.asarray(y, dtype=float)
    k = min(int(k), len(y))
    if k <= 0:
        return float("nan")
    top = np.argsort(-np.asarray(score, dtype=float), kind="stable")[:k]
    return float(np.mean(y[top] == 1))


def spearman(score: np.ndarray, truth: np.ndarray) -> float:
    """Rank correlation, NaN-safe and 0.0 rather than NaN on a degenerate input."""
    s, t = np.asarray(score, dtype=float), np.asarray(truth, dtype=float)
    ok = np.isfinite(s) & np.isfinite(t)
    if ok.sum() < 3 or np.std(s[ok]) == 0 or np.std(t[ok]) == 0:
        return 0.0
    rho = spearmanr(s[ok], t[ok]).statistic
    return float(rho) if np.isfinite(rho) else 0.0


def bootstrap_ci(values, alpha: float = 0.05, n: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile CI on the mean of per-fold scores.

    Every headline number in artifacts/metrics.json is a mean over four or six folds reported
    without one, and section 9.1 G records a precision@25 seed spread of 0.14 on a metric computed
    over 25 items. Fold means without an interval are how that goes unnoticed.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_evaluate.py -q && ruff check src/argotech/lab/evaluate.py`
Expected: 9 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/argotech/lab/evaluate.py tests/test_evaluate.py
git commit -m "Replace the F1-versus-precision argument with a decision curve

t/(1-t) is the missed-outbreak-to-wasted-trip cost ratio, so reporting
net benefit across thresholds answers the question section 9.1 D's
alert-rate table was posing and could not settle. Fold means now carry a
bootstrap interval, because a precision@25 with a seed spread of 0.14 and
no error bar is how a coin flip gets reported as a win."
```

---

### Task 6: `lab/run.py` and the E02 config — provenance-carrying execution

**Files:**
- Create: `src/argotech/lab/run.py`
- Create: `experiments/E02-within-vs-level.yaml`
- Modify: `pyproject.toml` (add `mlflow`, `pyyaml` to the `train` extra)
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `lab.targets.build_target`, `lab.splits.leave_one_cluster_out`/`forward_chaining`, `lab.arms.ARMS`/`resid_sd`, `lab.evaluate.*`, `lab.panel.manifest`.
- Produces: `load_config(path) -> dict`, `run_experiment(cfg, df) -> dict`, `main(argv=None) -> int`. The result dict has keys `config`, `provenance` (`git_sha`, `content_hash`, `seed`), `folds` (list of per-fold dicts) and `summary` (per-arm means with bootstrap CIs).

- [ ] **Step 1: Add the dependencies**

In `pyproject.toml`, extend the `train` extra:

```toml
train = [
  "pyarrow",   # parquet for the built training set
  "torch",     # frozen Presto encoder only; no training, no GPU
  "einops",    # required by the vendored Presto file
  "pyyaml",    # experiment configs; an experiment is a file, not a flag
  # Local file store only — mlruns/ plus the sqlite backend already on disk. No tracking server.
  # Here for two things the paper needs: run comparison across arms, and a staged model registry.
  "mlflow",
]
```

Run: `pip install -e '.[train,dev]'`

- [ ] **Step 2: Write the failing test**

Create `tests/test_run.py`:

```python
"""An experiment is a config file with a run id, not a flag on a 684-line script.

RESEARCH_SUMMARY records a metrics table cited by the README and two source files that exists in no
committed file. These pin the mechanism that makes that impossible: every result carries the data
hash, the git SHA and the seed that produced it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argotech.lab.run import load_config, run_experiment


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for c in range(3):
        for k in range(6):
            for j in range(12):
                level = (k - 2.5) * 0.5
                rows.append({"site_id": f"C{c}-S{k}", "cluster": f"C{c}",
                             "obs_date": f"2025-{j + 1:02d}-01",
                             "ndvi_z_peer": level + rng.normal(0, 0.3),
                             "rain_30": rng.normal(50, 10),
                             "forward_z": level + rng.normal(0, 0.3)})
    return pd.DataFrame(rows)


CFG = {"target": "within_y", "features": ["ndvi_z_peer", "rain_30"],
       "arms": ["zero", "persistence", "boosted"], "splits": ["spatial"],
       "min_history": 2, "shrink": 0.0, "seed": 42, "tau": -0.5}


def test_a_result_names_the_data_the_seed_and_the_code():
    out = run_experiment(CFG, _panel())
    p = out["provenance"]
    assert len(p["content_hash"]) == 64 and p["seed"] == 42 and p["git_sha"]


def test_every_arm_is_scored_in_every_fold():
    out = run_experiment(CFG, _panel())
    assert len(out["folds"]) == 3
    for fold in out["folds"]:
        assert set(fold["arms"]) == {"zero", "persistence", "boosted"}
        for scores in fold["arms"].values():
            assert {"net_benefit", "precision_at_25", "spearman"} <= set(scores)


def test_the_summary_carries_an_interval_per_arm():
    out = run_experiment(CFG, _panel())
    for name, s in out["summary"].items():
        assert "net_benefit_mean" in s and "net_benefit_ci" in s, name
        lo, hi = s["net_benefit_ci"]
        assert lo <= s["net_benefit_mean"] <= hi


def test_the_same_config_and_data_give_the_same_numbers():
    a, b = run_experiment(CFG, _panel()), run_experiment(CFG, _panel())
    assert a["summary"] == b["summary"]
    assert a["provenance"]["content_hash"] == b["provenance"]["content_hash"]


def test_an_unknown_arm_fails_loudly():
    with pytest.raises(ValueError, match="unknown arm"):
        run_experiment({**CFG, "arms": ["zero", "magic"]}, _panel())


def test_the_committed_e02_config_parses():
    cfg = load_config("experiments/E02-within-vs-level.yaml")
    assert cfg["target"] in ("level_z", "within_y", "within_xy", "delta_z")
    assert "zero" in cfg["arms"], "the zero predictor is the baseline E02 exists to beat"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_run.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'argotech.lab.run'`

- [ ] **Step 4: Write the implementation**

Create `src/argotech/lab/run.py`:

```python
"""One entrypoint. An experiment is a config file, not a flag.

    python -m argotech.lab.run experiments/E02-within-vs-level.yaml

The incumbent's experiments were `--no-radar`, `--no-bands`, `--seed` and `--folds-only` on a
684-line script, and its results were compared across builds whose data differed because the builder
keys its window off date.today(). Every run here records the panel's content hash, the git SHA and
the seed, so a number in a document can be traced to the run that produced it — or cannot be cited.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from argotech.lab.arms import ARMS, resid_sd
from argotech.lab.evaluate import (
    DEFAULT_TAU,
    DEFAULT_THRESHOLDS,
    bootstrap_ci,
    decision_curve,
    net_benefit,
    precision_at_k,
    prob_event,
    spearman,
)
from argotech.lab.panel import manifest
from argotech.lab.splits import forward_chaining, leave_one_cluster_out

SPLITS = {"spatial": leave_one_cluster_out, "temporal": forward_chaining}


def load_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("seed", 42)
    cfg.setdefault("min_history", 3)
    cfg.setdefault("shrink", 0.0)
    cfg.setdefault("tau", DEFAULT_TAU)
    cfg.setdefault("splits", ["spatial", "temporal"])
    return cfg


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def _score_arm(arm, train, test, features, tau, seed):
    """One arm on one fold: net benefit at the decision threshold, plus ranking diagnostics."""
    fitted = arm.fit(train, features)
    pred = fitted.predict(test, features)
    prob = prob_event(pred, resid_sd(fitted, train, features), tau)
    y = (test["ztilde"].to_numpy() <= tau).astype(int)
    curve = decision_curve(y, prob, DEFAULT_THRESHOLDS)
    return {
        # The headline: net benefit at the midpoint threshold. The whole curve travels with it.
        "net_benefit": net_benefit(y, prob, 0.20),
        "decision_curve": curve,
        # -pred is the risk score: a more negative predicted ztilde means a field falling
        # further behind its own norm, so it ranks higher for a visit.
        "precision_at_25": precision_at_k(y, -pred, 25),
        "spearman": spearman(pred, test["ztilde"].to_numpy()),
        "event_rate": float(np.mean(y)),
        "n": int(len(test)),
    }


def run_experiment(cfg: dict, df: pd.DataFrame) -> dict:
    """Every arm, every fold, every declared split. Returns folds and a summary with intervals."""
    from argotech.lab.targets import build_target  # local: keeps the import graph acyclic

    unknown = [a for a in cfg["arms"] if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; expected from {sorted(ARMS)}")

    frame, features = build_target(df, cfg["target"], features=cfg["features"],
                                   min_history=cfg["min_history"], shrink=cfg["shrink"])

    folds = []
    for split_name in cfg["splits"]:
        for fold_name, train, test in SPLITS[split_name](frame):
            scored = {name: _score_arm(ARMS[name](cfg["seed"]), train, test,
                                       features, cfg["tau"], cfg["seed"])
                      for name in cfg["arms"]}
            folds.append({"split": split_name, "fold": fold_name, "arms": scored})

    summary = {}
    for name in cfg["arms"]:
        per_fold = {m: [f["arms"][name][m] for f in folds]
                    for m in ("net_benefit", "precision_at_25", "spearman")}
        summary[name] = {f"{m}_mean": float(np.nanmean(v)) for m, v in per_fold.items()}
        summary[name].update({f"{m}_ci": bootstrap_ci(v, seed=cfg["seed"])
                              for m, v in per_fold.items()})

    return {"config": cfg,
            "provenance": {"git_sha": _git_sha(), "seed": cfg["seed"],
                           **{k: v for k, v in manifest(df).items()}},
            "folds": folds, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one lab experiment from its config file.")
    ap.add_argument("config", help="path to an experiments/*.yaml")
    ap.add_argument("--data", default="data/training_set.parquet")
    ap.add_argument("--out", default=None, help="where to write results (default: alongside config)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    result = run_experiment(cfg, pd.read_parquet(args.data))

    out = Path(args.out or Path(args.config).with_suffix(".result.json"))
    out.write_text(json.dumps(result, indent=2, default=float))

    try:
        import mlflow
        mlflow.set_experiment(cfg.get("name", Path(args.config).stem))
        with mlflow.start_run():
            mlflow.log_params({**{k: str(v) for k, v in cfg.items()}, **result["provenance"]})
            for arm, s in result["summary"].items():
                mlflow.log_metrics({f"{arm}.{k}": v for k, v in s.items()
                                    if not k.endswith("_ci")})
            mlflow.log_artifact(str(out))
    except ImportError:  # pragma: no cover — mlflow lives in the `train` extra
        print("mlflow not installed; results written to disk only", file=sys.stderr)

    print(f"{out}\n")
    for arm, s in sorted(result["summary"].items(), key=lambda kv: -kv[1]["net_benefit_mean"]):
        lo, hi = s["net_benefit_ci"]
        print(f"  {arm:<14} NB {s['net_benefit_mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
              f"   P@25 {s['precision_at_25_mean']:.3f}"
              f"   rho {s['spearman_mean']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `experiments/E02-within-vs-level.yaml`:

```yaml
# E02 — does a two-way demeaned target carry signal the level target did not?
#
# The invalidating experiment. If neither within_y nor within_xy beats `zero` on net benefit with
# non-overlapping intervals, the reformulated target carries no learnable signal at this resolution
# and the recommendation becomes shipping the domain/ agronomy alone.
#
# Run all four kinds by editing `target` and re-running; each writes its own .result.json.
name: E02-within-vs-level
target: within_y            # level_z | within_y | within_xy | delta_z
features:
  - ndvi_z_peer
  - ndmi
  - evi
  - vci
  - rvi_z_peer
  - et0_90
  - dry_spell_30
  - dry_spell_90
  - diurnal_range_30
  - radiation_90
  - rain_30
  - rain_90
  - rain_anomaly_30
  - rh_mean_30
  - water_deficit_30
  - water_satisfaction_30
  - heat_stress_days
  - gdd_90
  - stage_kc
  - elevation
arms: [zero, persistence, linear, boosted]
splits: [spatial, temporal]
min_history: 3
shrink: 0.0
tau: -1.0                   # the incumbent's severe cut, so an "event" means the same thing
seed: 42
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_run.py -q && ruff check src/argotech/lab/run.py`
Expected: 6 passed, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/argotech/lab/run.py experiments/E02-within-vs-level.yaml tests/test_run.py pyproject.toml
git commit -m "Make an experiment a config file with a run id

Every result now records the panel content hash, the git SHA and the
seed. RESEARCH_SUMMARY marks a metrics table cited in the README and two
source files as UNVERIFIED because it exists in no committed file; under
this runner a number that cannot name its run cannot be cited.

mlflow and pyyaml go in the train extra only. The serving image does not
grow."
```

---

### Task 7: `lab/variance.py` — fold E01 into the lab

E01 currently lives as a standalone probe. It produced numbers the spec cites, so it becomes tested library code rather than a script.

**Files:**
- Create: `src/argotech/lab/variance.py`
- Delete: `experiments/E01_variance_decomposition.py`
- Keep: `experiments/E01_variance_decomposition.out` (the committed record of the run)
- Test: `tests/test_variance.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `icc(y, groups) -> dict` with keys `icc`, `s2_between`, `s2_within`, `eta2_naive`, `k`, `n`; `decompose(df, target="forward_z", seed=0, n_boot=1000) -> dict` with keys `var_total`, `site`, `cluster`, `cohort`, `icc_ci`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_variance.py`:

```python
"""The estimator behind the spec's headline number, and why it is not naive eta-squared.

E01 puts the field effect at 34.5% of forward_z variance. Naive eta-squared reports 36.0% on the
same data because a group mean over n_i observations carries sigma_eps^2 / n_i of pure noise that
eta-squared credits to the group. On small groups that gap is large, and the whole argument is a
share of variance, so the correction is load-bearing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from argotech.lab.variance import decompose, icc


def _planted(icc_true: float, n_groups: int = 60, n_obs: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    s_between = np.sqrt(icc_true)
    s_within = np.sqrt(1 - icc_true)
    alpha = rng.normal(0, s_between, n_groups)
    g = np.repeat(np.arange(n_groups), n_obs)
    return alpha[g] + rng.normal(0, s_within, n_groups * n_obs), g


def test_icc_recovers_a_planted_value():
    y, g = _planted(0.35)
    assert abs(icc(y, g)["icc"] - 0.35) < 0.06


def test_icc_is_zero_when_groups_carry_no_signal():
    y, g = _planted(0.0)
    assert icc(y, g)["icc"] < 0.03


def test_naive_eta_squared_overstates_it_on_small_groups():
    y, g = _planted(0.20, n_groups=80, n_obs=3)
    r = icc(y, g)
    assert r["eta2_naive"] > r["icc"] + 0.05, \
        "the small-group bias this estimator exists to correct did not appear"


def test_a_single_group_yields_no_estimate():
    assert np.isnan(icc(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0]))["icc"])


def test_decompose_reports_every_grouping_with_an_interval():
    y, g = _planted(0.30, n_groups=30, n_obs=12)
    df = pd.DataFrame({"forward_z": y, "site_id": [f"S{i}" for i in g],
                       "cluster": [f"C{i % 3}" for i in g],
                       "label_date": [f"2025-{1 + i % 12:02d}-01" for i in range(len(y))]})
    out = decompose(df, n_boot=200)
    assert abs(out["site"]["icc"] - 0.30) < 0.08
    lo, hi = out["icc_ci"]
    assert lo < out["site"]["icc"] < hi
    assert out["var_total"] > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_variance.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'argotech.lab.variance'`

- [ ] **Step 3: Write the implementation**

Port the probe at `experiments/E01_variance_decomposition.py` into `src/argotech/lab/variance.py`, keeping the `icc` function verbatim (rename its `N` key to `n` to match the interface above) and wrapping the reporting section as `decompose(df, target="forward_z", seed=0, n_boot=1000) -> dict`. `decompose` returns:

```python
{"var_total": float(df[target].var()),
 "site":    icc(df[target], df["site_id"]),
 "cluster": icc(df[target], df["cluster"]),
 "cohort":  icc(df[target], df["cluster"].astype(str) + "|" + df["label_date"].astype(str)),
 "icc_ci":  (lo, hi)}          # percentile CI over n_boot site-level resamples
```

Bootstrap by resampling **sites** with replacement, not rows — the site is the independent unit, and resampling rows would treat 39 observations of one field as 39 independent facts.

- [ ] **Step 4: Run the tests and reproduce E01's committed numbers**

Run:
```bash
pytest tests/test_variance.py -q
python -c "
import pandas as pd, json
from argotech.lab.variance import decompose
r = decompose(pd.read_parquet('data/training_set.parquet'))
print(round(r['site']['icc'], 4), [round(x, 4) for x in r['icc_ci']], round(r['cluster']['icc'], 4))
"
```
Expected: 6 passed; the reproduction prints `0.345 [0.2357, 0.4353] 0.0` — matching `experiments/E01_variance_decomposition.out`. If the site ICC differs from 0.3450, stop: the port changed the estimator and the spec's headline number no longer traces to the code.

- [ ] **Step 5: Commit**

```bash
git rm experiments/E01_variance_decomposition.py
git add src/argotech/lab/variance.py tests/test_variance.py
git commit -m "Promote E01 from a probe to tested library code

The spec cites 34.5% CI [0.236, 0.435] as its headline number, so the
estimator that produced it needs a test that fails if it drifts. The
planted-ICC tests pin recovery; the small-group test pins the reason
this is a moment estimator and not naive eta-squared, which reports 36.0%
on the same data by crediting the group with its own sampling noise.

The .out file stays: it is the record of the run the spec quotes."
```

---

### Task 8: Run E02 — the gate

Not a code task. This is the experiment the plan exists to make possible, and its result decides whether Plans 2 and 3 get written.

**Files:**
- Create: `experiments/E02-within-vs-level.result.json` and three sibling result files
- Create: `experiments/E02-level-z.yaml`, `experiments/E02-within-xy.yaml`, `experiments/E02-delta-z.yaml` (copies of the E02 config differing only in `target` and `name`)
- Modify: `docs/superpowers/specs/2026-08-26-two-way-demeaned-estimand-design.md` (§10 result row)

- [ ] **Step 1: Run all four target kinds**

```bash
for k in level-z within-vs-level within-xy delta-z; do
  python -m argotech.lab.run experiments/E02-$k.yaml
done
```

- [ ] **Step 2: Read the result against the criterion**

The design is supported if, on the **spatial** folds, `within_y` or `within_xy` beats `zero` on mean net benefit **with non-overlapping bootstrap intervals**. Record all four kinds regardless of which wins.

Three outcomes, and the third is the one to watch for:

| Outcome | Reading | Next |
| --- | --- | --- |
| A demeaned arm beats `zero`, intervals disjoint | The reformulation carries signal | Write Plan 2 (serving) and Plan 3 (loop) |
| No arm beats `zero` | The residual is not learnable at this resolution | Recommend shipping `domain/` alone; the paper is the negative result plus E01's mechanism |
| `level_z` beats `zero` too | The gate is measuring the transform, not the target | Stop and diagnose — `zero` must be exactly climatology under `within_*`, and Task 4's equivalence test says it is |

- [ ] **Step 3: Record the result in the spec**

Append the measured summary table to §10 beside E02's row, in the same format §2 uses for E01, citing the four `.result.json` files. Do not restate a number the JSON does not contain.

- [ ] **Step 4: Commit**

```bash
git add experiments/ docs/superpowers/specs/
git commit -m "Run E02: <one line stating what the four targets scored>"
```

---

### Task 9: Retire `training/train.py` — CONDITIONAL on Task 8 outcome A or B

Do not start this task until Task 8 is committed. If Task 8 returned outcome C, diagnose first.

**Files:**
- Delete: `src/argotech/training/train.py`, `src/argotech/training/__init__.py`
- Move: `src/argotech/training/embed.py` → `src/argotech/lab/embed.py`
- Modify: `tests/test_features.py`, `tests/test_train_arms.py` (they import `argotech.training.train`)
- Modify: `README.md` (the Training section), `docs/model-design.md` (§7 layout block)

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: no `argotech.training` package.

- [ ] **Step 1: Find every remaining reference**

```bash
grep -rn "argotech.training\|training/train\|training\.train" src tests docs README.md .github
```

- [ ] **Step 2: Migrate the two dependent tests**

`tests/test_features.py` imports `rank_correlation` and others from `training.train`; `tests/test_train_arms.py` imports `make_model`, `make_regressor`, `ARMS`, `PRODUCTION_ARM`. Repoint the first at `argotech.lab.evaluate.spearman`. Delete `tests/test_train_arms.py` — it pins the classifier arms and the production-artifact guard, both of which this plan retires; the replacement coverage is `tests/test_arms.py`.

Run: `pytest -q` after each edit.

- [ ] **Step 3: Delete the package and move the embedder**

```bash
git mv src/argotech/training/embed.py src/argotech/lab/embed.py
git rm src/argotech/training/train.py src/argotech/training/__init__.py
grep -rl "argotech.training.embed" src tests | xargs sed -i '' 's/argotech\.training\.embed/argotech.lab.embed/g'
pytest -q && ruff check src tests
```

Expected: full suite green, ruff clean, `src/argotech/training/` gone.

- [ ] **Step 4: Update the docs**

In `README.md`, replace the Training section's `python -m argotech.training.train` with `python -m argotech.lab.run experiments/<name>.yaml` and state that experiments are config files. In `docs/model-design.md` §7, replace `training/` with `lab/` in the layout block and add a one-line pointer to the new spec above §9.1.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Retire the incumbent training entrypoint

684 lines holding dataset loading, four arms, two split strategies, five
metrics, a decision-rule sweep and a CLI. Its replacement is six modules
with 37 tests between them, and the parts worth keeping — the Sentinel
backfill, the 429 retry, the empty-cache guard — moved by git mv in Task
2 rather than being retyped."
```

---

## Deferred to later plans

Written down so they are not mistaken for gaps:

- **Plan 2 (serving)** — AOA gating, conformal intervals, `alpha_hat` maintained on `field_features`, the extended artifact contract, ONNX export. Spec §8.
- **Plan 3 (the loop)** — panel append from the nightly job, scheduled `store.label_join`, CI release gate, MLflow registry stages, drift monitoring. Spec §9.
- **Fairness slices** — spec §7 requires net benefit and ranking sliced by household headship, landholding size and district, with a slice regression blocking promotion. Deferred to Plan 3 because it gates *promotion*, and nothing is promoted by this plan. The `domain/risk.py` protected-attribute guard is untouched and still active throughout.
- **E03–E08** — shrinkage sweep, radar/Presto re-ablation, linear-vs-boosted under the new target, decision curves at product cost ratios, conformal coverage, the farm external set. Spec §10. E03 has a numeric objective from E01: drive realised variance reduction from 19.9% toward 34.5%.
- **The farm census** — spec §5 makes it blocking for E08, not for E02.
