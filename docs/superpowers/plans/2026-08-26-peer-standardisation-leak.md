# Peer Standardisation Leak — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fit the peer reference per fold instead of over the whole frame, and measure whether a transferable cohort key recovers what that costs.

**Architecture:** `ndvi_z_peer` / `rvi_z_peer` stop being static panel columns and become a fold-fitted transform: fit on training rows, apply to train and test. Two cohort keys are offered — the existing `cluster|MM` and a transferable geographic key — plus a `leaky` control that reproduces today's whole-frame behaviour, so the comparison measures how much of the model's skill came from the leak.

**Tech Stack:** Python 3.11+, pandas, numpy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-26-two-way-demeaned-estimand-design.md`

## Global Constraints

- Python >=3.11. Add NO new dependency.
- Ruff passes, `line-length = 110`. Use `venv/bin/python3 -m pytest` — `venv/bin/pytest` has a stale shebang.
- `src/argotech/domain/` must not be modified.
- **`src/argotech/serving/` must not be modified.** Serving standardises one live row against constants snapshotted into the artifact. At inference time there is no future, so whole-frame stats are correct there. This leak is a lab-only defect.
- `src/argotech/features/agronomic.py`: `peer_stats`, `peer_bucket`, `peer_reference` and `peer_z` keep their current behaviour and signatures — serving and the incumbent both call them. New behaviour goes in `lab/`, which calls them differently.
- Full suite green after each task: currently 151 passed, 1 skipped.
- The reproduction gate must keep printing `0.345 [0.2357, 0.4353] 0.0`.

## Measured facts this plan rests on (recomputed, do not re-derive)

- 42 of 45 `cluster|MM` buckets span multiple years; `Benue_River_Basin|01` pools 2023-01-08 … 2026-01-22.
- Across the three forward-chaining boundaries, **74.0% / 49.9% / 24.3%** of each training row's peer cohort lies at or after the boundary (max 94.2%).
- Under leave-one-cluster-out the held-out cluster's buckets draw from exactly one cluster: itself.
- Fixed geographic bands, donor rows available to each held-out cluster: 5°/500 m → `[0, 282, 1098, 0]`; 10°/1000 m → `[36, 1353, 1098, 647]`; **20°/1000 m → `[2175, 2799, 1792, 3139]`**. Cluster means: Benue 7.6°N/118 m, Kaduna 10.8°N/676 m, Kano 12.0°N/455 m, Kenya 0.5°N/1760 m.

---

### Task A: `lab/peers.py` — the fold-fitted transform

**Files:** Create `src/argotech/lab/peers.py`; Test `tests/test_peers.py`

**Interfaces produced:**
- `PEER_COLUMNS: tuple` = `(("ndvi", "ndvi_z_peer"), ("rvi", "rvi_z_peer"))`
- `peer_key(df, kind, lat_band=20.0, elev_band=1000.0) -> pd.Series` — `"cluster_month"` or `"geo_month"`
- `fit_peer_stats(train, kind, **bands) -> dict` — `{key: {column: [mu, sigma]}}`, from training rows only
- `apply_peer_z(df, stats, kind, **bands) -> tuple[pd.DataFrame, float]` — returns the frame with the two z columns rewritten, and the coverage fraction (share of rows that got a reference)

- [ ] **Step 1: Write `tests/test_peers.py`**

```python
"""The peer reference, fitted where it can be fitted honestly.

panel.py computed `peer_stats` over the whole frame and baked the result in as a static column,
while `peer_stats`'s own docstring said it must be computed per fold. Measured consequences: 74% of
an early training fold's peer cohort lay in its own future, and a held-out cluster was standardised
entirely on held-out rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from argotech.lab.peers import apply_peer_z, fit_peer_stats, peer_key


def _df() -> pd.DataFrame:
    rows = []
    for cluster, lat, elev in (("A", 7.0, 100.0), ("B", 11.0, 700.0), ("C", 0.5, 1800.0)):
        for year in (2024, 2025):
            for month in (1, 6):
                for i in range(6):
                    rows.append({"cluster": cluster, "latitude": lat, "elevation": elev,
                                 "site_id": f"{cluster}{i}",
                                 "obs_date": f"{year}-{month:02d}-1{i}",
                                 "ndvi": 0.3 + 0.02 * i + (0.2 if cluster == "C" else 0.0),
                                 "rvi": 0.4 + 0.01 * i})
    return pd.DataFrame(rows)


def test_cluster_month_key_is_cluster_and_month():
    df = _df()
    assert peer_key(df, "cluster_month").iloc[0] == "A|01"


def test_geo_month_key_drops_cluster_identity():
    keys = peer_key(_df(), "geo_month", lat_band=20.0, elev_band=1000.0)
    assert "A" not in keys.iloc[0] and "|01" in keys.iloc[0]


def test_geo_month_lets_a_held_out_cluster_borrow_peers():
    """The whole point: an unseen region must still get a reference from similar places."""
    df = _df()
    for kind, expect in (("cluster_month", False), ("geo_month", True)):
        train, test = df[df.cluster != "A"], df[df.cluster == "A"]
        _, coverage = apply_peer_z(test, fit_peer_stats(train, kind), kind)
        assert (coverage > 0) is expect, f"{kind}: coverage {coverage}"


def test_cluster_month_gives_a_held_out_cluster_no_reference_at_all():
    df = _df()
    train, test = df[df.cluster != "A"], df[df.cluster == "A"]
    out, coverage = apply_peer_z(test, fit_peer_stats(train, "cluster_month"), "cluster_month")
    assert coverage == 0.0
    assert out["ndvi_z_peer"].isna().all(), "no reference must mean NaN, never a fabricated 0.0"


def test_stats_fitted_on_train_ignore_test_rows_entirely():
    """The leak, stated as a property: a test row's value must not move the reference."""
    df = _df()
    train, test = df[df.cluster != "A"], df[df.cluster == "A"]
    before = fit_peer_stats(train, "geo_month")
    poisoned = df.copy()
    poisoned.loc[poisoned.cluster == "A", "ndvi"] = 99.0
    after = fit_peer_stats(poisoned[poisoned.cluster != "A"], "geo_month")
    assert before == after, "a held-out row changed the reference fitted on training rows"


def test_apply_rewrites_both_peer_columns():
    df = _df()
    out, coverage = apply_peer_z(df, fit_peer_stats(df, "cluster_month"), "cluster_month")
    assert coverage == 1.0
    for col in ("ndvi_z_peer", "rvi_z_peer"):
        assert col in out.columns and out[col].notna().any()
    assert abs(out["ndvi_z_peer"].mean()) < 0.5


def test_coverage_reports_the_share_with_a_reference():
    df = _df()
    train = df[df.cluster != "A"]
    _, coverage = apply_peer_z(df, fit_peer_stats(train, "cluster_month"), "cluster_month")
    expected = 1.0 - len(df[df.cluster == "A"]) / len(df)
    assert abs(coverage - expected) < 1e-9


def test_a_single_observation_bucket_yields_no_reference():
    """sd over one row is NaN; a reference we cannot measure must not be invented."""
    one = _df().head(1)
    stats = fit_peer_stats(one, "cluster_month")
    out, coverage = apply_peer_z(one, stats, "cluster_month")
    assert coverage == 0.0 and out["ndvi_z_peer"].isna().all()
```

- [ ] **Step 2: Run it, confirm it fails** (`ModuleNotFoundError: argotech.lab.peers`)

- [ ] **Step 3: Write `src/argotech/lab/peers.py`**

Reuse `argotech.features.agronomic.peer_z` and `peer_reference` for the arithmetic — do not reimplement them. `fit_peer_stats` mirrors `peer_stats`'s output shape but takes the key kind. Bands are applied as `floor(value / band) * band` so the cut points are fixed constants, never data-derived quantiles — a quantile fitted over the frame would put the leak back in through the binning.

A bucket whose standard deviation is NaN or zero yields no reference, and `peer_z` must return NaN rather than 0.0 — `peer_reference`'s docstring already refuses that fabrication for the same reason.

- [ ] **Step 4: Verify tests pass, ruff clean, full suite green**
- [ ] **Step 5: Commit**

---

### Task B: wire the transform into the fold loop

**Files:** Modify `src/argotech/lab/run.py`, `experiments/E02-within-vs-level.yaml`; Test `tests/test_run.py`

The ordering changes and this is the substantive part. `alpha_hat` is computed from `ndvi_z_peer`, so if the feature is fold-fitted then so is the target. `build_target` therefore moves **inside** the fold loop:

```
for each split protocol, for each (fold_name, train_raw, test_raw):
    if peer_key == "leaky":  train, test = train_raw, test_raw     # panel's baked column, the control
    else:
        stats = fit_peer_stats(train_raw, peer_key, **bands)
        train, _        = apply_peer_z(train_raw, stats, peer_key, **bands)
        test, coverage  = apply_peer_z(test_raw,  stats, peer_key, **bands)
    train_t, features = build_target(train, cfg["target"], ...)
    test_t,  _        = build_target(test,  cfg["target"], ...)
    score every arm on (train_t, test_t); record peer_coverage
```

- [ ] **Step 1: Add `peer_coverage` to each fold's result dict** and to the printed/logged summary, so a NaN reference is visible rather than silent. `leaky` records `1.0`.
- [ ] **Step 2: Add config keys** `peer_key` (`leaky` | `cluster_month` | `geo_month`, default `geo_month`), `lat_band` (default 20.0), `elev_band` (default 1000.0). `load_config` supplies the defaults.
- [ ] **Step 3: Write tests** asserting: a `leaky` run reproduces the pre-change numbers; a `cluster_month` spatial run reports `peer_coverage == 0.0` for every fold; a `geo_month` spatial run reports coverage > 0; and the three configs produce *different* summaries (if they do not, the transform is not wired in).
- [ ] **Step 4: Full suite, ruff, commit.**

Do NOT run E02.

---

### Task C: retire the superseded justification and record the method

**Files:** Modify `docs/superpowers/specs/2026-08-26-two-way-demeaned-estimand-design.md`, `docs/RESEARCH_SUMMARY.md`

- [ ] **Step 1:** `docs/RESEARCH_SUMMARY.md:53-57` currently argues the whole-frame peer reference is admissible because it is label-free ("That addresses label leakage; it does not address transduction over the feature distribution, and I would not defend it as settled"). That position is superseded. Replace it with what is now measured and done: the temporal figures above, the spatial finding, and the per-fold fit that replaces it. Keep the paragraph's honest register — it is describing a corrected defect, not claiming there was never one.
- [ ] **Step 2:** Add a spec section covering the fold-fitted reference, the three cohort keys, the `peer_coverage` metric, and the measured transfer-versus-specificity trade-off (donor rows per band width). State plainly that a cohort tight enough to be agronomically meaningful does not transfer to an unseen region at this sample size, and that band width is therefore a swept parameter rather than a chosen constant. This answers §7's open question "How should the reference cohort be defined so that it transfers to an unsampled region?" — cross-reference it.
- [ ] **Step 3:** Amend `peer_stats`'s docstring in `features/agronomic.py` — it claims "Computed per fold during evaluation", which was never true of any caller. Say where each caller stands now. **This is a docstring-only edit; do not change its behaviour.**
- [ ] **Step 4:** Commit.
