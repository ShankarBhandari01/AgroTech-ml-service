# One definition of the peer anomaly

**Date:** 2026-08-21
**Status:** approved, not yet implemented
**Affects:** `features/agronomic.py`, `training/dataset.py`, `training/train.py`, `models/registry.py`, `serving/pipeline.py`, the artifact bundle contract

## Problem

`ndvi_z_peer` and `rvi_z_peer` are different quantities at training and at serving. Both are in
`MODEL_FEATURES`; `ndvi_z_peer` is the top-ranked feature by permutation importance, and it is also
the entire input to the persistence hazard path.

- **Training** (`dataset.py`) standardises against the **concurrent cross-site cohort** — other
  sites in the same cluster on the same sensing date, self excluded. This is the same construction
  as the label.
- **Serving** (`pipeline.py`) passes the site's **own 12-month history** as the peer list. That is a
  temporal self-anomaly.

`satellite_block`'s own docstring says `peer_ndvi` is "the concurrent cohort, excluding this site."
Serving hands it a temporal series. One column name, two meanings.

This is the defect class the architecture was built to prevent — `features/agronomic.py` exists so
that "a feature cannot mean two different things if it is only defined once" — and it survived
because the function is shared while its *arguments* are not.

## Evidence

Found independently by two reviewers on 2026-08-21, with separate measurements.

Computing both conventions on identical inputs over the training samples with a real cohort:

| | mean | sd | vs the other |
| --- | ---: | ---: | --- |
| training convention (per-date cohort) | −0.008 | 1.172 | — |
| serving convention (self-history) | −0.057 | 1.224 | r = 0.626, **25.5% sign flips** |

The marginals nearly match, which is precisely why it is invisible. A quarter of fields flip sign.

**Failure scenario.** Two neighbouring fields in Kano, both at NDVI 0.42 in a district-wide dry
month. The training feature reads ≈ −0.1: both fields are normal *for the district*. The serving
feature reads ≈ −1.8: both are far below their own annual peak. The model was fitted to read −1.8 as
"collapsing relative to neighbours", so every field in the district is flagged at once — the exact
failure peer standardisation exists to prevent. The vegetation hazard tracks the calendar, not
the field.

It also means the shipped persistence baseline is not the one that was measured: persistence feeds
`ndvi_z_peer` straight into a logistic centred at `SEVERE_ANOMALY_Z = -1.0`, a constant calibrated
on the cohort scale.

## Approach

**A′ — one definition on both sides: standardise against a (cluster, calendar-month) snapshot.**

Rejected alternatives:

- **A (serving-only snapshot, training keeps its per-date cohort).** Measured: agreement rises from
  r = 0.626 to r = 0.743, 8.7% sign flips. Better, but still two definitions of one column. Shrinking
  a silent defect is how it survives the next review.
- **B (concurrent cohort computed by the nightly job).** Semantically faithful — a genuinely
  concurrent cohort, which is what the label is standardised against. Rejected for now because it
  makes `precompute` load-bearing for correctness rather than latency, leaves the live-path fallback
  with no cohort at all, and needs a registered farmer base dense enough per cluster to be
  meaningful. Recorded below as the upgrade.
- **C (change training to the self-history convention).** One line, but it points the features at a
  self-history reference while the label stays peer-relative. Trades a train/serve skew for a
  feature/label mismatch.

**What A′ changes about meaning.** The feature stops being "how does this field compare to its
neighbours *right now*" and becomes "…to what its district typically looks like this month". The
concurrent-cancellation property is lost. That is a real cost, accepted because a feature that means
one thing beats a feature that means the right thing in one half of the system.

**A second defect closes for free.** `ndvi_z_peer` currently has no minimum-cohort guard while the
label has one (`len(peers_future) >= 5`). A three-peer cohort produced **z = −80.2**, inflating the
column's sd from 1.18 to 1.52 and contaminating the climatology baseline through
`add_site_climatology`'s expanding mean. Every (cluster, month) bucket holds at least 9 samples, so
the pathological case cannot arise.

## Design

### 1. The statistic

```python
PEER_RELATIVE = ["ndvi", "rvi"]

peer_stats(df) -> {"Kaduna_Grain_Belt|09": {"ndvi": [mu, sigma], "rvi": [mu, sigma]}, ...}
```

The same shape `cluster_stats` already returns, with the bucket key extended by calendar month.
Plain lists, not a DataFrame: this crosses a joblib boundary and then a `predict` hot path, and must
not drag pandas into either — the reasoning `cluster_stats`'s docstring already records.

`satellite_block` and `radar_block` stop taking a peer *list* and take the `[mu, sigma]` pair. Both
callers look it up from the same table; neither computes it.

### 2. NaN discipline, inherited rather than reinvented

An unknown cluster, an unseen month, or a bucket with a degenerate sigma yields **NaN**, never 0.0.
`cluster_relative_row` already argues this and the argument transfers unchanged: 0.0 asserts
"exactly average for its region", and inventing that for a region never measured is the same
fabrication `radar_block` refuses for a field with no radar pass. `HistGradientBoosting` splits on
NaN natively, so the model degrades to the raw features rather than being lied to.

### 3. Within-fold statistics for evaluation

Peer stats must be computed from the **training folds only**. Computing them over the full frame
would standardise a held-out cluster against itself — a new leak of exactly the shape already
recorded against the `_cz` twins.

**Consequence, stated plainly: a held-out cluster has no bucket, so its peer features arrive NaN and
the model loses its top-ranked feature in every leave-one-cluster-out fold. The corrected numbers
will be worse than those currently reported, and they will be the first honest ones.** Current
figures for comparison (artifact `agro-20260821T065051Z`, seed 42, spatial): regressor P@25 0.760,
rho 0.377, macro F1 0.441.

The final artifact snapshots stats over the full frame, which is correct — at serving there is no
held-out anything.

### 4. Artifact contract

The bundle gains `peer_stats`. `ModelManager` requires it and raises `ValueError` when absent, the
way it now requires `target`. A bundle without it predates this change, and serving cannot
standardise without it; the failure must be loud, since the silent alternative is every field
standardised against nothing.

### 5. The label is deliberately untouched

`forward_z` keeps its per-date cohort. The label is never computed at serving, so it carries no
skew, and its concurrent construction is what makes it a peer-relative target worth predicting. The
asymmetry — climatological-peer features, concurrent-peer label — is the point, not an oversight.

### 6. Testing

The test whose absence let this defect live:

> for the same input, `dataset.py` and `pipeline.py` produce the **same** `ndvi_z_peer` and
> `rvi_z_peer`, row by row.

`tests/test_cluster_relative.py` pins exactly this property for the `_cz` twins — including an
anti-vacuity guard asserting the three-way split is actually exercised — and those twins never
drifted. This test is modelled on it directly.

Also required: a held-out cluster's peer features are NaN under within-fold stats; an unknown
cluster or month yields NaN and not 0.0; the artifact contract test extended to `peer_stats`.

## Out of scope

- **Approach B, the concurrent cohort.** The endpoint once the registered base is dense enough per
  cluster. `precompute.run` already visits every field and could build `(cluster, date) → [ndvi]` in
  one pass.
- **`cluster_stats` within-fold (the `_cz` twins).** The same leak, in the sibling mechanism, and
  the change is nearly free once `train.py` is already computing per-fold statistics. **Recommended
  to fold into the implementation plan**; called out here so the decision is explicit rather than
  incidental.
- **`decide()` taking quantiles from the test fold**, and **forward-chaining without a label
  embargo** — both measured as inflating results by roughly 2–3 points; both need `label_date`,
  which now exists but is inert.
- **The `climatological_rain_30` NaN escape route** introduced by the previous fix wave.
