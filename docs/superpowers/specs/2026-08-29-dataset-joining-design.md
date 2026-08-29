# Joining the datasets: what may be combined, at which key, for which estimand

**Status:** design. Supersedes nothing; formalises rules that E01–E06 established by measurement.

## What this decides

The project now has three data sources that do **not** share a key or a spatial resolution. This
spec fixes (a) which tier each source belongs to, (b) which estimands each tier is *admissible*
for, and (c) the order operations must run in so no join reintroduces one of the three leaks
already found and fixed.

The governing claim: **combining these is not a feature concatenation.** A single wide table is the
failure mode, not the goal.

---

## 1. The three tiers

| Tier | Key | Cardinality | Contents | File |
| --- | --- | --- | --- | --- |
| **T1** field × time | `site_id` × `obs_date` | 4,596 rows, 122 sites, 4 clusters, 2022-09-10 → 2026-06-21 | 28 time-varying: weather, FAO-56 water balance, phenology, Sentinel-2 optical, Sentinel-1 radar | `data/training_set.parquet` |
| **T2** field, static | `site_id` | 122 rows | `clay sand silt soc phh2o cec bdod` (SoilGrids v2.0), `slope_percent`; `elevation`/`latitude`/`longitude` already inline in T1 | `data/site_covariates.parquet` |
| **T3** EA × season | EA id × season | not yet acquired | LSMS-ISA household and plot modules: assets, plot area, realised yields | World Bank microdata (see the E07/E08 plan) |

T1 carries `label_date = obs_date + 30 days`. That 30-day lead is the forecast horizon and it sets
the embargo width in §5.

---

## 2. The rule: the estimand decides admissibility

A column is admissible for an estimand only if the estimand has not already annihilated the
variation that column could explain. This is not a modelling preference; for T2 it is arithmetic.

Let `x_i` be a T2 column: constant within field `i` by construction (soil and slope do not change
over the panel window).

| Estimand | What happens to a T2 column | Admissible? |
| --- | --- | --- |
| `level_z` | untouched; explains between-field variation | **Yes — this is T2's only home** |
| `within_y` | survives in `X`, but `y` has had its field mean removed, so the only component `x_i` could explain is now identically zero | **No** — it can fit noise and nothing else |
| `within_xy` | demeaned by field → exactly zero → dropped | **No** — mechanically impossible |
| `delta_z` | differences across time cancel a constant | **No** |

`within_xy` makes this visible; the runner prints it on every run:

```
[targets] within_xy: dropped time-invariant column(s) (demean to zero): elevation
```

**Consequence.** Soil and terrain can only ever address the **between**-field component — the field
effect, measured at 34.5% [0.2357, 0.4353] of variance and found entirely within-cluster
(between-cluster ICC 0.0000). That is exactly E06's question, and it is the only question static
data is capable of answering. No amount of soil data will improve within-field ranking.

**The `within_y` row is the trap.** Nothing errors, nothing is dropped, the model trains, and a
metric comes back. A T2 column under `within_y` is a pure overfitting surface. Excluding it must be
explicit, not left to the demeaning to catch — because under `within_y` the demeaning does not
catch it.

---

## 3. T1 ⋈ T2 — the only field-level join

```
panel = T1.merge(T2, on="site_id", how="left", validate="many_to_one")
```

Required, in this order:

1. `validate="many_to_one"` — a duplicated `site_id` in T2 silently multiplies panel rows. Assert it.
2. **Reconcile `cluster`.** It appears in both. Assert equality and drop T2's copy; a mismatch means
   the two files were built from different site lists and the join is meaningless.
3. **Assert no row loss and record the null rate.** SoilGrids has genuine gaps (water bodies,
   incomplete depth stacks). A left join turns a missing site into silent NaNs; the count belongs in
   the run's provenance block, not in a log line.
4. **Tag the joined columns as static** so §2's admissibility gate can act on them by provenance
   rather than by name matching.

T2 must **not** be merged into `data/training_set.parquet` on disk. The 38 committed experiment
provenance hashes are content hashes of that file; rewriting it invalidates every one. Join at use
time, in memory. This mirrors the rule the E07/E08 plan already states.

---

## 4. T3 — never joins at field resolution

Public LSMS-ISA releases enumeration-area centroids, displaced 0–5 km (rural, 99%) or 0–10 km (1%).
T1's satellite features are a 500 m box. **The displacement is ten times the feature footprint**, so
the pixel at a published coordinate is in general not that household's pixel.

Therefore:

- **Forbidden:** joining a T3 household to a T1 site, or to any pixel, by nearest neighbour or radius.
  A row produced this way is noise wearing a farm's name.
- **Required:** aggregate T1 upstreams to the EA scale *first*, then join T3 to that. The unit of
  analysis becomes the EA-season, not the field-day.

The EA-scale panel is a separate build with its own key (`ea_id × season`), not a variant of T1. It
re-fetches the same weather and Sentinel upstreams over the EA's plausible support rather than a
500 m box. Read *Privacy Protection, Measurement Error, and the Integration of Remote Sensing and
Socioeconomic Survey Data* (arXiv:2202.05220) before implementing it; the error is quantified there.

**Wave 5 caveat:** GHS-Panel Wave 5 (post-planting Jul–Sep 2023, post-harvest Jan–Mar 2024) overlaps
T1's window, but its geovariables are listed as *forthcoming*. Without them Wave 5 supports the
non-spatial half (household attributes → exposure and vulnerability) and nothing spatial.

---

## 5. Ordering, so joins do not reintroduce a leak

All three leaks found in this project came from a reference statistic computed over rows the model
should not have seen. A join is an opportunity to repeat that. The order is fixed:

1. **Split first.** Spatially blocked (leave-one-cluster-out) and/or purged forward chaining:
   train on `label_date < boundary`, test on `obs_date >= boundary`, with a 30-day embargo equal to
   the label lead.
2. **Fit every reference on the training side only** — `fit_peer_stats`, `alpha_hat`, and any
   standardisation of T2 or T3 columns. Apply to test; never refit.
3. **Join T2** (static, no fitted quantity — safe at any point, but do it after the split so any
   later standardisation of it inherits the fold).
4. **Derive the estimand**, which applies §2's gate.

A T2 column standardised over the whole frame is the same defect as the peer-standardisation leak,
in a new place: the test fold's soil values would inform the training fold's mean.

---

## 6. Which tier serves which risk term

```
risk = Hazard × Exposure × Vulnerability
         │         │            │
      T1 (+T2      T3 + backend farmer payload
      between-     ────────────┬───────────────
      field        the two terms with no data today
      baseline)
```

This is the actual integration design: **each source feeds the term it can address at the resolution
it has.** T3's value is not that it improves hazard — it cannot reach field resolution — but that it
is the only available evidence for exposure and vulnerability, which today are unvalidated.

Weighted by measured impact, that is also where the value is: `expected_loss` swings ~$3,531 across a
plausible exposure range against $450 for the full observed hazard range and $225 for vulnerability
(notebook 05, §4). And §9.3 of `model-design.md` shows hazard is pinned at exactly 1.0 for 59.6% of
the panel, so for most fields the ranking is already driven by exposure regardless.

---

## 7. Acceptance checks

Each must fail if the rule it names is broken. A test that passes against broken code is worse than
no test — four did so earlier in this project.

- [ ] T2 column present in a `within_xy` design matrix → the run fails, not warns.
- [ ] T2 column present under `within_y` → the run fails. **Assert this with an arm that actually
      reads `test[features]`**; a `zero` baseline never evaluates its features and will pass
      regardless.
- [ ] Duplicate `site_id` in T2 → the merge raises, not silently fans out rows.
- [ ] `cluster` disagreement between T1 and T2 → raises.
- [ ] Peer/reference statistics fitted after a T2 join produce **identical** values to fitting before
      it — proving the join introduced no cross-fold information.
- [ ] Any nearest-neighbour join from T3 to a field or pixel → rejected in review by §4.

## 8. Open

- The EA-scale rebuild (§4) is unspecified beyond its key and its prohibition; it needs its own
  design once Wave 5 geovariable availability is known.
- Whether `level_z` remains a reported estimand at all. E05 found that removing the field effect
  removes the temporal skill with it, so `level_z` results are largely a restatement of the field
  effect — which is the thing T2 is being asked to explain. If E06 explains it, `level_z` becomes
  interpretable; if not, `level_z` skill should be read as "the model memorised which fields are
  bad", not as agronomic prediction.
