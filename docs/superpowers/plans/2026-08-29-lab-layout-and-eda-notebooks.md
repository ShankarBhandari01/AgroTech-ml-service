# Reorganise `lab/`, and add the EDA notebooks

**Goal:** 13 flat modules (2,718 lines) grouped by role, and notebooks that actually interrogate the
panel — because nobody has, and that is how NDVI < 0 sat undetected in 5% of rows until this week.

## Global Constraints
- **Behaviour must not change.** Every move is `git mv` + repointed imports. No logic edits.
- `tests/test_import_boundary.py` must pass at every commit — it is the layering guard.
- Full suite green: currently **185 passed, 1 skipped**. Ruff clean apart from 2 pre-existing errors.
- Reproduction gate must still print `0.345 [0.2357, 0.4353] 0.0`.
- No new runtime dependency. `[project.dependencies]` byte-identical.
- Notebooks read **committed** files only — `data/training_set.parquet` and `experiments/csv/*.csv`.
  Never re-fetch, never write.

---

### Task 1 — group `lab/` by role

| subpackage | modules |
| --- | --- |
| `lab/panel/` | `panel.py` |
| `lab/estimand/` | `targets.py`, `peers.py` |
| `lab/eval/` | `splits.py`, `evaluate.py`, `variance.py` |
| `lab/arms/` | `arms.py`, `export.py` |
| `lab/presto/` | `embed.py`, `embeddings.py`, `_presto_vendored.py` |
| stays at top | `run.py` (the entrypoint), `__init__.py` |

- [ ] `git mv` every file; add `__init__.py` per subpackage. **History must survive** — the moved
      files carry a 429-retry fix, an empty-cache guard and a leak fix whose provenance matters.
- [ ] Repoint imports across `src/` and `tests/`. Grep; do not guess. `run.py` imports most of these.
- [ ] Update `pyproject.toml`'s ruff `extend-exclude` for `_presto_vendored.py`'s new path — it is
      kept byte-comparable to the upstream release so it can be re-vendored by diff, and
      reformatting it destroys that.
- [ ] `experiments/*.yaml` and `docs/` reference module paths — grep and fix.
- [ ] **The Presto isolation is the point**: `presto/` is 1,136 lines (42% of `lab/`) and the only
      torch-dependent code. After the move, confirm nothing outside `lab/presto/` imports torch.
- [ ] Suite, ruff, gate, boundary test, commit.

### Task 2 — `notebooks/02-panel-eda.ipynb`
The panel has never been examined. Write the notebook that would have caught what we found late.

- [ ] Coverage: rows per site per cluster, observation gaps, the exact-30-day lag, date range.
- [ ] **Missingness by column**, and by cluster — a hole that follows cluster boundaries is the one
      shape that invalidates leave-one-cluster-out.
- [ ] **NDVI validity**: the distribution, the 233 rows (5.07%) below 0, the two at exactly −1.0
      (NIR = 0, a no-data sentinel). Show why these are not vegetation.
- [ ] `forward_z` tails: range −11.35 to +7.08 on a quantity that is a z-score by construction, and
      the 8× enrichment of |z|>3 rows for a negative label-date NDVI.
- [ ] Seasonality: NDVI by calendar month per cluster — the growing seasons differ, which is what
      the peer cohort's month key is for.

### Task 3 — `notebooks/03-the-field-effect.ipynb`
The central research object, and the groundwork for E06.

- [ ] Reproduce the variance decomposition: field-effect ICC **0.345**, CI **[0.2357, 0.4353]**,
      cluster ICC **0.0000**.
- [ ] Show it is **entirely within-cluster**: between-cluster ICC of the per-site effect is
      **0.0000**, cluster means +0.015 / +0.028 / +0.063 / −0.040 against a within-cluster sd of
      **0.5135**. This is what makes soil the candidate and rules out climate.
- [ ] Show what the panel explains today: elevation r² **0.017**, latitude **0.020**, longitude
      **0.034** — i.e. ~97% unexplained.
- [ ] State the E06 hypothesis and why a *static* covariate is required: the field effect is
      time-invariant by construction, so a dynamic series like SMAP cannot explain it.

### Task 4 — `notebooks/04-leakage-and-validation.ipynb`
Make the three leaks visible, since they are the methodological contribution.

- [ ] **Peer reference**: coverage per held-out cluster by key — `cluster_month` `[0,0,0,0]` versus
      `geo_month` `[1.0,1.0,1.0,0.472]`. An honest cluster-keyed fit leaves an unseen region with no
      reference at all.
- [ ] **The temporal cut**: cutting train on `obs_date` puts 121 of 2,931 rows in training whose
      outcomes postdate the first test prediction; cutting on `label_date` gives 2,810 with zero.
- [ ] **The band trade-off**: donor rows per held-out cluster at 5°/500 m `[0,282,1098,0]`,
      10°/1000 m `[36,1353,1098,647]`, 20°/1000 m `[2175,2799,1792,3139]`.
- [ ] Close with the multiple-comparisons position: spatial 165 tested / 8 significant-positive /
      ~4.1 expected (1.9×); temporal 65 / 13 / ~1.6 (8.0×).

**Every code cell in every notebook must be executed headless before committing**, and the report
must say so. A notebook whose cells error is worse than no notebook.

## Not in scope
- No behaviour changes, no new experiments, no re-fetching. E06 is blocked separately: the
  SoilGrids API timed out after 60 s.
