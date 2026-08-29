# E09 — What coordinate privacy costs: a displacement placebo on known-location sites

**Status:** queued. **Gates E08** — run before acquiring LSMS-ISA geovariables for spatial work.
**Spec:** `docs/superpowers/specs/2026-08-29-dataset-joining-design.md` §4a.

## The question

E07/E08 propose joining LSMS-ISA to satellite features. But public LSMS-ISA coordinates are
**displaced** enumeration-area centroids, not household locations: uniform random direction, distance
uniform in [0, 5 km] rural (10 km for 1%), 2 km urban, constrained to the admin unit.

Sentinel features here use a ±500 m AOI half-width (`data/sentinel.py:204`). The displacement is ten
times that half-width and ~64× the AOI in area.

**So: how much of the signal we can measure survives that displacement?**

This is answerable *now*, on data already in hand, without registering for a single LSMS file.

## Why this must run first

If the known signal does not survive simulated displacement, E08 is not a hard experiment — it is an
impossible one, and E09 says so for the cost of a rerun rather than weeks of acquisition and
harmonisation work.

This is a live possibility, not a formality. The field effect is 34.5% [0.2357, 0.4353] and
**entirely within-cluster** (between-cluster ICC 0.0000). A within-cluster effect is exactly the kind
that a few-kilometre displacement destroys: the displaced point stays in the same cluster, so it
inherits the cluster mean and loses the field-specific deviation — which is where all the signal is.

## Method

1. **Sample displacements.** For each of the 122 sites, draw `θ ~ U(0, 2π)` and `d ~ U(0, 5 km)`,
   giving `(lat', lon')`. Draw **M ≥ 30** independent displacement realisations per site — a single
   draw measures one arbitrary outcome, not the distribution. Seed and record the seed.
2. **Rebuild features at the displaced coordinates**, through the existing pipeline, unchanged. Both
   arms must use identical code; only the coordinates differ.
3. **Keep the labels attached to the true site.** The label is a property of the real field. This is
   the whole point: it reproduces the LSMS situation, where the outcome is real and the coordinates
   are not.
4. **Re-run the E02 matrix** on true and displaced features under identical splits (spatially blocked
   and purged forward chaining, 30-day embargo).
5. **Report the retention ratio** per metric, with a site-level bootstrap CI across the M draws.

## What to measure

- **Field-effect retention** — re-run the E01 method-of-moments ICC on displaced features. Expected
  to fall sharply; the size of the fall is the headline number.
- **Temporal-skill retention** — the 13/65 significant-positive result is the benchmark. How many
  survive?
- **Per-upstream retention**, separately, because §4a predicts these differ by an order of magnitude:
  ERA5 is sub-pixel at 5 km and should be **nearly unaffected**; Sentinel-2/-1 should collapse.
  Confirming that asymmetry is itself the useful result — it tells E08 which features to keep.

## Cost, stated plainly

The Sentinel re-fetch is the expensive part: 122 sites × M draws against a rate-limited API, and the
SoilGrids fetch is already contending for the same budget. Mitigations:

- Run the **weather-only arm first**. It is cheap, and §4a predicts it is nearly unaffected — so if
  weather retention is *not* ~1.0, the harness is wrong and no Sentinel budget should be spent yet.
- Start at **M = 30** and widen only if the CI is too loose to decide.
- Reuse `.cache/` keyed on rounded displaced coordinates so repeated draws that land in the same
  Sentinel AOI hit cache rather than the API.

## Acceptance

- [ ] Both arms run identical code paths; a diff of the two configs shows only coordinates.
- [ ] Labels remain bound to the true site in both arms — assert it, do not assume it.
- [ ] The seed and all M draws are recorded in the provenance block so the result is reproducible.
- [ ] Weather-only retention is ~1.0, or the harness is rejected before any Sentinel spend.
- [ ] Retention is reported with an interval, not a point estimate.

## Interpretation, fixed in advance

To avoid reading whatever comes back as confirmation:

- **Weather retained, Sentinel destroyed** (the predicted outcome): E08 proceeds, but restricted to
  weather-derived features at EA scale. The canopy half of the hazard model does not transfer.
- **Both retained:** surprising; suspect the harness before believing it — most likely the
  displacement was not actually applied, or features are dominated by something coarse.
- **Both destroyed:** E08 as specified cannot work. The EA-scale route is closed and exposure and
  vulnerability must be validated another way (backend farmer payload, or a partner dataset with
  true coordinates).

## Independent value

Beyond gating E08, this is a publishable result in its own right: a quantified statement of what
coordinate privacy costs remote-sensing agronomy, measured on a panel where the ground truth *is*
known — which is precisely what the survey-displacement literature cannot do with public data alone.
Companion reading: *Privacy Protection, Measurement Error, and the Integration of Remote Sensing and
Socioeconomic Survey Data* (arXiv:2202.05220).
