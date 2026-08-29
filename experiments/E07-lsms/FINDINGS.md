# E07 — exposure from GHS-Panel Wave 5: what it settled, and the assumption it rests on

Data: Nigeria GHS-Panel Wave 5 (2023–24), `data/lsms/` (git-ignored; the data-use agreement permits
analysis, not redistribution). 5,067 households, 9,232 plots, ~400 EAs, 108 CSVs.

## The stated assumption, and how to falsify it later

**GHS-Panel households are a national survey sample, not this product's registered farmers.** They
are used here as a *stand-in* to establish the SHAPE of the exposure problem — how much it varies,
and how much of the ranking it drives — not to claim any particular farmer's exposure. Transfer to
real users is an assumption, and it is recorded here so it can be tested rather than forgotten.

**The falsification test, runnable the moment real users exist.** `FarmerPayload`
(`serving/schemas/request.py`) already carries `farm_size` and `yield_value`, so no new collection is
needed. Compare the registered-farmer distribution against the survey's:

| quantity | GHS-Panel W5 | falsified if registered farmers show |
| --- | --- | --- |
| farm size, median | 0.84 ha | a median above ~2 ha (a different farming population) |
| plot area IQR ratio | 7.7x | below ~3x (exposure would stop dominating the ranking) |
| exposure share of var(log expected_loss) | 76.9% | below ~50% |
| yield, median (sole-cropped) | 1.25 t/ha | outside roughly 0.5–2.5 t/ha |

If the first two hold, the conclusions below transfer. If exposure's IQR ratio collapses, Rule 3 in
`docs/CONCEPTS.md` needs revisiting for this product specifically.

## What was identified, and on what evidence

Nothing here trusts a column name — the CSV export carries no variable labels.

| target | source | confidence | evidence |
| --- | --- | --- | --- |
| `area_ha` | `sect11a1.SR_hect` | **CONFIRMED** | `SR_meter/SR_hect == 10000.0` for 100.00% of 9,231 plots — the exact ha→m² constant |
| `harvest_kg` | `secta3i.sa3iq9a × sa3iq9_conv` | **CONFIRMED** | units are local and labelled (`150. BASIN`, `211. TUBER`), which is why a conversion column exists |
| `price_per_t` | `secta3i.sa3iq10 / harvest_kg` | **CONFIRMED** | 267 USD/t, inside this repo's own `FARMGATE_PRICE_USD_PER_T` range (120–420) set before the survey was opened |
| `yield_t_ha` | `harvest_kg / area_ha` | **CROP-DEPENDENT** | vs FAO on sole-cropped plots: rice 0.87x, sorghum 0.84x agree; maize 0.44x, cassava 0.21x do not |
| `crop_diversity` | count of `cropcode` per household | **CONFIRMED** | structural, needs no label |
| the other six coping factors | — | **NOT identified** | bare question codes; guessing would silently produce a wrong vulnerability term |

Two biases carried rather than absorbed: GPS-measured area runs at **0.88×** self-reported (farmers
over-state plots ~14%, spearman +0.549, n=5,137), and **41.3%** of plots are intercropped, so every
per-crop yield uses the sole-cropped denominator — the naive version understates every crop sharing
a plot.

## Finding 1 — exposure dominates the ranking, more than the illustrative grid implied

Full table in `docs/CONCEPTS.md`, Rule 3. Holding panel hazard fixed and swapping only the exposure
distribution: exposure's share of `var(log expected_loss)` goes from **34.0%** (illustrative grid) to
**76.9%** (real). Top-100 queue reconstructed from one term: **64/100 by exposure, 5/100 by hazard**;
`spearman(expected_loss, hazard)` inside the top 100 is **−0.031**.

`yield` alone is **42.4%** — nearly double the entire hazard model's 22.5% — and it is the least
reliable quantity in the chain. Only **9.0%** of households have all yields from trusted crops.

## Finding 2 — the serving yield fallback is crop-blind, and biased for the dominant cereals

`serving/pipeline.py:_expected_yield` returns `1.5 + (ndvi − 0.4) × 3.5`, clipped to [0.5, 5.0], for
any farmer who reports no yield. Evaluated over the panel's real NDVI and compared to GHS-Panel:

| | fallback | real (sole-cropped) |
| --- | --- | --- |
| p25 | 0.71 | 0.50 |
| median | 1.17 | 1.25 |
| p75 | 1.95 | 3.33 |
| p95 | 2.90 | 17.88 |
| mean | 1.38 | 6.25 |

**The centre is well calibrated (0.93x at the median) and the spread is not.** Per crop:

| crop | real median | fallback | ratio |
| --- | --- | --- | --- |
| millet | 0.63 | 1.17 | **1.84x too high** |
| maize | 0.80 | 1.17 | **1.46x too high** |
| sorghum | 0.84 | 1.17 | **1.38x too high** |
| rice | 1.75 | 1.17 | **0.67x too low** |

The root cause is that the mapping is **crop-blind**: one NDVI→yield line cannot serve maize grain
(0.80 t/ha) and cassava tubers (2.50 t/ha fresh weight) at once. The [0.5, 5.0] clip also makes the
high-yield tail unreachable — 0.0% of fallback values hit the ceiling while real p95 is 17.88 t/ha
(root crops, fresh weight — not comparable to cereal grain, which is why the per-crop rows above are
the fair comparison and the pooled distribution is not).

This matters in proportion to Finding 1: yield drives 42.4% of ranking variance, so a 1.4–1.8x bias
on the dominant cereals is not a rounding error in the queue.

**Not changed here.** `_expected_yield` is live serving code and the fix is a product decision:
per-crop coefficients, a wider clip, or refusing to impute at all and marking exposure unknown. The
docstring already calls it "explicitly a placeholder"; this quantifies by how much.

## Finding 3 — vulnerability, built for the first time

Five of seven `COPING_FACTORS` now come from data (`vulnerability.py`, 4,771 households):

| signal | variable | grain | prevalence |
| --- | --- | --- | --- |
| `has_irrigation` | `s11b1q56` | plot → any | **2.3%** |
| `used_fertilizer` | `s11c2q5` | plot → any | 21.1% |
| `has_extension_access` | `sa5bq1` | hh × source → any | 16.5% |
| `asset_score` | `sa4q1` | hh × item → share owned | median 0.118 |
| `market_access` | `c5q3`, `infra_code == '222. Market'` | **community** | median 0.889 (2.0 km) |
| `crop_diversity` | count of `cropcode` | hh | median 1 |
| `received_credit` | — | — | **not built** |

2.3% irrigation is the number to notice: irrigation is the coping factor that most directly offsets
the drought hazard, and essentially nobody has it.

Median vulnerability score is **0.798** (p10 0.625, p90 0.930) — far higher and far tighter than
notebook 05's uniform 0–1 sweep assumed.

**Three defects caught while building it**, each of which would have failed silently:

* **1 = YES, 2 = NO.** Every yes/no in this survey. `astype(bool)` on a column of 1s and 2s makes
  every *NO* truthy and **inverts the coping signal for the majority**. `_yes()` is the only place a
  1/2 answer is converted anywhere in the module.
* **`sectc5` is one row per (EA, infrastructure)** across 24 types — nursery school, mosque,
  community centre, market. Aggregating distance across all of them answers "how far is the nearest
  *anything*", which is always small and unrelated to market access. It now filters on the decoded
  label so a renumbered code fails loudly rather than silently selecting a school.
* **Merging a bare Series on `right_index=True` attached nothing** — the column appeared as all-NaN
  with no error, which is how the first build reported `market_access n=0`.

**Bias carried:** `assess_vulnerability` reads a missing signal as `0.0`, i.e. "no credit", so the
absent `received_credit` inflates every household's vulnerability uniformly.

## Finding 4 — exposure and vulnerability are NOT independent, and the queue deprioritises the vulnerable

Every prior analysis in this repository sampled exposure and vulnerability **independently** —
including Rule 3's variance decomposition. With both terms now real for the same 3,011 households,
that assumption is measurable, and it is false:

| | spearman |
| --- | --- |
| exposure ~ vulnerability | **−0.410** |
| farm size ~ vulnerability | −0.378 |
| yield ~ vulnerability | +0.076 |

Poorer, less-equipped households farm **smaller plots**. Composing through the real `assess_risk`:

```
spearman(expected_loss, vulnerability)  = -0.289
vulnerability, top-200 by expected_loss :  0.766
vulnerability, everyone else            :  0.799
```

**Ranking by `expected_loss` systematically deprioritises the most vulnerable households** — not by
intent, by arithmetic. `expected_loss = exposure x loss_rate`, and the households with the least
coping capacity have the least value at risk, so they sort down the queue.

This is structural, not a bug. `(0.5 + 0.5v)` was designed so vulnerability *raises* the loss rate,
but it can only move the score within a factor of two, while exposure varies **7.7x across the
interquartile range**. Vulnerability's intended correction is overwhelmed by exposure's spread, then
pushed the wrong way by the negative correlation between them.

It also lands on the fairness machinery already here. `PROTECTED_ATTRIBUTES` is enforced so protected
characteristics cannot *drive* vulnerability — but nothing inspects whether the composed ranking
disadvantages poor households through farm size, and it does.

**What is real and what is not.** The −0.410 correlation is measured from survey data alone and does
not depend on any hazard. The expected-loss levels and the severity mix (1,886 of 3,011 CRITICAL) are
**composed** — these households have no coordinates, so hazard was drawn from the panel's
distribution. The 63% CRITICAL rate hints the severity thresholds are miscalibrated for this
population, but that claim rests on a drawn hazard and is not made here.

## What E07 cannot do

`assess_vulnerability` remains unvalidated: six of seven coping factors are unidentified without the
variable labels. The Stata (`.dta`) export from the same catalogue page carries embedded labels and
would resolve this in one download.

E08 remains blocked for a separate reason: this release ships **no coordinates**. All 108 CSVs were
scanned; the only `lon`-matching column is `wt_longpanel_wave5`, a survey weight. See the joining
spec §4b for why E09's results limit what E08 could show even with them.
