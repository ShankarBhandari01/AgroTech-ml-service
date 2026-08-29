# E11 — the composed risk equation, scored against a realised outcome for the first time

**Run:** `python experiments/E11-composed-validation/validate_composition.py` → `e11_result.json`
**Provenance:** git `d1e6e2f` + uncommitted working tree, seed `20260829`, 2,875 households, 355 EAs.
**Inputs:** `experiments/E07-lsms/composed_risk.parquet` ⋈ `experiments/E10-croploss/dataset.parquet`
on `hhid`. **Both are git-ignored** — they are record-level LSMS extracts, and the World Bank
data-use agreement permits analysis, not redistribution. Regenerate them by re-running
`experiments/E07-lsms/{exposure,vulnerability}.py` and `experiments/E10-croploss/build_dataset.py`
against a re-downloaded copy of the microdata; every number below is then reproducible from
`e11_result.json`'s recorded seed and git SHA.

Every term in `domain.risk` has been validated separately. The composition never had been. E10
produced a real agronomic outcome on the same survey households E07 composed risk for, so the test
was a join and had been sitting on disk unrun.

## What this can and cannot validate

E07's `hazard` is **drawn** from the panel's distribution — Wave 5 ships no coordinates. So this
scores the **exposure × vulnerability** composition, which is 77.4% of `var(log expected_loss)` and
the part actually under dispute. It cannot validate the hazard model; nothing here can, until
outcome capture runs.

That drawn hazard also gives a free **negative control**: `hazard` alone must land at chance.
It does — AUC **0.496**, CI [0.473, 0.519]. The harness is sound.

**Unit is the household**, because a visit goes to a farmer. Bootstrap resamples **EAs**, never
households — the same rule this repo applies to sites. Prevalence of `any_loss` is 0.2894.

## Result

| ranking | AUC | 95% CI (EA bootstrap) | PR-AUC | P@50 | P@200 | vuln@50 |
| --- | --- | --- | --- | --- | --- | --- |
| **`expected_loss`** (production queue) | **0.537** | [0.505, 0.567] | 0.316 | 0.300 | 0.315 | 0.728 |
| `risk_score` | 0.466 | [0.443, 0.489] | 0.268 | *tied* | *tied* | 0.954 |
| `expected_loss`, log-compressed exposure | 0.512 | [0.487, 0.538] | 0.301 | 0.320 | 0.305 | 0.818 |
| `exposure` alone | 0.564 | [0.526, 0.597] | 0.334 | 0.360 | 0.289 | 0.721 |
| `vulnerability` alone | 0.430 | [0.395, 0.464] | 0.256 | *tied* | *tied* | 0.958 |
| `hazard` alone — **negative control** | 0.496 | [0.473, 0.519] | 0.289 | *tied* | *tied* | 0.818 |
| **E10 fitted model — ceiling** | **0.652** | [0.621, 0.683] | 0.429 | 0.560 | 0.550 | 0.665 |

Base rate 0.289. Population mean vulnerability 0.780. *tied* = `precision_at_k` returned prevalence
because the top-50 was an arbitrary tie-break (see below).

Paired EA bootstrap — both arms on the same resample, which is the only way to compare two rankings
whose intervals share their noise:

| contrast | ΔAUC | 95% CI | P(Δ>0) |
| --- | --- | --- | --- |
| `expected_loss` − `exposure` alone | **−0.0269** | [−0.0438, −0.0100] | 0.001 |
| `expected_loss` − `hazard` (control) | +0.0403 | [+0.0137, +0.0653] | 0.999 |
| `risk_score` − `expected_loss` | **−0.0704** | [−0.1048, −0.0349] | 0.000 |
| E10 ceiling − `expected_loss` | **+0.1149** | [+0.0840, +0.1442] | 1.000 |

## Four findings

**1. Composing makes the ranking worse than its own exposure term.** `expected_loss` scores
**below** `exposure` alone, and the paired interval excludes zero (−0.0269 [−0.0438, −0.0100]).
Multiplying exposure by a drawn hazard and a real vulnerability *destroys* ordering information
rather than adding any. The composition is not merely exposure-dominated — it is exposure, degraded.

**2. The queue does not survive the measurement error already documented in this repo.** Under
plot-area error at the recorded rank agreement (spearman +0.549, n=5,137), only **10% of the top 50
survives**; 16% under area error plus the FAO yield correction. The systematic yield correction
alone is benign (84% retained) — it is the *random* area error that destroys the order.

| scenario | top-50 retained | top-100 | top-200 | spearman vs base |
| --- | --- | --- | --- | --- |
| A — yields corrected toward FAO by crop mix | 0.84 | 0.76 | 0.855 | 0.976 |
| B — plot-area measurement error | **0.10** | 0.15 | 0.25 | 0.657 |
| C — both | 0.16 | 0.21 | 0.30 | 0.629 |

**This is the robustness answer.** Ranking 1..50 asserts a precision the inputs cannot support.
Ship "these ~200 farmers, unordered" or fix area measurement; do not ship a numbered queue.

**3. `risk_score` cannot order a 50-farmer queue at all.** It is `round(score, 1)` — **557 distinct
values across 2,875 households**, tied at k=50, so the top 50 is an artifact of row order.
`model-design.md` §9.3a learned exactly this lesson and stored `hazard.combined` at 6dp; it was
never applied to `risk_score`, which is the field a queue would actually sort on. One-line fix.

**4. The equity trade-off, now measured rather than argued.** §9.4 left the ranking-target choice
open. Here it is, against a real outcome:

| queue key | AUC | mean vulnerability of top 50 |
| --- | --- | --- |
| `expected_loss` | 0.537 | 0.728 (**below** the 0.780 population mean) |
| `expected_loss`, log-compressed exposure | 0.512 | 0.818 |
| `risk_score` | 0.466 | 0.954 |

`risk_score` reverses the equity inversion completely — and ranks **worse than chance** against
realised loss. The log-compressed middle option costs 0.025 AUC and buys +0.09 vulnerability. That
is the trade the product owes a decision on; it is no longer hypothetical.

## The confound, checked rather than assumed

`any_loss` rises mechanically with plot count, and exposure correlates with plot count
(spearman +0.228). Most of exposure's edge **is** the count:

- plot count alone: AUC 0.553, against exposure's 0.564.
- within fixed plot-count strata exposure keeps only a small edge — 0.521 / 0.534 / 0.549 at 1 / 2 / 3 plots.
- on `share_loss`, a rate that count cannot inflate: exposure **+0.117**, `expected_loss` +0.069.

So larger farms do lose crop slightly more often — exposure is not purely a magnitude term, and
that partially defends its presence. But +0.117 spearman is not a mandate to own 77% of the
ranking.

## Decision curve (net benefit; ceiling is the event rate, 0.2894)

| threshold | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 |
| --- | --- | --- | --- | --- | --- |
| `expected_loss` | 0.2104 | 0.1117 | 0.0093 | −0.0002 | 0.0000 |
| `exposure` alone | 0.2104 | 0.1099 | 0.0237 | −0.0009 | −0.0003 |
| **E10 fitted model** | 0.2104 | **0.1196** | **0.0588** | **0.0212** | **0.0045** |
| visit everyone | 0.2104 | 0.1117 | −0.0152 | −0.1843 | −0.4212 |
| visit nobody | 0 | 0 | 0 | 0 | 0 |

At t ≤ 0.20 the composed equation is **identical to visiting everyone** — it flags the whole
population and adds nothing. At t = 0.30 it clears "visit nobody" by 0.0093, i.e. barely. Only the
fitted model clears both trivial strategies across the range.

Probabilities come from Platt scaling fitted **out of fold by EA** — the composed score is not a
probability of this label, and fitting the scaling in-sample would let the curve report its own fit.
That bridge is an assumption, stated here as `evaluate.prob_event` states its normal-CDF equivalent.

## What follows

1. **Do not ship a numbered queue.** Finding 2 is decisive and independent of everything else.
2. **Fix `risk_score`'s rounding** (Finding 3). One line, and §9.3a already set the precedent.
3. **The composition needs a reason to exist.** Finding 1 says multiplying the terms costs
   ordering. Either the multiplicative form is wrong for ranking, or hazard must be real before the
   product can be judged — and only outcome capture settles that.
4. **E10's fitted model beats the equation by +0.115 AUC** on the one outcome either has been scored
   against. That is the argument for supervised outcomes over more composition.

## Limits, stated

- **Hazard is drawn.** A real hazard could change every number here. This bounds the composition, not the model.
- **GHS-Panel is a national survey sample, not registered farmers.** E07's falsification test applies unchanged.
- **The label is `sa3iq6`** — was area harvested less than area planted. A partial-loss indicator, not yield.
- **The ceiling is not E10's headline 0.727.** That was per crop-plot; this is per household after
  max-aggregation over a subset, a different and harder unit. The two are not comparable and only
  the within-this-file contrast should be read.
- One test, one pre-registered question, no matrix swept — deliberately, given 8/165 spatial cells
  cleared zero against ~4.1 expected elsewhere in this project.
