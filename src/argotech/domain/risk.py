"""Risk composition: Risk = Hazard x Exposure x Vulnerability.

The IPCC AR5/AR6 risk framing, which is the standard in agricultural early warning and the reason
this module exists. The current production model collapses hazard signals (rainfall, NDVI, humidity)
and vulnerability signals (assets, credit, extension access, household composition) into one opaque
3-class classifier. That has three consequences worth stating plainly:

1. It cannot be validated. Hazard is verifiable against weather and satellite records within days;
   vulnerability is only verifiable against outcomes months later. Fused into one label, neither
   can be measured.
2. It cannot be explained. "High risk" gives an extension officer no way to know whether to bring
   fungicide or a credit application.
3. It cannot be audited for fairness. A protected attribute inside the fused model silently moves
   the priority ranking that allocates real resources.

Splitting the three terms fixes all three. Each is separately computable, separately testable, and
composed by an arithmetic rule anyone can check.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from argotech.domain import agronomy

# Attributes that must never enter the vulnerability score. They are retained upstream for
# *fairness auditing* — measuring whether the ranking disadvantages these groups — but a model that
# allocates extension visits and credit must not use them as inputs.
PROTECTED_ATTRIBUTES = frozenset({"head_gender", "household_max_education", "religion", "ethnicity"})


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# --- Hazard component maps: smooth, strictly monotone, never exactly 0 or 1 ---------------------
#
# Every hazard component is a logistic, matching `vegetation_hazard_from_anomaly` below, which was
# always built this way. A clamped linear ramp -- `_clamp01((x - a) / (b - a))` -- pins at exactly
# 1.0 past its upper threshold, and `noisy_or` has an ABSORBING STATE there: once any component is
# 1.0, `prod(1 - h_i)` is 0 and every other component becomes irrelevant. Measured on the committed
# panel, the old drought ramp was exactly 1.0 for 59.5% of rows, leaving ONE distinct combined-hazard
# value across the entire top half of the queue (docs/model-design.md 9.3). A ranking cannot see past
# a constant.
#
# The map is deliberately ASYMMETRIC. A logistic centred mid-ramp would return ~0.1 at a component's
# "no hazard" threshold, and four such components would floor combined hazard near 0.34 for a field
# under no stress at all. So the edges are pinned at RAMP_LO / RAMP_HI instead: essentially zero
# where the old ramp read zero, high but with headroom where it read one. That headroom is the
# second point -- with severe drought at 0.90 rather than 1.0, a field facing severe drought AND
# severe disease now scores above one facing drought alone, which is exactly what noisy-OR is for
# and what saturation destroyed.
#
# `u` keeps its old meaning in every caller: 0 at the "no hazard" threshold, 1 at "severe". Every
# published threshold below is therefore unchanged; only the shape between and beyond them is.
RAMP_LO = 0.02      # component value at u = 0 (the old ramp's zero)
RAMP_HI = 0.90      # component value at u = 1 (the old ramp's one)
_L0 = math.log(RAMP_LO / (1.0 - RAMP_LO))
_L1 = math.log(RAMP_HI / (1.0 - RAMP_HI))
RAMP_SOFTNESS = 1.0 / (_L1 - _L0)
RAMP_CENTRE = -_L0 * RAMP_SOFTNESS

# Floating point can reach 1.0 even where the mathematics cannot: for a sufficiently extreme input
# `1/(1 + exp(-x))` rounds to exactly 1.0 in float64. The absorbing state is a property of the
# VALUE, not of the formula, so the guard is applied to the result rather than assumed away.
_EPS = 1e-9


def _logistic(x: float) -> float:
    """Numerically stable 1 / (1 + exp(-x)).

    The naive form raises OverflowError once x < -709 in float64 -- not a theoretical concern: the
    pre-existing `vegetation_hazard_from_anomaly` overflowed on an extreme peer anomaly, which a
    degenerate cohort standard deviation can produce. Splitting on the sign keeps both tails finite.
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _no_saturate(h: float) -> float:
    """Keep a hazard strictly inside (0, 1) so noisy-OR can never be absorbed."""
    return min(max(h, _EPS), 1.0 - _EPS)


def _soft_ramp(u: float) -> float:
    """Smooth, strictly monotone replacement for `_clamp01(u)` in hazard maps.

    `u` is the caller's existing normalised stress: 0 at the no-hazard threshold, 1 at severe.
    Returns RAMP_LO at u=0 and RAMP_HI at u=1, approaching 0 and 1 asymptotically without arriving.
    """
    if math.isnan(u):
        return _EPS
    return _no_saturate(_logistic((u - RAMP_CENTRE) / RAMP_SOFTNESS))


# Mirrors `training.dataset.SEVERE_Z`, the anomaly at which the label calls a field severely
# stressed. Restated rather than imported: serving must not depend on the training package. A test
# pins the two together, so a change to one fails rather than drifting.
SEVERE_ANOMALY_Z = -1.0

# Logistic width. A hard ramp between the label's two cut points assigns exactly 0.0 to every field
# above -0.35 — 68% of them — and ties are invisible to a ranking: measured leave-one-cluster-out
# over all six clusters it costs 23% of the rank correlation (rho 0.351 -> 0.271). A smooth
# strictly-monotone squash keeps the whole ordering inside the same [0, 1] the hazard API needs.
ANOMALY_SOFTNESS = 0.5


def vegetation_hazard_from_anomaly(peer_anomaly_z: float | None) -> float:
    """Map a peer-standardised NDVI anomaly onto a vegetation hazard in [0, 1].

        h = 1 / (1 + exp((z - SEVERE_ANOMALY_Z) / ANOMALY_SOFTNESS))

    Centred on the severe cut, so h = 0.5 exactly at the threshold the label calls severe, rising
    toward 1 as the field falls further behind its neighbours and toward 0 as it pulls ahead.

    **This is the shared mapping for both hazard sources**, which is why the parameter is named for
    the quantity and not for its origin. `serving/pipeline.py` selects the origin from
    `VEGETATION_HAZARD_SOURCE`:

    * `model` (the default) passes `forward_z` — the anomaly a `HistGradientBoostingRegressor`
      *predicts* for 30 days ahead.
    * `persistence` passes `ndvi_z_peer` — the field's own anomaly *observed today*, carried forward.

    Identical centre and softness either way, so the two are directly comparable as rankings. From
    `artifacts/metrics.json` (seed 42, leave-one-cluster-out, six folds): the model ranks at
    Spearman rho 0.377 and P@25 0.760, persistence at 0.368 / 0.467, site climatology at
    0.363 / 0.680. The model's case is P@25 — ahead of persistence in all six folds and of
    climatology in four — not rho, where all three are within 0.014 of each other.

    The magnitudes are *not* comparable, and no rank metric can see it: a regressor predicts a
    conditional mean, so its output is compressed (predicted std 0.508 against an observed
    `forward_z` std of 1.165), and the model path therefore returns systematically smaller hazards
    than persistence would for the same field.

    An absent anomaly yields 0.0 — no canopy evidence means no canopy hazard, and the physical
    hazards carry the assessment, exactly as they do when no satellite scene is available at all.
    """
    if peer_anomaly_z is None or math.isnan(peer_anomaly_z):
        return 0.0
    return _no_saturate(_logistic(-(peer_anomaly_z - SEVERE_ANOMALY_Z) / ANOMALY_SOFTNESS))


def noisy_or(probabilities: list[float]) -> float:
    """Combine independent hazards: 1 - prod(1 - p_i).

    Two independent 0.4 hazards give 0.64, not 0.4 (max) and not 0.4 (mean). Compounding is the
    behaviour you want — a field facing both drought and blight is worse off than one facing either.
    """
    out = 1.0
    for p in probabilities:
        out *= (1.0 - _clamp01(p))
    return 1.0 - out


# ---------------------------------------------------------------------------------------------
# Hazard — probability/intensity of an adverse agro-climatic event. Derived from physics and
# epidemiology, so it is available for every field immediately, with no training labels.
# ---------------------------------------------------------------------------------------------

@dataclass
class Hazard:
    drought: float
    disease: float
    heat: float
    vegetation: float
    pest: float
    excess_rain: float
    weeds: float
    combined: float
    dominant: str

    def as_dict(self) -> dict:
        return asdict(self)


def assess_hazard(water_satisfaction: float, dry_spell_days: int,
                  cumulative_dsv: int, heat_days: int,
                  vegetation: float = 0.0,
                  spray_threshold: int = 18,
                  gdd_since_onset: float | None = None,
                  rain_anomaly_mm: float | None = None,
                  weed_pressure: float | None = None) -> Hazard:
    """Map agronomic indicators onto hazard intensities in [0, 1].

    `vegetation` is where the canopy signal plugs in: `vegetation_hazard_from_anomaly` applied to a
    peer anomaly that is either predicted 30 days out or observed today, depending on
    `VEGETATION_HAZARD_SOURCE`. Either way it enters as one more independent hazard rather than as a
    competing overall score. Leave it at 0 to get the pure physics/epidemiology assessment — which
    is also the baseline the canopy component has to beat before it is worth including.

    The mappings are explicit and monotone by construction. They are *priors*, replaced component by
    component as outcome labels accumulate — see `docs/model-design.md`, phase 2.
    """
    # Water stress: satisfaction 1.0 -> no hazard, 0.4 -> severe. A 10+ day dry spell is damaging
    # even when the seasonal total looks adequate, so it enters separately.
    drought = _soft_ramp((1.0 - water_satisfaction - 0.15) / 0.5)
    drought = max(drought, _soft_ramp((dry_spell_days - 7) / 14.0))

    # Disease: the accumulated severity value relative to the published spray threshold.
    disease = _soft_ramp(cumulative_dsv / max(1, spray_threshold))

    # Heat: days above the pollen-viability threshold during flowering. Three such days is severe.
    heat = _soft_ramp(heat_days / 3.0)

    # --- Terms added after E07 measured which shocks Nigerian farmers actually report -------------
    #
    # GHS-Panel Wave 5 (experiments/E07-lsms/, notebooks/08-lsms-eda.ipynb), agronomic shocks by
    # share of households: weeds 15.7%, drought 13.9%, excess rain 10.6%, pests 9.2%, crop disease
    # 6.7%, hail/frost 1.7%. The taxonomy above modelled drought, disease and heat -- and heat maps
    # to the RAREST reported shock, while the three most common after drought had no term at all.
    # Attributed causes of real crop loss (n=1,952, censoring removed) agree on the ordering:
    # water-related 64.5%, pest/disease 24.9%.
    #
    # Each new term contributes ONLY when its upstream is supplied. A hazard component with no data
    # must contribute nothing, never a constant: `noisy_or` compounds, so a fixed value would raise
    # every field's combined hazard without discriminating between any of them -- the same
    # can't-rank failure as the saturation defect, arriving from the opposite direction.

    # Pest: fall armyworm degree-day generations, the dominant maize pest in Sub-Saharan Africa
    # since 2016. `agronomy.fall_armyworm_generations` already existed and was wired into nothing.
    # One completed generation is the conventional scouting trigger; three is heavy pressure.
    if gdd_since_onset is None:
        pest = 0.0
    else:
        generations = agronomy.fall_armyworm_generations(gdd_since_onset)
        pest = _soft_ramp((generations - 1.0) / 2.0)

    # Excess rain: waterlogging and lodging, which `water_satisfaction` cannot express because it is
    # a satisfaction RATIO capped at 1.0 -- surplus and exactly-enough are the same number to it.
    # Measured against the site's own climatology, so it is an anomaly rather than a raw total.
    if rain_anomaly_mm is None:
        excess_rain = 0.0
    else:
        excess_rain = _soft_ramp((rain_anomaly_mm - 50.0) / 100.0)

    # Weeds: the LARGEST agronomic shock reported (15.7%), and there is no upstream for it anywhere
    # in this project -- weed pressure depends on time since last weeding and labour availability,
    # neither of which any feed carries. The parameter exists so the taxonomy is honest about the
    # gap and a caller with survey data can supply it; absent that it contributes nothing at all.
    weeds = 0.0 if weed_pressure is None else _soft_ramp(float(weed_pressure))

    parts = {"drought": drought, "disease": disease, "heat": heat,
             "vegetation": _no_saturate(_clamp01(vegetation)),
             "pest": pest, "excess_rain": excess_rain, "weeds": weeds}
    dominant = max(parts, key=parts.get) if max(parts.values()) > 0.05 else "none"

    # Rounding is applied BEFORE the saturation guard, not after, and that order is load-bearing.
    # Capping each component at 1 - 1e-9 does not keep the combination below 1: noisy_or of four
    # such components is 1 - (1e-9)**4 = 1 - 1e-36, which IS exactly 1.0 in float64. round(x, 3)
    # does the same to a component. The absorbing state is a property of the stored value, so the
    # guard has to be the last thing that touches it.
    #
    # 6dp, not 3: `combined` is the RANKED quantity and rounding is a tie-generator. With the smooth
    # ramps above, the panel's top half spans 0.9928..1.0, so 3dp collapsed 140 distinct values to 8.
    return Hazard(
        drought=_no_saturate(round(drought, 6)),
        disease=_no_saturate(round(disease, 6)),
        heat=_no_saturate(round(heat, 6)),
        vegetation=_no_saturate(round(_clamp01(vegetation), 6)),
        pest=round(pest, 6),
        excess_rain=round(excess_rain, 6),
        weeds=round(weeds, 6),
        combined=_no_saturate(round(noisy_or(list(parts.values())), 6)),
        dominant=dominant,
    )


# ---------------------------------------------------------------------------------------------
# Exposure — the value that is actually at stake. Absolute, in currency, and the term that makes
# the score rankable across farmers rather than only comparable within one.
# ---------------------------------------------------------------------------------------------

@dataclass
class Exposure:
    area_ha: float
    expected_yield_t_ha: float
    price_per_t: float
    value_at_risk: float

    def as_dict(self) -> dict:
        return asdict(self)


def assess_exposure(area_ha: float, expected_yield_t_ha: float, price_per_t: float) -> Exposure:
    return Exposure(
        area_ha=area_ha,
        expected_yield_t_ha=expected_yield_t_ha,
        price_per_t=price_per_t,
        value_at_risk=round(max(0.0, area_ha * expected_yield_t_ha * price_per_t), 2),
    )


# ---------------------------------------------------------------------------------------------
# Vulnerability — inability to absorb the shock. Bounded in [0, 1], built only from attributes a
# farmer or a programme can actually change.
# ---------------------------------------------------------------------------------------------

# Each entry: (coping capacity contribution when present/at max, weight).
COPING_FACTORS = {
    "has_irrigation": 1.0,
    "has_extension_access": 0.8,
    "received_credit": 0.7,
    "used_fertilizer": 0.5,
    "crop_diversity": 0.8,   # normalised 0-1 upstream
    "asset_score": 1.0,      # normalised 0-1 upstream
    "market_access": 0.6,    # normalised 0-1 upstream
}

# What a *gap* in each factor means, in words an extension officer can act on.
#
# Kept beside COPING_FACTORS so adding a factor without wording is an obvious omission rather than a
# silent fallback. The previous fallback derived the phrase mechanically from the field name, which
# inverted the meaning: a missing `has_irrigation` rendered as "Limited coping capacity: has
# irrigation", i.e. it read as though *having* irrigation were the problem.
COPING_GAP_LABELS = {
    "has_irrigation": "No irrigation available",
    "has_extension_access": "No contact with an extension officer",
    "received_credit": "No access to credit",
    "used_fertilizer": "No fertiliser applied",
    "crop_diversity": "Little crop diversification",
    "asset_score": "Few productive assets",
    "market_access": "Poor market access",
}


def describe_gap(factor: str) -> str:
    """Human wording for a coping-capacity gap. Falls back to naming the factor as *missing*.

    The fallback still cannot invert the meaning, which is the property that matters: an unlabelled
    factor reads as "Missing: <factor>", never as though the farmer had it.
    """
    return COPING_GAP_LABELS.get(factor, f"Missing: {factor.replace('_', ' ')}")


@dataclass
class Vulnerability:
    score: float
    coping_capacity: float
    gaps: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def assess_vulnerability(signals: dict[str, float]) -> Vulnerability:
    """Weighted coping capacity, inverted. `signals` values are 0-1 (booleans as 0/1).

    Raises on a protected attribute rather than silently ignoring it: a fairness constraint that is
    only a comment gets violated the first time someone adds a column.
    """
    leaked = PROTECTED_ATTRIBUTES & signals.keys()
    if leaked:
        raise ValueError(f"protected attributes must not drive vulnerability: {sorted(leaked)}")

    weighted = 0.0
    total_weight = 0.0
    gaps = []
    for name, weight in COPING_FACTORS.items():
        value = _clamp01(float(signals.get(name, 0.0)))
        weighted += weight * value
        total_weight += weight
        if value < 0.34:
            gaps.append(name)

    capacity = weighted / total_weight if total_weight else 0.0
    return Vulnerability(
        score=round(1.0 - capacity, 3),
        coping_capacity=round(capacity, 3),
        gaps=gaps,
    )


# ---------------------------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------------------------

@dataclass
class RiskAssessment:
    risk_score: float            # 0-100, the *rate* of expected loss — comparable across farms
    expected_loss: float         # currency — what to rank a triage queue by
    severity: str
    hazard: Hazard
    exposure: Exposure
    vulnerability: Vulnerability

    def as_dict(self) -> dict:
        return {
            "risk_score": self.risk_score,
            "expected_loss": self.expected_loss,
            "severity": self.severity,
            "hazard": self.hazard.as_dict(),
            "exposure": self.exposure.as_dict(),
            "vulnerability": self.vulnerability.as_dict(),
        }


# Fraction of the crop lost given the hazard fully materialises on a maximally vulnerable farm.
# Calibrate against realised harvest outcomes once they exist; until then it is a stated constant,
# not a number pretending to be learned.
MAX_LOSS_FRACTION = 0.6

# Severity bands on `risk_score`, as `(lower_bound, name)`, highest first.
#
# THESE ARE UNCALIBRATED, AND THE PROBLEM IS NOT THAT THEY ARE THE WRONG NUMBERS.
#
# Measured on the committed panel (4,596 rows, real hazard, vulnerability held at E07's median
# 0.798), `risk_score` is BIMODAL, not spread: percentiles are p10 3.9, p25 7.6, **p50 89.2**,
# p75 89.8, p90 89.8. All three cut points below sit between the 28.8th and 36.7th percentile --
# three of the four bands are decided inside a region holding roughly 8% of fields. The result is
# 63.3% CRITICAL / 28.8% NORMAL / 4.2% ELEVATED / 3.7% WATCH: a near-binary output wearing four
# labels. E07's independent households reproduce it (62.6% / 29.8% / 4.0% / 3.6%).
#
# The cause is upstream of these numbers. `_soft_ramp` is a logistic, so a field past a component's
# severe threshold asymptotes toward RAMP_HI and one below the no-hazard threshold toward RAMP_LO,
# leaving the middle thinly populated. model-design.md 9.3a fixed the *ties* (137 distinct values in
# the top half, up from 1) and did not change the SHAPE -- which it never claimed to.
#
# So moving these cuts cannot fix it: no absolute threshold discriminates well on a distribution
# with nothing in the middle. The fix is to band by CAPACITY -- quantiles of the population actually
# being served, so "CRITICAL" means "the k fields an officer can reach", not an absolute score. That
# needs the served distribution, which `predictions` already records on every row, so it is a query
# away once there is traffic rather than a modelling problem. Named here so recalibration is one
# place instead of four literals in a branch.
SEVERITY_BANDS = ((65.0, "CRITICAL"), (35.0, "ELEVATED"), (15.0, "WATCH"), (float("-inf"), "NORMAL"))


def assess_risk(hazard: Hazard, exposure: Exposure, vulnerability: Vulnerability) -> RiskAssessment:
    """Compose the three terms into a loss rate and an absolute expected loss.

    Vulnerability is applied as a modulation (0.5 + 0.5v) rather than a raw multiplier so that a
    well-resourced farm facing a severe hazard is still flagged. A farmer with irrigation and credit
    still loses a crop to late blight if nobody tells them to spray.
    """
    loss_rate = hazard.combined * (0.5 + 0.5 * vulnerability.score) * MAX_LOSS_FRACTION
    exact = 100.0 * _clamp01(loss_rate / MAX_LOSS_FRACTION)

    # 4dp, not 1dp, for the same reason `Hazard.combined` is stored at 6dp (model-design.md 9.3a):
    # ROUNDING IS A TIE-GENERATOR, and this is a RANKED field. Measured on E07's 3,011 households,
    # `round(score, 1)` left 566 distinct values and the 50th and 51st farmer TIED -- so the top 50
    # of a triage queue was an artifact of row order, not of risk. At 4dp it is 2,089 distinct and
    # untied; 6dp adds 7 more values and nothing else. `score` is 100x a [0,1] quantity, so 4dp here
    # is the same granularity 9.3a chose for `combined`. E11 measured this; the queue key had never
    # been checked for ties because only `combined` was.
    #
    # A farmer is shown one decimal place. That is a presentation choice and belongs to whatever
    # renders it -- the same argument 9.3a made for `combined`, and the reason storage precision and
    # display precision are not the same decision.
    score = round(exact, 4)

    # Severity reads `exact`, never `score`: a display rounding must not decide a category. On E07's
    # households the two agree on all 3,011 rows, so this changes no observed outcome -- it removes
    # a coupling that could flip a band at a boundary for no reason anyone would be able to explain.
    severity = next(name for cut, name in SEVERITY_BANDS if exact >= cut)

    return RiskAssessment(
        risk_score=score,
        expected_loss=round(exposure.value_at_risk * loss_rate, 2),
        severity=severity,
        hazard=hazard,
        exposure=exposure,
        vulnerability=vulnerability,
    )
