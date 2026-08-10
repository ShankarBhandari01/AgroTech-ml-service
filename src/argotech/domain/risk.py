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

from dataclasses import dataclass, field, asdict


# Attributes that must never enter the vulnerability score. They are retained upstream for
# *fairness auditing* — measuring whether the ranking disadvantages these groups — but a model that
# allocates extension visits and credit must not use them as inputs.
PROTECTED_ATTRIBUTES = frozenset({"head_gender", "household_max_education", "religion", "ethnicity"})


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


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
    combined: float
    dominant: str

    def as_dict(self) -> dict:
        return asdict(self)


def assess_hazard(water_satisfaction: float, dry_spell_days: int,
                  cumulative_dsv: int, heat_days: int,
                  spray_threshold: int = 18) -> Hazard:
    """Map agronomic indicators onto three hazard intensities in [0, 1].

    The mappings are explicit and monotone by construction. They are *priors*, replaced component by
    component as outcome labels accumulate — see `docs/model-design.md`, phase 2.
    """
    # Water stress: satisfaction 1.0 -> no hazard, 0.4 -> severe. A 10+ day dry spell is damaging
    # even when the seasonal total looks adequate, so it enters separately.
    drought = _clamp01((1.0 - water_satisfaction - 0.15) / 0.5)
    drought = max(drought, _clamp01((dry_spell_days - 7) / 14.0))

    # Disease: the accumulated severity value relative to the published spray threshold.
    disease = _clamp01(cumulative_dsv / max(1, spray_threshold))

    # Heat: days above the pollen-viability threshold during flowering. Three such days is severe.
    heat = _clamp01(heat_days / 3.0)

    parts = {"drought": drought, "disease": disease, "heat": heat}
    dominant = max(parts, key=parts.get) if max(parts.values()) > 0.05 else "none"

    return Hazard(
        drought=round(drought, 3),
        disease=round(disease, 3),
        heat=round(heat, 3),
        combined=round(noisy_or(list(parts.values())), 3),
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


def assess_risk(hazard: Hazard, exposure: Exposure, vulnerability: Vulnerability) -> RiskAssessment:
    """Compose the three terms into a loss rate and an absolute expected loss.

    Vulnerability is applied as a modulation (0.5 + 0.5v) rather than a raw multiplier so that a
    well-resourced farm facing a severe hazard is still flagged. A farmer with irrigation and credit
    still loses a crop to late blight if nobody tells them to spray.
    """
    loss_rate = hazard.combined * (0.5 + 0.5 * vulnerability.score) * MAX_LOSS_FRACTION
    score = round(100.0 * _clamp01(loss_rate / MAX_LOSS_FRACTION), 1)

    if score >= 65:
        severity = "CRITICAL"
    elif score >= 35:
        severity = "ELEVATED"
    elif score >= 15:
        severity = "WATCH"
    else:
        severity = "NORMAL"

    return RiskAssessment(
        risk_score=score,
        expected_loss=round(exposure.value_at_risk * loss_rate, 2),
        severity=severity,
        hazard=hazard,
        exposure=exposure,
        vulnerability=vulnerability,
    )
