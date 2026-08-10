"""Published agronomic models: thermal time, phenology, water balance, disease pressure.

These are physical/epidemiological relationships from the agronomy literature, not fitted models.
They need no training data, they transfer across regions, and they give the learned layer features
that already carry agronomic meaning instead of raw weather numbers.

Every function here is a pure function of already-fetched observations.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------------------------
# Thermal time and phenology
# ---------------------------------------------------------------------------------------------

# Base temperature (°C) and cumulative GDD at the end of each stage, per crop.
# Maize values follow the standard base-10 °C scale; sorghum and rice are the common FAO figures.
# ponytail: three crops hard-coded, move to a crops table once agronomists want to tune per-variety.
CROP_GDD = {
    "maize":   {"base": 10.0, "cap": 30.0, "stages": [(120, "Emergence"), (700, "Vegetative"), (1000, "Flowering"), (1500, "Grain Fill"), (1700, "Maturity")]},
    "sorghum": {"base": 10.0, "cap": 34.0, "stages": [(130, "Emergence"), (750, "Vegetative"), (1100, "Flowering"), (1650, "Grain Fill"), (1900, "Maturity")]},
    "rice":    {"base": 10.0, "cap": 32.0, "stages": [(100, "Emergence"), (800, "Vegetative"), (1100, "Flowering"), (1600, "Grain Fill"), (1800, "Maturity")]},
}
DEFAULT_CROP = "maize"


def growing_degree_days(tmin: float, tmax: float, base: float = 10.0, cap: float = 30.0) -> float:
    """Daily thermal time with both a base and an upper cutoff.

    The cutoff matters in the Sahel: without it a 42 °C day is scored as excellent growth when it is
    in fact lethal to pollen.
    """
    tmax_c = min(tmax, cap)
    tmin_c = max(min(tmin, cap), base)
    return max(0.0, (tmax_c + tmin_c) / 2.0 - base)


def phenology_stage(cumulative_gdd: float, crop: str = DEFAULT_CROP) -> str:
    """Map accumulated thermal time since planting to a growth stage.

    Replaces the hard-coded "Vegetative / Flowering" string: stage drives crop water requirement,
    heat-stress sensitivity and which advisory is even actionable.
    """
    spec = CROP_GDD.get(crop.strip().lower(), CROP_GDD[DEFAULT_CROP])
    for threshold, name in spec["stages"]:
        if cumulative_gdd < threshold:
            return name
    return "Post-Harvest"


# FAO-56 single crop coefficient by stage — how much of reference ET the canopy actually transpires.
KC_BY_STAGE = {
    "Emergence": 0.35,
    "Vegetative": 0.75,
    "Flowering": 1.20,
    "Grain Fill": 1.05,
    "Maturity": 0.60,
    "Post-Harvest": 0.30,
}


# ---------------------------------------------------------------------------------------------
# Water balance
# ---------------------------------------------------------------------------------------------

@dataclass
class WaterBalance:
    requirement_mm: float   # Kc * ET0 over the window
    rainfall_mm: float
    deficit_mm: float       # unmet demand, 0 when rainfall covers it
    satisfaction: float     # 0-1, rainfall / requirement, capped
    longest_dry_spell_days: int


def water_balance(daily_rain_mm: list[float], daily_et0_mm: list[float], stage: str,
                  irrigation_mm: float = 0.0, dry_day_threshold: float = 1.0) -> WaterBalance:
    """Crop water balance over a window, FAO-56 style.

    `daily_et0_mm` is reference evapotranspiration — Open-Meteo returns it directly as
    `et0_fao_evapotranspiration`, so no Penman-Monteith implementation is needed here. Water stress
    is demand minus supply; total rainfall alone (what the current pipeline uses) cannot express it,
    because 40 mm is generous at emergence and a drought at flowering.
    """
    kc = KC_BY_STAGE.get(stage, 1.0)
    requirement = kc * sum(daily_et0_mm)
    supply = sum(daily_rain_mm) + irrigation_mm
    deficit = max(0.0, requirement - supply)
    satisfaction = 1.0 if requirement <= 1e-6 else min(1.0, supply / requirement)

    longest = current = 0
    for r in daily_rain_mm:
        current = current + 1 if r < dry_day_threshold else 0
        longest = max(longest, current)

    return WaterBalance(
        requirement_mm=round(requirement, 1),
        rainfall_mm=round(supply, 1),
        deficit_mm=round(deficit, 1),
        satisfaction=round(satisfaction, 3),
        longest_dry_spell_days=longest,
    )


def heat_stress_days(daily_tmax: list[float], stage: str, threshold: float = 35.0) -> int:
    """Days above the pollen-viability threshold. Only counted during flowering, when a single
    such day can cost a large share of the harvest; outside flowering the crop recovers."""
    if stage != "Flowering":
        return 0
    return sum(1 for t in daily_tmax if t >= threshold)


# ---------------------------------------------------------------------------------------------
# Disease pressure
# ---------------------------------------------------------------------------------------------

# Wallin's late-blight severity table: (temp_lo, temp_hi, [(min_wet_hours, dsv), ...] descending).
_WALLIN = [
    (7.2, 11.6, [(25, 4), (22, 3), (19, 2), (16, 1)]),
    (11.7, 15.0, [(22, 4), (19, 3), (16, 2), (13, 1)]),
    (15.1, 26.6, [(23, 4), (19, 3), (15, 2), (11, 1)]),
]


def daily_severity_value(mean_temp_during_wet: float, wet_hours: int) -> int:
    """One day's Disease Severity Value (Wallin 1962, the basis of BLITECAST).

    Leaf wetness duration is proxied by hours at RH >= 90%. This replaces the repo's ad-hoc
    "hours at RH >= 85" and "hours in 18-24 °C" counters with the published model those counters
    were approximating — same inputs, validated thresholds, and an accumulation with a spray
    threshold that means something operationally.
    """
    for lo, hi, table in _WALLIN:
        if lo <= mean_temp_during_wet <= hi:
            for min_hours, dsv in table:
                if wet_hours >= min_hours:
                    return dsv
            return 0
    return 0


def accumulate_dsv(daily_values: list[int], spray_threshold: int = 18) -> tuple[int, bool]:
    """Cumulative DSV since the last spray, and whether the spray threshold is reached."""
    total = sum(daily_values)
    return total, total >= spray_threshold


def fall_armyworm_generations(cumulative_gdd_base10: float, dd_per_generation: float = 390.0) -> float:
    """Completed FAW generations since the biofix. The dominant maize pest across Sub-Saharan
    Africa since 2016; degree-day accumulation is how scouting campaigns are timed."""
    return round(cumulative_gdd_base10 / dd_per_generation, 2)
