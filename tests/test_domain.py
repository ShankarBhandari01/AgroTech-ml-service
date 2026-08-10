"""Self-check for the domain layer. Run: python tests/test_domain.py (or pytest)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argotech.domain import agronomy, indices, risk


def test_indices():
    assert abs(indices.ndvi(0.4, 0.1) - 0.6) < 1e-9
    assert indices.ndvi(0.0, 0.0) == 0.0                      # no ZeroDivisionError
    assert indices.savi(0.4, 0.1, 0.5) > indices.ndvi(0.4, 0.1) * 0.5  # sparse-canopy correction
    assert indices.anomaly_z(0.5, 0.5, 0.0) == 0.0            # degenerate baseline
    assert abs(indices.vci(0.5, 0.2, 0.8) - 50.0) < 1e-6
    assert indices.vci(0.1, 0.2, 0.8) == 0.0                  # clamped, not negative

    healthy = indices.crop_health_index(0.75, 0.30, 0.2, 0.8, peer_mean=0.55, peer_std=0.1)
    stressed = indices.crop_health_index(0.25, -0.10, 0.2, 0.8, peer_mean=0.55, peer_std=0.1)
    assert healthy.score > stressed.score
    assert healthy.status == "Healthy" and stressed.status in ("Stressed", "Severely Stressed")
    assert 0.0 <= stressed.vigour <= 1.0


def test_agronomy():
    # Upper cutoff: a lethal 45 C day must not score more thermal time than an ideal 30 C day.
    assert agronomy.growing_degree_days(20, 45) == agronomy.growing_degree_days(20, 30)
    assert agronomy.growing_degree_days(5, 8) == 0.0           # below base

    assert agronomy.phenology_stage(50) == "Emergence"
    assert agronomy.phenology_stage(900) == "Flowering"
    assert agronomy.phenology_stage(5000) == "Post-Harvest"

    wb = agronomy.water_balance([0.0] * 10 + [5.0] * 4, [5.0] * 14, "Flowering")
    assert wb.deficit_mm > 0 and wb.longest_dry_spell_days == 10
    assert 0.0 <= wb.satisfaction <= 1.0
    wet = agronomy.water_balance([20.0] * 14, [5.0] * 14, "Flowering")
    assert wet.deficit_mm == 0.0 and wet.satisfaction == 1.0

    # Heat only counts during flowering.
    assert agronomy.heat_stress_days([38, 39, 40], "Flowering") == 3
    assert agronomy.heat_stress_days([38, 39, 40], "Vegetative") == 0

    assert agronomy.daily_severity_value(20.0, 24) == 4        # warm + long wet period
    assert agronomy.daily_severity_value(20.0, 5) == 0         # too dry
    assert agronomy.daily_severity_value(35.0, 24) == 0        # outside the model's temp range
    total, spray = agronomy.accumulate_dsv([4, 4, 4, 4, 4])
    assert total == 20 and spray is True


def test_risk():
    assert abs(risk.noisy_or([0.4, 0.4]) - 0.64) < 1e-9        # hazards compound
    assert risk.noisy_or([]) == 0.0

    calm = risk.assess_hazard(water_satisfaction=1.0, dry_spell_days=0, cumulative_dsv=0, heat_days=0)
    assert calm.combined == 0.0 and calm.dominant == "none"

    blight = risk.assess_hazard(water_satisfaction=0.95, dry_spell_days=1, cumulative_dsv=18, heat_days=0)
    assert blight.dominant == "disease" and blight.disease == 1.0

    drought = risk.assess_hazard(water_satisfaction=0.3, dry_spell_days=21, cumulative_dsv=0, heat_days=0)
    assert drought.dominant == "drought"

    # A protected attribute must raise, not be silently dropped.
    try:
        risk.assess_vulnerability({"asset_score": 0.5, "head_gender": 1})
        raise AssertionError("expected protected-attribute guard to fire")
    except ValueError as e:
        assert "head_gender" in str(e)

    resilient = risk.assess_vulnerability(
        {"has_irrigation": 1, "has_extension_access": 1, "received_credit": 1,
         "used_fertilizer": 1, "crop_diversity": 1.0, "asset_score": 0.9, "market_access": 0.8})
    fragile = risk.assess_vulnerability({"used_fertilizer": 1})
    assert resilient.score < fragile.score
    assert "has_irrigation" in fragile.gaps and not resilient.gaps

    exposure = risk.assess_exposure(area_ha=2.0, expected_yield_t_ha=3.0, price_per_t=250.0)
    assert exposure.value_at_risk == 1500.0

    bad = risk.assess_risk(drought, exposure, fragile)
    good = risk.assess_risk(calm, exposure, resilient)
    assert bad.risk_score > good.risk_score
    assert bad.expected_loss > good.expected_loss == 0.0
    assert good.severity == "NORMAL"
    assert 0.0 <= bad.risk_score <= 100.0

    # Same hazard, same exposure: the more vulnerable farm ranks higher, but the resilient one is
    # still flagged rather than zeroed out.
    a = risk.assess_risk(drought, exposure, fragile)
    b = risk.assess_risk(drought, exposure, resilient)
    assert a.risk_score > b.risk_score > 0.0


if __name__ == "__main__":
    test_indices()
    test_agronomy()
    test_risk()
    print("domain self-check passed")
