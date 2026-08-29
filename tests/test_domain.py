"""Self-check for the domain layer. Run: python tests/test_domain.py (or pytest)."""

import math
import sys
from pathlib import Path

import pytest

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

    # The hazard maps are smooth and strictly monotone, so a component is never exactly 0 or 1.
    # A calm field therefore carries a small floor rather than a hard zero -- that is the point:
    # exact zeros tie 22% of the panel at the bottom of the queue where nothing can order them.
    calm = risk.assess_hazard(water_satisfaction=1.0, dry_spell_days=0, cumulative_dsv=0, heat_days=0)
    assert 0.0 < calm.combined < 0.05 and calm.dominant == "none"

    # At the published spray threshold the disease component reads RAMP_HI, not 1.0. Leaving
    # headroom is what lets a field facing disease AND drought outrank one facing disease alone.
    blight = risk.assess_hazard(water_satisfaction=0.95, dry_spell_days=1, cumulative_dsv=18, heat_days=0)
    assert blight.dominant == "disease"
    assert blight.disease == pytest.approx(risk.RAMP_HI, abs=1e-6)

    drought = risk.assess_hazard(water_satisfaction=0.3, dry_spell_days=21, cumulative_dsv=0, heat_days=0)
    assert drought.dominant == "drought"

    # The learned model enters as one more independent hazard, and must raise the combined value.
    with_model = risk.assess_hazard(water_satisfaction=0.95, dry_spell_days=1, cumulative_dsv=0,
                                    heat_days=0, vegetation=0.8)
    assert with_model.dominant == "vegetation"
    assert with_model.combined > calm.combined

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
    # Expected loss is no longer ever exactly zero: the hazard maps are strictly monotone, so a
    # calm, well-resourced farm carries a small floor (~1.3% of value at risk here) instead of a
    # hard 0.00. That is deliberate -- an exact zero is a tie, and 22% of the committed panel used
    # to sit on it, unrankable. The ordering, which is what the queue uses, is unaffected.
    assert bad.expected_loss > good.expected_loss > 0.0
    assert good.expected_loss < 0.05 * exposure.value_at_risk
    assert good.severity == "NORMAL"
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


def test_no_hazard_component_can_reach_the_noisy_or_absorbing_state():
    """noisy_or has an absorbing state at 1.0: once any component is exactly 1, prod(1 - h_i) is 0
    and every other component becomes irrelevant to the result. The clamped ramps this replaced hit
    it for 59.5% of the committed panel, leaving ONE distinct combined value across the whole top
    half of the queue. This is the invariant that must not regress.

    Floating point is the real adversary here, not algebra: 1/(1 + exp(-x)) rounds to exactly 1.0
    in float64 well before the mathematics says so, which is why `_no_saturate` guards the value.
    """
    extremes = [-1e12, -1e6, -50.0, -1.0, 0.0, 0.5, 1.0, 50.0, 1e6, 1e12]
    for u in extremes:
        h = risk._soft_ramp(u)
        assert 0.0 < h < 1.0, f"_soft_ramp({u}) = {h!r} reached a boundary"
    for z in extremes:
        v = risk.vegetation_hazard_from_anomaly(z)
        assert 0.0 <= v < 1.0, f"vegetation_hazard_from_anomaly({z}) = {v!r} reached 1.0"

    # Drive every input to its worst and confirm the combination still leaves headroom.
    worst = risk.assess_hazard(water_satisfaction=-1e6, dry_spell_days=10**6,
                               cumulative_dsv=10**6, heat_days=10**6, vegetation=1.0)
    assert worst.combined < 1.0, f"combined saturated at {worst.combined!r}"
    for name in ("drought", "disease", "heat", "vegetation"):
        assert getattr(worst, name) < 1.0, f"{name} saturated"

    # And the property that saturation destroys: adding a second severe hazard must still raise
    # the combined value. Under the old clamped ramps this was false whenever one component hit 1.
    one = risk.assess_hazard(water_satisfaction=0.0, dry_spell_days=30, cumulative_dsv=0, heat_days=0)
    two = risk.assess_hazard(water_satisfaction=0.0, dry_spell_days=30, cumulative_dsv=36, heat_days=0)
    assert two.combined > one.combined, "a second severe hazard did not raise combined hazard"


def test_evi_is_nan_where_undefined_not_a_plausible_number():
    """EVI's denominator `NIR + 6*Red - 7.5*Blue + 1` goes negative under haze, thin cloud, or bad
    atmospheric correction, and the ratio then blows up: measured on this repository's band cache,
    3.08% of denominators are negative and 4.03% of EVI values land outside [-1, 1], reaching 82.4.
    The committed panel carries the same contamination (2.63% of rows, range -32.4 .. 82.4).

    Two wrong fixes this test also pins down:
      * guarding only `denom == 0` -- the original code -- never fires, because the failure is a
        negative denominator, not a zero one;
      * clipping to [-1, 1] -- with denom < 0 over a green canopy the ratio is negative, so clipping
        reports a cloudy pixel over a HEALTHY field as -1, "worst possible vegetation".
    """
    healthy = indices.evi(nir=0.30, red=0.05, blue=0.04)
    assert 0.0 < healthy <= 1.0, f"a clean vegetated pixel should be in range, got {healthy!r}"

    # High blue over a green canopy: denominator is negative, EVI is undefined.
    hazy = indices.evi(nir=0.30, red=0.20, blue=0.50)
    assert math.isnan(hazy), f"expected NaN for a negative denominator, got {hazy!r}"

    assert math.isnan(indices.evi(nir=0.0, red=0.0, blue=1.0 / 7.5)), "denom == 0 must be NaN"

    # Nothing in range may ever escape [-1, 1]: sweep a grid of plausible reflectances and assert
    # every finite result is a valid index value.
    for nir in (0.0, 0.05, 0.2, 0.4, 0.7, 1.0):
        for red in (0.0, 0.05, 0.2, 0.5, 1.0):
            for blue in (0.0, 0.05, 0.15, 0.4, 1.0):
                v = indices.evi(nir, red, blue)
                assert math.isnan(v) or -1.0 <= v <= 1.0, f"evi({nir},{red},{blue}) = {v!r}"


def test_expected_yield_is_per_crop_and_reproduces_the_old_formula_at_its_baseline():
    """`_expected_yield` was crop-blind: `1.5 + (ndvi - 0.4) * 3.5` for maize grain and cassava
    tubers alike. Measured against GHS-Panel Wave 5 that is 1.84x too high for millet, 1.46x for
    maize, 1.38x for sorghum, and 0.67x too LOW for rice -- and yield drives 42.4% of the variance
    in `expected_loss`, so the bias lands squarely on the triage ranking.

    Two properties are pinned here:
      1. the NDVI term scales the crop's baseline, so with a 1.5 baseline it is EXACTLY the old
         formula -- the change generalises rather than replaces;
      2. crop ordering follows the survey (millet < maize < sorghum < rice < cassava), which the
         single global constant could not express at all.
    """
    from argotech.serving.pipeline import (
        DEFAULT_YIELD_T_HA,
        YIELD_BASELINE_T_HA,
        YIELD_NDVI_ANCHOR,
        YIELD_NDVI_SENSITIVITY,
        PredictionsService,
    )

    class NoReport:
        yield_value = None

    # 1. exact back-compatibility of the functional form at the old baseline
    for ndvi in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        old = 1.5 + (ndvi - YIELD_NDVI_ANCHOR) * 3.5
        new = 1.5 * (1.0 + YIELD_NDVI_SENSITIVITY * (ndvi - YIELD_NDVI_ANCHOR))
        assert abs(old - new) < 1e-9, f"form changed at ndvi={ndvi}: {old} vs {new}"

    # 2. crop ordering matches the survey medians, which a crop-blind constant cannot do
    y = {c: PredictionsService._expected_yield(NoReport(), 0.45, c)
         for c in ("millet", "maize", "sorghum", "rice", "cassava")}
    assert y["millet"] < y["maize"] < y["sorghum"] < y["rice"] < y["cassava"], y

    # an unknown or absent crop falls back to the pooled baseline, never to a crash
    assert PredictionsService._expected_yield(NoReport(), 0.45, None) > 0
    assert (PredictionsService._expected_yield(NoReport(), 0.45, "not-a-crop")
            == PredictionsService._expected_yield(NoReport(), 0.45, None))

    # 3. every crop stays inside its OWN observed range, at both NDVI extremes
    for crop, (_base, lo, hi) in YIELD_BASELINE_T_HA.items():
        for ndvi in (-1.0, 0.0, 0.5, 1.0, 2.0):
            v = PredictionsService._expected_yield(NoReport(), ndvi, crop)
            assert lo <= v <= hi, f"{crop} at ndvi={ndvi} gave {v}, outside [{lo}, {hi}]"

    # 4. a reported yield still wins outright -- imputation must never override real data
    class Reported:
        yield_value = 3.2
    assert PredictionsService._expected_yield(Reported(), 0.45, "maize") == 3.2

    # 5. the zero-guard survives: a zero is an unfilled field, not a farm that harvests nothing
    class Zero:
        yield_value = 0
    assert PredictionsService._expected_yield(Zero(), 0.45, "maize") > 0
    assert DEFAULT_YIELD_T_HA[1] > 0


def test_risk_score_and_severity_are_exposure_free():
    """`risk_score`, and therefore `severity` / `prediction` / `priority_label`, must not depend on
    exposure. This is the property that makes the served priority ordering exposure-free, and it is
    load-bearing for equity, not just tidiness.

    Measured on 3,011 Nigeria GHS-Panel households (docs/model-design.md 9.4): exposure and
    vulnerability correlate at spearman -0.410, because poorer households farm smaller plots. So an
    ordering that depends on exposure puts the LEAST vulnerable farmers first
    (spearman(rank, vulnerability) = -0.289), while an exposure-free one reverses that to +0.362.
    The two top-200 queues share only 5 members.

    If someone later folds `value_at_risk` into `risk_score` -- an easy and superficially reasonable
    change -- the served priority silently acquires that bias. This test fires instead.
    """
    hazard = risk.assess_hazard(water_satisfaction=0.35, dry_spell_days=12,
                                cumulative_dsv=6, heat_days=1)
    vuln = risk.assess_vulnerability({"has_irrigation": 0, "used_fertilizer": 1,
                                      "asset_score": 0.2, "market_access": 0.5})

    # Three farms differing ONLY in exposure, spanning the real GHS-Panel range (p5 $8 to p95 $1,310)
    tiny = risk.assess_risk(hazard, risk.assess_exposure(0.05, 0.8, 250.0), vuln)
    mid = risk.assess_risk(hazard, risk.assess_exposure(0.60, 1.2, 250.0), vuln)
    big = risk.assess_risk(hazard, risk.assess_exposure(5.00, 3.0, 420.0), vuln)

    assert tiny.risk_score == mid.risk_score == big.risk_score, (
        "risk_score moved with exposure: "
        f"{tiny.risk_score} / {mid.risk_score} / {big.risk_score}")
    assert tiny.severity == mid.severity == big.severity, "severity moved with exposure"

    # ...while expected_loss MUST move with exposure -- that is its whole purpose, and the reason
    # the two fields order a queue differently.
    assert tiny.expected_loss < mid.expected_loss < big.expected_loss

    # And vulnerability must still raise the rate, so an exposure-free ordering favours the
    # vulnerable rather than merely ignoring them.
    resilient = risk.assess_vulnerability({"has_irrigation": 1, "used_fertilizer": 1,
                                           "has_extension_access": 1, "received_credit": 1,
                                           "crop_diversity": 1.0, "asset_score": 0.9,
                                           "market_access": 0.9})
    fragile = risk.assess_vulnerability({"used_fertilizer": 1})
    same_exposure = risk.assess_exposure(0.6, 1.2, 250.0)
    assert (risk.assess_risk(hazard, same_exposure, fragile).risk_score
            > risk.assess_risk(hazard, same_exposure, resilient).risk_score)


def test_risk_score_can_order_a_queue_without_ties():
    """`risk_score` is the key an exposure-free triage queue sorts on, so its stored precision
    decides whether the queue has an order at all.

    docs/model-design.md 9.3a already learned this on `Hazard.combined` -- at 3dp, rounding alone
    collapsed 140 distinct top-half values back to 8, so it is stored at 6dp. `risk_score` was left
    at `round(score, 1)` and nobody checked, because only `combined` was audited. E11 measured the
    consequence on 3,011 GHS-Panel households: 566 distinct values across the population, the 50th
    and 51st farmer TIED, and therefore a top-50 that was an artifact of DataFrame row order rather
    than of risk. `evaluate.precision_at_k` returns the prevalence on exactly that input, which is
    the metric refusing to score an arbitrary selection.

    This test fires if the precision is ever reduced again.
    """
    exposure = risk.assess_exposure(0.6, 1.2, 250.0)
    vuln = risk.assess_vulnerability({"used_fertilizer": 1, "asset_score": 0.3})

    # Fields separated by small but real differences in water satisfaction -- the kind that sit next
    # to each other in a queue, and precisely the pairs a coarse rounding merges.
    scores = [risk.assess_risk(
        risk.assess_hazard(water_satisfaction=0.50 - i * 0.00001, dry_spell_days=5,
                           cumulative_dsv=4, heat_days=0),
        exposure, vuln).risk_score for i in range(40)]

    # The step is deliberately fine enough that the OLD `round(score, 1)` merges these fields --
    # verified by asserting the span is under a tenth of a point. Without this the test would pass
    # against the very defect it exists to catch, which is the failure mode
    # docs/superpowers/specs/2026-08-29-dataset-joining-design.md section 7 warns about by name.
    assert max(scores) - min(scores) < 0.1, (
        f"fields span {max(scores) - min(scores):.4f} points, which 1dp could resolve; this test "
        "would not detect a regression to coarser rounding")
    assert len(set(scores)) == len(scores), (
        f"risk_score merged {len(scores) - len(set(scores))} of {len(scores)} distinguishable "
        f"fields into ties; a queue cannot order them: {sorted(set(scores))}")
    # Strictly increasing: `i` lowers water satisfaction, which raises drought hazard. The order
    # itself must survive the rounding, not merely the distinctness.
    assert scores == sorted(scores)


def test_severity_is_decided_by_the_exact_score_not_the_rounded_one():
    """A display rounding must not decide a category.

    Severity previously read the ROUNDED `risk_score`, so a field at 64.96 was rounded up to 65.0
    and banded CRITICAL while a field at 64.94 was ELEVATED -- a band flip caused by a presentation
    choice. On E07's 3,011 households the two agree on every row, so this is not a bug report; it is
    a coupling removed before it could produce an outcome nobody could explain.
    """
    exposure = risk.assess_exposure(0.6, 1.2, 250.0)

    # The discriminating case, constructed rather than searched for. With vulnerability at 1.0 the
    # modulation `(0.5 + 0.5v)` is exactly 1, so `exact` is just 100 x combined. At combined 0.6497
    # the exact score is 64.97 -- ELEVATED -- but `round(64.97, 1)` is 65.0, which the old code read
    # as CRITICAL. A field's band must not depend on a display rounding.
    maxed = risk.Vulnerability(score=1.0, coping_capacity=0.0, gaps=[])
    boundary = risk.Hazard(drought=0.6497, disease=0.0, heat=0.0, vegetation=0.0,
                           pest=0.0, excess_rain=0.0, weeds=0.0,
                           combined=0.6497, dominant="drought")
    ra = risk.assess_risk(boundary, exposure, maxed)
    assert ra.risk_score == pytest.approx(64.97, abs=1e-6), ra.risk_score
    assert ra.severity == "ELEVATED", (
        f"score {ra.risk_score} banded {ra.severity}; a rounding decided the band")

    for ws in (0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
        for asset in (0.1, 0.5, 0.9):
            ra = risk.assess_risk(
                risk.assess_hazard(water_satisfaction=ws, dry_spell_days=6,
                                   cumulative_dsv=5, heat_days=0),
                exposure, risk.assess_vulnerability({"asset_score": asset}))
            bands = [(65, "CRITICAL"), (35, "ELEVATED"), (15, "WATCH"), (-1, "NORMAL")]
            expected = next(name for cut, name in bands if ra.risk_score >= cut)
            assert ra.severity == expected, (
                f"score {ra.risk_score} banded {ra.severity}, expected {expected}")
