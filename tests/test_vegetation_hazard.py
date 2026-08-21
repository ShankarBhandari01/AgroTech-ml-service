"""The persistence vegetation hazard, and its encoding on the wire.

Run: python tests/test_vegetation_hazard.py (or pytest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argotech.domain import risk
from argotech.serving.pipeline import _severity_to_probabilities


def test_the_severe_threshold_matches_the_label_it_mirrors():
    """risk.SEVERE_ANOMALY_Z restates training.dataset.SEVERE_Z because serving must not import the
    training package. Restated constants drift; this is what stops it silently."""
    from argotech.training.dataset import SEVERE_Z

    assert risk.SEVERE_ANOMALY_Z == SEVERE_Z


def test_hazard_is_monotone_bounded_and_centred_on_the_severe_cut():
    f = risk.vegetation_hazard_from_anomaly
    zs = [3.0, 1.0, 0.0, -0.35, -1.0, -2.0, -5.0]
    hs = [f(z) for z in zs]

    assert all(0.0 <= h <= 1.0 for h in hs), hs
    assert hs == sorted(hs), "a worse anomaly must never lower the hazard"
    assert len(set(hs)) == len(hs), "no ties — ties are invisible to a ranking, which is the point"
    assert abs(f(risk.SEVERE_ANOMALY_Z) - 0.5) < 1e-9, "0.5 exactly at the severe cut"
    assert f(-5.0) > 0.99 and f(3.0) < 0.01


def test_absent_anomaly_is_no_hazard_not_a_guess():
    """No canopy evidence must leave the physical hazards to carry the assessment, as when no
    satellite scene exists at all — never a fabricated mid-range value."""
    assert risk.vegetation_hazard_from_anomaly(None) == 0.0
    assert risk.vegetation_hazard_from_anomaly(float("nan")) == 0.0


def test_probability_encoding_preserves_expected_severity():
    """The encoding exists so both sources agree on `0.5*medium + 1.0*high`. If that breaks, a
    consumer computing expected severity silently gets two different numbers per source."""
    for h in [0.0, 0.05, 0.25, 0.5, 0.5001, 0.75, 1.0]:
        p = _severity_to_probabilities(h)
        assert abs((p.low + p.medium + p.high) - 1.0) < 2e-3, (h, p)
        assert min(p.low, p.medium, p.high) >= 0.0, (h, p)
        assert abs((0.5 * p.medium + 1.0 * p.high) - h) < 2e-3, (h, p)

    # Out-of-range input is clamped rather than producing a negative probability.
    assert _severity_to_probabilities(-1.0).low == 1.0
    assert _severity_to_probabilities(2.0).high == 1.0


def test_persistence_out_ranks_the_hard_ramp_it_replaced():
    """The reason for the logistic: a clipped ramp ties every field above -0.35 at exactly 0.0.

    Reproduced in miniature — the ramp collapses four distinct fields to one value, the logistic
    keeps all four orderable.
    """
    zs = [0.5, 0.0, -0.2, -0.34]
    ramp = [max(0.0, min(1.0, (-0.35 - z) / (-0.35 + 1.0))) for z in zs]
    logistic = [risk.vegetation_hazard_from_anomaly(z) for z in zs]

    assert len(set(ramp)) == 1, "the ramp ties them all"
    assert len(set(logistic)) == 4, "the logistic keeps them orderable"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok: {name}")
    print("\nall vegetation-hazard checks passed")
