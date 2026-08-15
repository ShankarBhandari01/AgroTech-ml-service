"""Checks on what the API actually says to the people reading it.

Both cases here were found by running the service and reading a real response, not by a failing
assertion — which is why they are worth pinning: nothing else in the suite looks at wording.
"""

from __future__ import annotations

from argotech.domain import risk
from argotech.serving.schemas.response import PredictionProbabilities


def test_a_coping_gap_never_reads_as_though_the_farmer_has_the_thing():
    """The original defect: a missing `has_irrigation` rendered as "…: has irrigation".

    Mechanically stripping the underscore inverted the meaning of every boolean factor, in a string
    shown to extension officers deciding where to drive.
    """
    for factor in risk.COPING_FACTORS:
        label = risk.describe_gap(factor)
        assert label, f"{factor} has no wording"
        assert not label.startswith("has "), f"{factor} reads as possession: {label!r}"
        assert "Limited coping capacity: has" not in label


def test_every_coping_factor_has_deliberate_wording():
    """Adding a factor without wording should be caught here, not in production copy."""
    assert set(risk.COPING_FACTORS) <= set(risk.COPING_GAP_LABELS)


def test_the_fallback_cannot_invert_meaning_either():
    """An unlabelled factor must still read as absence."""
    assert risk.describe_gap("has_tractor").startswith("Missing:")


def test_probabilities_carry_what_they_are_over():
    """`probabilities` describes the vegetation hazard, not overall risk.

    A consumer reading raw JSON sees low=0.66 beside prediction=2 and reasonably concludes the two
    contradict each other; the payload has to say which question the probabilities answer.
    """
    p = PredictionProbabilities(low=0.662, medium=0.211, high=0.127)
    assert "vegetation hazard" in p.of
    assert "of" in p.model_dump(), "the caveat must survive serialisation"


def test_probabilities_still_serialise_the_original_three_keys():
    """The Kotlin backend consumes this object; the addition must be purely additive."""
    dumped = PredictionProbabilities(low=1.0, medium=0.0, high=0.0).model_dump()
    assert {"low", "medium", "high"} <= set(dumped)
    assert dumped["low"] == 1.0
