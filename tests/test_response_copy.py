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


def test_probabilities_contains_only_numbers():
    """The Kotlin client types this as `Map<String, Double>` (PythonMlResponse.kt).

    A string-valued entry here is not an "unknown property" that @JsonIgnoreProperties skips — it
    is a map value that cannot coerce to Double. FastApiMlClientImpl catches the resulting
    exception and returns its static FALLBACK, so the symptom would be every prediction silently
    degrading while this service reported 200. Keep this object numeric.
    """
    dumped = PredictionProbabilities(low=0.662, medium=0.211, high=0.127).model_dump()
    assert set(dumped) == {"low", "medium", "high"}, (
        f"non-numeric key would break the Kotlin Map<String, Double> binding: {set(dumped)}")
    assert all(isinstance(v, float) for v in dumped.values())


def test_the_caveat_rides_at_the_response_root_where_it_is_safe_to_ignore():
    """`PythonMlResponse` carries @JsonIgnoreProperties(ignoreUnknown = true), so an unknown field
    at the root is skipped rather than fatal. That is the only safe place to add one."""
    from argotech.serving.schemas.response import PredictionResponse

    field = PredictionResponse.model_fields["probabilities_of"]
    assert "vegetation hazard" in field.default
