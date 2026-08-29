"""The reference distribution both paths standardise against.

`ndvi_z_peer` meant two different things — a concurrent cross-site cohort in training, the site's
own 12-month history at serving, correlated only 0.626 with 25.5% sign flips. These pin the shape
of the single replacement so the two callers cannot drift again.
"""

from __future__ import annotations

import math

import pandas as pd

from argotech.features.agronomic import (
    PEER_RELATIVE,
    peer_bucket,
    peer_reference,
    peer_stats,
)


def _frame() -> pd.DataFrame:
    rows = []
    for cluster, base in (("A", 0.5), ("B", 0.2)):
        for month in ("03", "09"):
            for i in range(10):
                rows.append({"cluster": cluster,
                             "obs_date": f"2025-{month}-1{i % 9}",
                             "ndvi": base + i * 0.01,
                             "rvi": base + i * 0.02})
    return pd.DataFrame(rows)


def test_the_bucket_is_cluster_and_calendar_month():
    assert peer_bucket("Kaduna_Grain_Belt", "2026-09-14") == "Kaduna_Grain_Belt|09"
    assert peer_bucket(None, "2026-09-14") is None, "no cluster means no reference"


def test_stats_are_keyed_per_cluster_and_month_for_every_peer_column():
    stats = peer_stats(_frame())
    assert set(stats) == {"A|03", "A|09", "B|03", "B|09"}
    for bucket in stats:
        assert set(stats[bucket]) == set(PEER_RELATIVE)
        for mu, sigma in stats[bucket].values():
            assert math.isfinite(mu) and sigma > 0.0


def test_a_cluster_offset_does_not_leak_across_buckets():
    """A's mean must not move when B's values change — the property that makes per-fold stats safe."""
    df = _frame()
    a_only = peer_stats(df[df.cluster == "A"])
    both = peer_stats(df)
    assert a_only["A|03"]["ndvi"] == both["A|03"]["ndvi"]


def test_an_unknown_bucket_yields_no_reference_rather_than_a_default():
    """None, not [0.0, 1.0]. A fabricated reference asserts 'average for its region' about a region
    that was never measured — the same fabrication `cluster_relative_row` refuses to make."""
    stats = peer_stats(_frame())
    assert peer_reference(stats, "A|03", "ndvi") is not None
    assert peer_reference(stats, "Z|03", "ndvi") is None
    assert peer_reference(stats, None, "ndvi") is None
    assert peer_reference(None, "A|03", "ndvi") is None
    assert peer_reference(stats, "A|03", "not_a_column") is None
