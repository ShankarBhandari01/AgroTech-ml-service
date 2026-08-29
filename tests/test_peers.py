"""The peer reference, fitted where it can be fitted honestly.

panel.py computed `peer_stats` over the whole frame and baked the result in as a static column,
while `peer_stats`'s own docstring said it must be computed per fold. Measured consequences: 74% of
an early training fold's peer cohort lay in its own future, and a held-out cluster was standardised
entirely on held-out rows.
"""

from __future__ import annotations

import pandas as pd

from argotech.lab.estimand.peers import apply_peer_z, fit_peer_stats, peer_key


def _df() -> pd.DataFrame:
    rows = []
    for cluster, lat, elev in (("A", 7.0, 100.0), ("B", 11.0, 700.0), ("C", 0.5, 1800.0)):
        for year in (2024, 2025):
            for month in (1, 6):
                for i in range(6):
                    rows.append({"cluster": cluster, "latitude": lat, "elevation": elev,
                                 "site_id": f"{cluster}{i}",
                                 "obs_date": f"{year}-{month:02d}-1{i}",
                                 "ndvi": 0.3 + 0.02 * i + (0.2 if cluster == "C" else 0.0),
                                 "rvi": 0.4 + 0.01 * i})
    return pd.DataFrame(rows)


def test_cluster_month_key_is_cluster_and_month():
    df = _df()
    assert peer_key(df, "cluster_month").iloc[0] == "A|01"


def test_geo_month_key_drops_cluster_identity():
    keys = peer_key(_df(), "geo_month", lat_band=20.0, elev_band=1000.0)
    assert "A" not in keys.iloc[0] and "|01" in keys.iloc[0]


def test_geo_month_lets_a_held_out_cluster_borrow_peers():
    """The whole point: an unseen region must still get a reference from similar places."""
    df = _df()
    for kind, expect in (("cluster_month", False), ("geo_month", True)):
        train, test = df[df.cluster != "A"], df[df.cluster == "A"]
        _, coverage = apply_peer_z(test, fit_peer_stats(train, kind), kind)
        assert (coverage > 0) is expect, f"{kind}: coverage {coverage}"


def test_cluster_month_gives_a_held_out_cluster_no_reference_at_all():
    df = _df()
    train, test = df[df.cluster != "A"], df[df.cluster == "A"]
    out, coverage = apply_peer_z(test, fit_peer_stats(train, "cluster_month"), "cluster_month")
    assert coverage == 0.0
    assert out["ndvi_z_peer"].isna().all(), "no reference must mean NaN, never a fabricated 0.0"


def test_stats_fitted_on_train_ignore_test_rows_entirely():
    """The leak, stated as a property: a test row's value must not move the reference."""
    df = _df()
    train = df[df.cluster != "A"]
    before = fit_peer_stats(train, "geo_month")
    poisoned = df.copy()
    poisoned.loc[poisoned.cluster == "A", "ndvi"] = 99.0
    after = fit_peer_stats(poisoned[poisoned.cluster != "A"], "geo_month")
    assert before == after, "a held-out row changed the reference fitted on training rows"


def test_apply_rewrites_both_peer_columns():
    df = _df()
    out, coverage = apply_peer_z(df, fit_peer_stats(df, "cluster_month"), "cluster_month")
    assert coverage == 1.0
    for col in ("ndvi_z_peer", "rvi_z_peer"):
        assert col in out.columns and out[col].notna().any()
    assert abs(out["ndvi_z_peer"].mean()) < 0.5


def test_coverage_reports_the_share_with_a_reference():
    df = _df()
    train = df[df.cluster != "A"]
    _, coverage = apply_peer_z(df, fit_peer_stats(train, "cluster_month"), "cluster_month")
    expected = 1.0 - len(df[df.cluster == "A"]) / len(df)
    assert abs(coverage - expected) < 1e-9


def test_a_single_observation_bucket_yields_no_reference():
    """sd over one row is NaN; a reference we cannot measure must not be invented."""
    one = _df().head(1)
    stats = fit_peer_stats(one, "cluster_month")
    out, coverage = apply_peer_z(one, stats, "cluster_month")
    assert coverage == 0.0 and out["ndvi_z_peer"].isna().all()
