"""The manifest is what makes a reported number traceable to the data that produced it.

docs/RESEARCH_SUMMARY.md records a metrics table cited by the README and two source files that
exists in no committed file. A content hash beside every panel is how that stops happening.
"""

from __future__ import annotations

import json
import statistics
from datetime import date, timedelta

import pandas as pd

from argotech.lab.panel import panel as panel_mod
from argotech.lab.panel.panel import build_samples, manifest, write_panel


def _df() -> pd.DataFrame:
    return pd.DataFrame({"site_id": ["A", "A", "B"], "cluster": ["C0", "C0", "C1"],
                         "obs_date": ["2025-01-01", "2025-02-01", "2025-01-01"],
                         "forward_z": [0.1, -0.2, 0.3]})


def test_manifest_describes_the_panel():
    m = manifest(_df())
    assert m["rows"] == 3 and m["sites"] == 2 and m["clusters"] == ["C0", "C1"]
    assert m["date_min"] == "2025-01-01" and m["date_max"] == "2025-02-01"
    assert len(m["content_hash"]) == 64
    assert m["code_version"], "the code version that built the panel must be recorded"


def test_the_hash_is_stable_across_identical_frames():
    assert manifest(_df())["content_hash"] == manifest(_df())["content_hash"]


def test_the_hash_is_stable_across_row_order():
    shuffled = _df().iloc[::-1].reset_index(drop=True)
    assert manifest(_df())["content_hash"] == manifest(shuffled)["content_hash"], \
        "row order is not data; a re-sorted panel is the same panel"


def test_the_hash_changes_when_a_value_changes():
    changed = _df()
    changed.loc[0, "forward_z"] = 0.10001
    assert manifest(_df())["content_hash"] != manifest(changed)["content_hash"]


def test_write_panel_emits_a_sidecar(tmp_path):
    p = write_panel(_df(), tmp_path / "panel.parquet")
    side = json.loads(p.with_suffix(".parquet.manifest.json").read_text())
    assert side["content_hash"] == manifest(_df())["content_hash"]
    assert pd.read_parquet(p).shape == (3, 4)


# ---------------------------------------------------------------------------------------------
# NDVI validity and the peer-standardisation leak fix (build_samples)
# ---------------------------------------------------------------------------------------------
#
# `build_samples` consumes the shape `collect_site` returns, so these fixtures build that shape
# directly rather than hitting the network. WINDOW_DAYS=90, so `_daily` spans a wide enough range
# that a 2024-04-15 observation always has a full look-back window.

_DAILY_START = date(2024, 1, 1)
_DAILY_DAYS = 400


def _daily() -> dict:
    times = [(_DAILY_START + timedelta(days=i)).isoformat() for i in range(_DAILY_DAYS)]
    return {
        "time": times,
        "temperature_2m_max": [30.0] * _DAILY_DAYS,
        "temperature_2m_min": [18.0] * _DAILY_DAYS,
        "precipitation_sum": [2.0] * _DAILY_DAYS,
        "et0_fao_evapotranspiration": [4.0] * _DAILY_DAYS,
        "relative_humidity_2m_mean": [60.0] * _DAILY_DAYS,
        "shortwave_radiation_sum": [20.0] * _DAILY_DAYS,
    }


def _obs(sensing_date: str, ndvi: float) -> dict:
    return {"sensing_date": sensing_date, "ndvi": ndvi, "ndwi": 0.1, "evi": 0.3}


def _collected(site_id: str, cluster: str, history: list[dict]) -> dict:
    return {
        "site_id": site_id, "cluster": cluster, "cluster_idx": 0,
        "latitude": 10.0, "longitude": 8.0, "elevation": 500.0,
        "daily": _daily(), "history": history, "sar": [],
    }


_FEATURE_DATE = "2024-04-15"   # day index 105 in _daily — comfortably past the 90-day look-back
_LABEL_DATE = "2024-05-10"     # 25 days later, inside MAX_LABEL_GAP_DAYS


def _peer_sites(values: list[float], prefix: str = "P") -> list[dict]:
    return [_collected(f"{prefix}{i}", "C0", [_obs(_LABEL_DATE, v)]) for i, v in enumerate(values)]


def test_cohort_and_label_exclude_ndvi_below_zero():
    """A negative-NDVI peer (water/cloud/shadow, never canopy) must never reach the mean/sd, and a
    field whose own label observation is negative must be dropped rather than labelled."""
    field = _collected("F", "C0", [_obs(_FEATURE_DATE, 0.5), _obs(_LABEL_DATE, 0.6)])
    valid_peers = [0.50, 0.55, 0.45, 0.60, 0.65]
    peers = _peer_sites(valid_peers)
    invalid_peer = _collected("P_bad", "C0", [_obs(_LABEL_DATE, -0.1)])

    df = build_samples([field, *peers, invalid_peer])
    assert len(df) == 1, "the field's own row must still be produced from the 5 valid peers"
    mean, sd = statistics.fmean(valid_peers), statistics.pstdev(valid_peers)
    assert df.iloc[0]["forward_z"] == round((0.6 - mean) / sd, 4), \
        "the negative-ndvi peer must not have entered the mean/sd"


def test_a_negative_ndvi_label_observation_is_rejected():
    valid_peers = [0.50, 0.55, 0.45, 0.60, 0.65]
    field = _collected("F", "C0", [_obs(_FEATURE_DATE, 0.5), _obs(_LABEL_DATE, -0.05)])
    df = build_samples([field, *_peer_sites(valid_peers)])
    assert df.empty, "a water/cloud/shadow pixel is not a valid label, even with a healthy cohort"


def test_a_peer_sharing_the_fields_exact_ndvi_value_is_retained():
    """Regression test for the value-based exclusion bug: `if v != label_obs['ndvi']` used to strip
    every cohort entry equal to the field's own value, including a genuine peer that happened to
    match. Excluding by site identity instead must keep that peer and still produce a row."""
    field = _collected("F", "C0", [_obs(_FEATURE_DATE, 0.5), _obs(_LABEL_DATE, 0.5)])
    same_value_peer = _collected("P_same", "C0", [_obs(_LABEL_DATE, 0.5)])
    other_peers = _peer_sites([0.40, 0.45, 0.55, 0.60])

    df = build_samples([field, same_value_peer, *other_peers])
    assert len(df) == 1, (
        "value-based exclusion would drop both the field's own entry and P_same (also 0.5), "
        "leaving only 4 peers and no row at all")
    peers = [0.5, 0.40, 0.45, 0.55, 0.60]
    mean, sd = statistics.fmean(peers), statistics.pstdev(peers)
    assert df.iloc[0]["forward_z"] == round((0.5 - mean) / sd, 4), \
        "P_same must have been included in the mean/sd"


def test_the_sd_guard_rejects_a_near_degenerate_cohort():
    field = _collected("F", "C0", [_obs(_FEATURE_DATE, 0.5), _obs(_LABEL_DATE, 0.501)])
    near_constant_peers = [0.500, 0.500, 0.500, 0.500, 0.501]
    assert statistics.pstdev(near_constant_peers) < panel_mod.MIN_COHORT_SD
    df = build_samples([field, *_peer_sites(near_constant_peers)])
    assert df.empty, "a near-constant cohort must not be turned into a z-score"


def test_a_cohort_spread_above_the_guard_still_produces_a_row():
    field = _collected("F", "C0", [_obs(_FEATURE_DATE, 0.5), _obs(_LABEL_DATE, 0.5)])
    peers = [0.50, 0.50, 0.50, 0.50, 0.55]
    assert statistics.pstdev(peers) > panel_mod.MIN_COHORT_SD
    df = build_samples([field, *_peer_sites(peers)])
    assert len(df) == 1


# ---------------------------------------------------------------------------------------------
# `collect_site`'s pinned `end_date` (reproducible rebuilds)
# ---------------------------------------------------------------------------------------------

def test_a_pinned_end_date_gives_the_same_archive_window_regardless_of_todays_date(monkeypatch):
    calls = []

    def fake_fetch_archive(lat, lon, start, end):
        calls.append((start, end))
        s, e = date.fromisoformat(start), date.fromisoformat(end)
        n = (e - s).days + 1
        times = [(s + timedelta(days=i)).isoformat() for i in range(n)]
        return {"elevation": 100.0, "daily": {
            "time": times,
            "temperature_2m_max": [30.0] * n, "temperature_2m_min": [18.0] * n,
            "precipitation_sum": [2.0] * n, "et0_fao_evapotranspiration": [4.0] * n,
            "relative_humidity_2m_mean": [60.0] * n, "shortwave_radiation_sum": [20.0] * n,
        }}

    fixed_history = [_obs("2024-01-01", 0.5)] * 6
    monkeypatch.setattr(panel_mod.meteo, "fetch_archive", fake_fetch_archive)
    monkeypatch.setattr(panel_mod, "_sentinel_history", lambda site, days: fixed_history)
    monkeypatch.setattr(panel_mod, "_sar_history", lambda site, days: [])

    class _FrozenToday(date):
        _now = date(2024, 6, 1)

        @classmethod
        def today(cls):
            return cls._now

    monkeypatch.setattr(panel_mod, "date", _FrozenToday)
    site = {"site_id": "S0", "cluster": "C0", "cluster_idx": 0, "latitude": 10.0, "longitude": 8.0}
    pinned = date(2024, 6, 1)

    r1 = panel_mod.collect_site(site, years=1, end_date=pinned)
    _FrozenToday._now = date(2024, 9, 1)     # "today" moves on; the pin must not
    r2 = panel_mod.collect_site(site, years=1, end_date=pinned)
    assert calls[0] == calls[1], "a pinned end_date must give the identical archive window"
    assert r1["daily"] == r2["daily"]

    # Without a pin, the same drift in "today" changes the requested window — the default is
    # unchanged, which is exactly why a rebuild today is not reproducible.
    panel_mod.collect_site(site, years=1)
    assert calls[2] != calls[0], "unpinned collection must still track date.today()"
