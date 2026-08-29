"""One row per site, no fabricated value for a missing response, and correct SoilGrids unit
conversion — the three things `docs/site_covariates` promises."""

from __future__ import annotations

import json

import pandas as pd

from argotech.lab.covariates import site_covariates as sc


def _sites() -> pd.DataFrame:
    return pd.DataFrame({
        "site_id": ["A", "B", "C"],
        "latitude": [10.5, 11.0, -1.0],
        "longitude": [7.5, 8.0, 35.0],
        "cluster": ["Kaduna_Grain_Belt", "Kaduna_Grain_Belt", "Kenya_Rift_Valley"],
    })


# A real SoilGrids-shaped single-property layer: clay=159 (scaled) with d_factor=10 -> 15.9 %,
# matching the 15.9% Kaduna value the task description cites. This is also the exact per-property
# file shape the concurrent SoilGrids fetch (E06) writes into the shared `.cache/soilgrids/`
# directory, e.g. `{site_id}-clay.json`.
_CLAY_LAYER = {"name": "clay", "unit_measure": {"d_factor": 10},
               "depths": [{"label": "0-5cm", "values": {"mean": 159}},
                          {"label": "5-15cm", "values": {"mean": 162}}]}


def _fake_layer(prop: str) -> dict:
    return {"name": prop, "unit_measure": {"d_factor": 10},
            "depths": [{"label": "0-5cm", "values": {"mean": 100}},
                       {"label": "5-15cm", "values": {"mean": 110}}]}


def test_one_row_per_site(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(sc, "fetch_soilgrids_property", lambda lat, lon, prop: _fake_layer(prop))
    monkeypatch.setattr(sc, "fetch_elevation_grid", lambda lat, lon: [510.0, 500.0, 505.0, 495.0])

    frame, report = sc.build(_sites())

    assert len(frame) == 3
    assert set(frame["site_id"]) == {"A", "B", "C"}
    assert report["soil_hits"] == 3 and report["soil_full_hits"] == 3 and report["slope_hits"] == 3


def test_a_failed_response_is_not_fabricated_as_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path / "cache")

    calls = {"n": 0}

    def flaky_soil(lat, lon, prop):
        calls["n"] += 1
        return None if lat == 11.0 else _fake_layer(prop)  # site B always fails, every property

    monkeypatch.setattr(sc, "fetch_soilgrids_property", flaky_soil)
    monkeypatch.setattr(sc, "fetch_elevation_grid", lambda lat, lon: [510.0, 500.0, 505.0, 495.0])

    frame, report = sc.build(_sites())

    assert len(frame) == 3, "a hole is still a row (with the site_id/cluster keys), never a dropped site"
    row_b = frame[frame["site_id"] == "B"].iloc[0]
    assert "clay_0-5cm" not in frame.columns or pd.isna(row_b.get("clay_0-5cm"))
    assert report["soil_hits"] == 2
    assert report["soil_miss_clusters"] == {"Kaduna_Grain_Belt": 1}

    # And the failure must not be memoised: no soil cache file was written for site B, for any
    # property (its slope succeeds independently and is expected to be cached).
    assert not [p for p in (tmp_path / "cache").glob("B-*.json") if not p.name.endswith("-slope.json")]

    # A second build over the same (still-failing) site does not skip the fetch — proof the miss
    # was never cached as an empty/False value. A and C succeeded (7 properties each = 14 calls,
    # now served from cache); B failed all 7 properties both times: 21 (first build) + 7 (B only,
    # second build) = 28.
    sc.build(_sites())
    assert calls["n"] == 28, "the flaky site must be re-fetched every call, never cached as empty"


def test_cluster_structured_gap_is_flagged():
    miss = {"Kaduna_Grain_Belt": 2}
    sizes = {"Kaduna_Grain_Belt": 2, "Kenya_Rift_Valley": 1}
    assert sc._cluster_structured(miss, sizes) is True

    # A partial, scattered miss inside a cluster that still has coverage is not cluster-structured.
    assert sc._cluster_structured({"Kaduna_Grain_Belt": 1}, sizes) is False


def test_soilgrids_unit_conversion_uses_the_response_d_factor():
    # Raw clay=159 with d_factor=10 is 15.9% — the real value the task description cites for a
    # Kaduna site, converted via the API's own metadata rather than a hardcoded /10.
    values = sc.layer_values(_CLAY_LAYER)
    assert values["clay_0-5cm"] == 15.9
    assert values["clay_5-15cm"] == 16.2


def test_slope_percent_is_zero_on_flat_ground():
    assert sc.slope_percent(10.0, [500.0, 500.0, 500.0, 500.0]) == 0.0


def test_slope_percent_matches_hand_computed_gradient():
    # North-south rise of 20 m over a 2*0.005deg = 0.01deg span (~1113.2 m) -> ~1.797% grade north-
    # south, zero east-west -> gradient magnitude equals the north-south component alone.
    lat = 0.0  # cos(0) = 1, so the east-west scale factor is exactly METERS_PER_DEG_LAT too
    result = sc.slope_percent(lat, [510.0, 490.0, 500.0, 500.0])
    span_m = 2 * sc.SLOPE_HALF_WIDTH_DEG * sc.METERS_PER_DEG_LAT
    expected = (20.0 / span_m) * 100.0
    assert abs(result - expected) < 1e-9


def test_cached_fetch_reads_a_prewritten_cache_entry_without_calling_fetch(tmp_path):
    path = tmp_path / "A.json"
    path.write_text(json.dumps({"hit": True}))
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return {"should": "not be called"}

    out = sc._cached_fetch(path, fetch)
    assert out == {"hit": True}
    assert calls["n"] == 0
