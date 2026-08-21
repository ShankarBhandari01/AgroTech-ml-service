"""Training and serving must compute the SAME peer anomaly for the same input.

`tests/test_cluster_relative.py` pins exactly this property for the `_cz` twins, and those never
drifted. `ndvi_z_peer` had no such test and drifted into two definitions — a concurrent cross-site
cohort in training, the site's own 12-month history at serving, r = 0.626 with 25.5% sign flips.
"""

from __future__ import annotations

import math

from argotech.features.agronomic import (
    peer_bucket,
    peer_reference,
    radar_block,
    satellite_block,
)

STATS = {"Kaduna_Grain_Belt|09": {"ndvi": [0.50, 0.10], "rvi": [0.60, 0.20]}}
OBS = {"ndvi": 0.32, "ndwi": 0.10, "evi": 0.25, "sensing_date": "2026-09-14"}
SAR = {"vv": 0.20, "vh": 0.05, "rvi": 0.40}


def _blocks(cluster: str | None):
    bucket = peer_bucket(cluster, OBS["sensing_date"])
    sat = satellite_block(OBS, [0.4, 0.5, 0.6], peer_reference(STATS, bucket, "ndvi"))
    rad = radar_block(SAR, peer_reference(STATS, bucket, "rvi"))
    return sat, rad


def test_the_anomaly_is_the_standardisation_the_snapshot_describes():
    """(0.32 - 0.50) / 0.10 = -1.8, and (0.40 - 0.60) / 0.20 = -1.0. No other convention."""
    sat, rad = _blocks("Kaduna_Grain_Belt")
    assert abs(sat["ndvi_z_peer"] - (-1.8)) < 1e-9
    assert abs(rad["rvi_z_peer"] - (-1.0)) < 1e-9


def test_both_callers_agree_because_neither_computes_the_reference():
    """The training and serving call sites differ only in where `stats` came from. Passing the same
    reference must give the same number — the property that makes one column mean one thing."""
    training_side, _ = _blocks("Kaduna_Grain_Belt")
    serving_side, _ = _blocks("Kaduna_Grain_Belt")
    assert training_side["ndvi_z_peer"] == serving_side["ndvi_z_peer"]


def test_an_unmeasured_region_yields_nan_not_zero():
    """0.0 would assert 'exactly average for its district' about a district never measured."""
    sat, rad = _blocks(None)
    assert math.isnan(sat["ndvi_z_peer"])
    assert math.isnan(rad["rvi_z_peer"])
