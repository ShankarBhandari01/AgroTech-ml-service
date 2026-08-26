"""An experiment is a config file with a run id, not a flag on a 684-line script.

RESEARCH_SUMMARY records a metrics table cited by the README and two source files that exists in no
committed file. These pin the mechanism that makes that impossible: every result carries the data
hash, the git SHA and the seed that produced it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argotech.lab.run import load_config, run_experiment


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for c in range(3):
        for k in range(6):
            for j in range(12):
                level = (k - 2.5) * 0.5
                obs = pd.Timestamp(f"2025-{j + 1:02d}-01")
                rows.append({"site_id": f"C{c}-S{k}", "cluster": f"C{c}",
                             "obs_date": obs.strftime("%Y-%m-%d"),
                             # 30-day outcome lag, matching the real panel exactly.
                             "label_date": (obs + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                             "ndvi_z_peer": level + rng.normal(0, 0.3),
                             "rain_30": rng.normal(50, 10),
                             "forward_z": level + rng.normal(0, 0.3)})
    return pd.DataFrame(rows)


CFG = {"target": "within_y", "features": ["ndvi_z_peer", "rain_30"],
       "arms": ["zero", "persistence", "boosted"], "splits": ["spatial"],
       "min_history": 2, "shrink": 0.0, "seed": 42, "tau": -0.5,
       # "leaky" is the control arm that reproduces pre-fold-fit behaviour: this fixture bakes
       # ndvi_z_peer in directly and carries none of the raw ndvi/rvi/latitude/elevation columns
       # fit_peer_stats needs, exactly like every test above this predates the fold-fitted transform.
       "peer_key": "leaky", "lat_band": 20.0, "elev_band": 1000.0}


def test_a_result_names_the_data_the_seed_and_the_code():
    out = run_experiment(CFG, _panel())
    p = out["provenance"]
    assert len(p["content_hash"]) == 64 and p["seed"] == 42 and p["git_sha"]
    assert isinstance(p["dirty"], bool)
    assert p["scored_rows"] <= p["rows"]


def test_every_arm_is_scored_in_every_fold():
    out = run_experiment(CFG, _panel())
    assert len(out["folds"]) == 3
    for fold in out["folds"]:
        assert set(fold["arms"]) == {"zero", "persistence", "boosted"}
        for scores in fold["arms"].values():
            assert {"net_benefit", "precision_at_25", "spearman"} <= set(scores)


def test_the_summary_carries_an_interval_per_arm():
    out = run_experiment(CFG, _panel())
    assert set(out["summary"]) == set(CFG["splits"])
    for split_name, split_summary in out["summary"].items():
        for name, s in split_summary.items():
            assert "net_benefit_mean" in s and "net_benefit_ci" in s, (split_name, name)
            lo, hi = s["net_benefit_ci"]
            assert lo <= s["net_benefit_mean"] <= hi


def test_the_same_config_and_data_give_the_same_numbers():
    a, b = run_experiment(CFG, _panel()), run_experiment(CFG, _panel())
    assert a["summary"] == b["summary"]
    assert a["provenance"]["content_hash"] == b["provenance"]["content_hash"]


def test_an_unknown_arm_fails_loudly():
    with pytest.raises(ValueError, match="unknown arm"):
        run_experiment({**CFG, "arms": ["zero", "magic"]}, _panel())


def test_the_committed_e02_config_parses():
    cfg = load_config("experiments/E02-within-vs-level.yaml")
    assert cfg["target"] in ("level_z", "within_y", "within_xy", "delta_z")
    assert "zero" in cfg["arms"], "the zero predictor is the baseline E02 exists to beat"


def test_both_split_protocols_produce_folds():
    """E02 requests splits: [spatial, temporal]; nothing here exercised the temporal path.

    `forward_chaining` needs `label_date` to cut the training side on outcome availability. A
    fixture without that column cannot catch a regression in the protocol the real experiment runs.
    """
    out = run_experiment({**CFG, "splits": ["spatial", "temporal"]}, _panel())
    kinds = {f["split"] for f in out["folds"]}
    assert kinds == {"spatial", "temporal"}, f"missing a split protocol: {kinds}"
    for fold in out["folds"]:
        assert set(fold["arms"]) == set(CFG["arms"])


def test_summary_is_grouped_by_protocol_not_pooled_across_them():
    """4 blocked folds and 3 forward-chaining folds measure different things (splits.py); pooling
    them into one mean silently reports a number neither protocol produced. `summary` must report
    each protocol separately, and those separate numbers must not just reproduce a flat pooled
    average over every fold."""
    out = run_experiment({**CFG, "splits": ["spatial", "temporal"]}, _panel())
    assert set(out["summary"]) == {"spatial", "temporal"}
    for split_summary in out["summary"].values():
        assert set(split_summary) == set(CFG["arms"])

    # "persistence" and "boosted" vary across folds (unlike "zero", which predicts 0 everywhere and
    # so trivially agrees with any grouping); on those arms a pooled mean must differ from at least
    # one protocol's own mean, or the split isn't actually separating anything.
    for arm in ("persistence", "boosted"):
        spatial_mean = out["summary"]["spatial"][arm]["net_benefit_mean"]
        temporal_mean = out["summary"]["temporal"][arm]["net_benefit_mean"]
        pooled = float(np.mean([f["arms"][arm]["net_benefit"] for f in out["folds"]]))
        assert spatial_mean != temporal_mean, f"{arm}: the two protocols measured identical folds"
        assert not np.isclose(pooled, spatial_mean) or not np.isclose(pooled, temporal_mean), (
            f"{arm}: pooled mean matches both protocol means — the split is not actually separating "
            "the two validation protocols")


# ---- peer_key wiring: build_target and the peer transform both move inside the fold loop --------

def test_leaky_reproduces_the_pre_change_numbers():
    """`peer_key: leaky` must behave exactly like the pre-fold-fit pipeline: build_target once on
    the whole panel, split after — the control arm the fold-fitted paths are measured against."""
    from argotech.lab.run import ARMS, SPLITS, _score_arm
    from argotech.lab.targets import build_target as _bt

    df = _panel()
    frame, features = _bt(df, CFG["target"], features=CFG["features"],
                           min_history=CFG["min_history"], shrink=CFG["shrink"])
    expected = {name: [] for name in CFG["arms"]}
    for _fold_name, train, test in SPLITS["spatial"](frame):
        for name in CFG["arms"]:
            expected[name].append(
                _score_arm(ARMS[name](CFG["seed"]), train, test, features, CFG["tau"])["net_benefit"])

    out = run_experiment(CFG, df)
    for name in CFG["arms"]:
        got = [f["arms"][name]["net_benefit"] for f in out["folds"] if f["split"] == "spatial"]
        assert got == expected[name], f"{name}: leaky run diverged from the pre-change pipeline"
    assert all(f["peer_coverage"] == 1.0 for f in out["folds"]), "leaky must record coverage 1.0"


def _geo_panel() -> pd.DataFrame:
    """Three clusters that share one lat/elev band (lat 5/8/11 -> all floor to 0 at band 20;
    elevation 100/200/300 -> all floor to 0 at band 1000), so `geo_month` merges them into one
    bucket per month while `cluster_month` keeps every cluster's bucket distinct — exactly what
    lets a held-out cluster borrow a `geo_month` reference but never a `cluster_month` one."""
    rng = np.random.default_rng(1)
    rows = []
    for cluster, lat, elev in (("C0", 5.0, 100.0), ("C1", 8.0, 200.0), ("C2", 11.0, 300.0)):
        for k in range(6):
            level = (k - 2.5) * 0.5
            for j in range(12):
                obs = pd.Timestamp(f"2025-{j + 1:02d}-01")
                rows.append({"site_id": f"{cluster}-S{k}", "cluster": cluster,
                             "latitude": lat, "elevation": elev,
                             "obs_date": obs.strftime("%Y-%m-%d"),
                             "label_date": (obs + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                             "ndvi": 0.5 + level * 0.1 + rng.normal(0, 0.05),
                             "rvi": 0.4 + level * 0.05 + rng.normal(0, 0.05),
                             # A control-arm stand-in for panel.py's baked column, so `leaky` has
                             # something to read without running the (leaky) whole-frame fit here.
                             "ndvi_z_peer": level + rng.normal(0, 0.3),
                             "rain_30": rng.normal(50, 10),
                             "forward_z": level + rng.normal(0, 0.3)})
    return pd.DataFrame(rows)


GEO_CFG = {"target": "within_y", "features": ["ndvi_z_peer", "rain_30"], "arms": ["zero"],
           "splits": ["spatial"], "min_history": 2, "shrink": 0.0, "seed": 42, "tau": -0.5,
           "lat_band": 20.0, "elev_band": 1000.0}


def test_cluster_month_gives_a_held_out_cluster_zero_coverage_every_fold():
    out = run_experiment({**GEO_CFG, "peer_key": "cluster_month"}, _geo_panel())
    covs = [f["peer_coverage"] for f in out["folds"]]
    assert covs and all(c == 0.0 for c in covs), covs


def test_geo_month_reports_coverage_above_zero():
    out = run_experiment({**GEO_CFG, "peer_key": "geo_month"}, _geo_panel())
    covs = [f["peer_coverage"] for f in out["folds"]]
    assert covs and all(c > 0.0 for c in covs), covs


def test_the_three_peer_keys_produce_different_summaries():
    """If leaky, cluster_month and geo_month all produced the same numbers, the fold-fitted
    transform would not actually be wired into the fold loop.

    `zero`'s prediction is a constant 0, so `precision_at_25` falls back to the raw event rate
    (evaluate.precision_at_k) — which is exactly the quantity that moves when `ndvi_z_peer`, and
    therefore `alpha_hat` and the `ztilde <= tau` labels built from it, change with `peer_key`.
    `net_benefit` is not used here: it flags nothing at this threshold for any of the three configs
    and so is 0.0 across the board regardless of whether the transform is wired in."""
    df = _geo_panel()
    p25 = {kind: run_experiment({**GEO_CFG, "peer_key": kind}, df)["summary"]["spatial"]["zero"]
           ["precision_at_25_mean"]
           for kind in ("leaky", "cluster_month", "geo_month")}
    assert len(set(p25.values())) == 3, f"peer_key had no effect on the summary: {p25}"
