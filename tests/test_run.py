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
       "min_history": 2, "shrink": 0.0, "seed": 42, "tau": -0.5}


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
