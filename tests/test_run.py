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
                rows.append({"site_id": f"C{c}-S{k}", "cluster": f"C{c}",
                             "obs_date": f"2025-{j + 1:02d}-01",
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


def test_every_arm_is_scored_in_every_fold():
    out = run_experiment(CFG, _panel())
    assert len(out["folds"]) == 3
    for fold in out["folds"]:
        assert set(fold["arms"]) == {"zero", "persistence", "boosted"}
        for scores in fold["arms"].values():
            assert {"net_benefit", "precision_at_25", "spearman"} <= set(scores)


def test_the_summary_carries_an_interval_per_arm():
    out = run_experiment(CFG, _panel())
    for name, s in out["summary"].items():
        assert "net_benefit_mean" in s and "net_benefit_ci" in s, name
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
