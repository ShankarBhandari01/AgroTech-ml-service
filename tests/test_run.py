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
    """`peer_key: leaky` must behave like the pre-fold-fit pipeline: build_target once, split after.

    Exact for the spatial protocol, verified below. NOT exact for temporal: forward_chaining's
    purge/embargo gap (splits.py) drops a thin sliver of rows from every fold's train-union-test,
    and those rows' own `ndvi_z_peer` — already observed before the boundary, only their LABEL
    isn't settled yet — was part of a site's history the pre-change pipeline's single whole-df
    `build_target` call saw, and this fold-scoped one (deliberately: see `_score_fold`) does not.
    That is measured and small (see the bound below), not hidden. What this test actually needs to
    hold regardless — the fix for the review's critical finding — is that a site's cross-boundary
    history WITHIN a fold's own train+test union is no longer truncated; the previous version of
    this test only exercised the spatial protocol, where every site sits wholly on one side of a
    LOCO fold, so it could not have caught that truncation.
    """
    from argotech.lab.run import ARMS, SPLITS, _score_arm
    from argotech.lab.targets import build_target as _bt

    df = _panel()
    cfg = {**CFG, "splits": ["spatial", "temporal"]}
    frame, features = _bt(df, cfg["target"], features=cfg["features"],
                           min_history=cfg["min_history"], shrink=cfg["shrink"])
    expected = {split_name: {name: [] for name in cfg["arms"]} for split_name in cfg["splits"]}
    for split_name in cfg["splits"]:
        for _fold_name, train, test in SPLITS[split_name](frame):
            for name in cfg["arms"]:
                expected[split_name][name].append(
                    _score_arm(ARMS[name](cfg["seed"]), train, test, features,
                               cfg["tau"])["net_benefit"])

    out = run_experiment(cfg, df)
    for name in cfg["arms"]:
        got_spatial = [f["arms"][name]["net_benefit"] for f in out["folds"] if f["split"] == "spatial"]
        assert got_spatial == expected["spatial"][name], \
            f"{name}: spatial leaky run diverged from the pre-change pipeline"

        got_temporal = [f["arms"][name]["net_benefit"]
                         for f in out["folds"] if f["split"] == "temporal"]
        exp_temporal = expected["temporal"][name]
        assert len(got_temporal) == len(exp_temporal)
        drift = max(abs(a - b) for a, b in zip(got_temporal, exp_temporal, strict=True))
        # Measured on this fixture (fixed seed, deterministic): zero 0.0, boosted 0.0046,
        # persistence 0.0192 — the largest of the three. 0.03 was ~1.9x that and could never
        # actually fail; 0.025 keeps ~30% headroom above the real purge-gap drift while still
        # catching a regression that meaningfully changes it.
        assert drift < 0.025, (
            f"{name}: temporal leaky run diverged by more than the purge gap explains "
            f"(drift={drift:.4f}): {got_temporal} vs {exp_temporal}")
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
    and so is 0.0 across the board regardless of whether the transform is wired in.

    `within_y`'s target is defined in terms of `ndvi_z_peer` (via `alpha_hat`), so under
    `cluster_month` — where a held-out cluster gets zero peer coverage — `alpha_hat` is honestly
    NaN everywhere (targets.alpha_hat's fix: no fabricated 0.0) and every spatial fold is skipped.
    That absence of a scored summary is itself evidence the transform is wired in; only leaky and
    geo_month, which both give this fixture's held-out cluster a real reference, are compared by
    precision_at_25."""
    df = _geo_panel()
    results = {kind: run_experiment({**GEO_CFG, "peer_key": kind}, df)
               for kind in ("leaky", "cluster_month", "geo_month")}

    assert "zero" not in results["cluster_month"]["summary"]["spatial"], \
        "cluster_month must skip every fold under within_y, not score one"
    assert all(f.get("skipped") for f in results["cluster_month"]["folds"])

    p25 = {kind: results[kind]["summary"]["spatial"]["zero"]["precision_at_25_mean"]
           for kind in ("leaky", "geo_month")}
    assert p25["leaky"] != p25["geo_month"], f"peer_key had no effect on the summary: {p25}"


# ---- Review findings: alpha_hat's cross-boundary history, and within_xy's shared feature list ---

def test_alpha_hat_sees_a_sites_full_history_across_the_temporal_boundary():
    """Regression for the review's critical finding: build_target must not be called separately on
    train and test. A post-boundary test row's alpha_hat has to see its own pre-boundary (train-
    side) history too — that history is genuinely available at prediction time, and truncating it
    is information loss, not a leak fix."""
    from argotech.lab.run import _score_fold

    panel = pd.DataFrame(
        [{"site_id": "S0", "cluster": "C0", "obs_date": f"2025-01-{i + 1:02d}",
          "ndvi_z_peer": float(i), "rain_30": 1.0, "forward_z": 0.0}
         for i in range(6)]
        + [{"site_id": "S0", "cluster": "C0", "obs_date": f"2025-02-{i + 1:02d}",
            "ndvi_z_peer": v, "rain_30": 1.0, "forward_z": 0.0}
           for i, v in enumerate((6.0, 7.0))])
    train_raw = panel[panel["obs_date"] < "2025-02-01"]
    test_raw = panel[panel["obs_date"] >= "2025-02-01"]

    cfg = {"target": "within_y", "features": ["ndvi_z_peer", "rain_30"],
           "min_history": 1, "shrink": 0.0, "peer_key": "leaky"}
    _, test_t, _, _ = _score_fold(cfg, panel, train_raw, test_raw)

    # Row 1's prior history is all 6 train rows (mean 2.5); row 2's is those 6 plus row 1 (mean 3.0).
    # Truncated to test-side-only, row 1 would have NO prior observation (NaN, dropped) and row 2
    # would average just the one prior test row (6.0), not 2.5 and 3.0.
    assert list(test_t["alpha_hat"]) == [2.5, 3.0], (
        "a test row's alpha_hat must be the expanding mean over ALL strictly-prior observations of "
        f"its own site, train-side included: got {list(test_t['alpha_hat'])}")


def test_embargoed_rows_feed_a_later_test_rows_alpha_hat():
    """forward_chaining (splits.py) rightly excludes an embargoed row -- obs_date < boundary <=
    label_date -- from both train and test: its LABEL isn't knowable yet. But its FEATURE
    (`ndvi_z_peer`) is: the row was observed before the boundary. `alpha_hat` is built from that
    feature, never from the label, so a later test row of the same site must still see it as part
    of its strictly-prior history. `_score_fold` must therefore build the target over the whole
    panel (train + test + embargo gap), not just the train/test union."""
    from argotech.lab.run import _score_fold

    panel = pd.DataFrame([
        {"site_id": "S0", "cluster": "C0", "obs_date": "2025-01-01", "label_date": "2025-01-05",
         "ndvi_z_peer": 1.0, "rain_30": 1.0, "forward_z": 0.0},   # train: label_date < boundary
        {"site_id": "S0", "cluster": "C0", "obs_date": "2025-01-10", "label_date": "2025-02-05",
         "ndvi_z_peer": 5.0, "rain_30": 1.0, "forward_z": 0.0},   # embargoed: neither cut fires
        {"site_id": "S0", "cluster": "C0", "obs_date": "2025-02-01", "label_date": "2025-02-10",
         "ndvi_z_peer": 9.0, "rain_30": 1.0, "forward_z": 0.0},   # test: obs_date >= boundary
    ])
    boundary = "2025-01-15"
    train_raw = panel[panel["label_date"] < boundary]
    test_raw = panel[panel["obs_date"] >= boundary]
    assert len(train_raw) == 1 and len(test_raw) == 1 and len(panel) == 3, \
        "fixture must have exactly one embargoed row, in neither train nor test"

    cfg = {"target": "within_y", "features": ["ndvi_z_peer", "rain_30"],
           "min_history": 1, "shrink": 0.0, "peer_key": "leaky"}
    _, test_t, _, _ = _score_fold(cfg, panel, train_raw, test_raw)

    # The test row's only prior observations of S0 are the train row (1.0) and the embargoed row
    # (5.0) -> expanding mean 3.0. Dropping the embargoed row (the current defect) would give 1.0.
    assert list(test_t["alpha_hat"]) == [3.0], (
        "the test row's alpha_hat must include the embargoed row's ndvi_z_peer (a feature, known "
        f"at prediction time), got {list(test_t['alpha_hat'])}")


def test_within_xy_spatial_run_survives_a_column_degenerate_only_in_the_held_out_cluster():
    """Regression for BUG-1: `heat_stress_days`-shaped defect — a feature identically constant in
    exactly the held-out cluster, varying everywhere else. Must not raise, and every fold's train
    and test frames must end up with the same feature list (guaranteed here by construction: both
    are slices of one build_target call, so there is no separate-decision path left to disagree)."""
    rows = []
    for cluster in ("A", "B", "C"):
        for k in range(4):
            for j in range(6):
                rows.append({"site_id": f"{cluster}{k}", "cluster": cluster,
                             "obs_date": f"2025-{j + 1:02d}-01",
                             "ndvi_z_peer": (k - 1.5) + 0.1 * j,
                             "forward_z": (k - 1.5) + 0.1 * j,
                             # constant every row in cluster A only; a real, varying signal elsewhere
                             "quirky": 0.0 if cluster == "A" else float(j)})
    df = pd.DataFrame(rows)
    # "zero" alone made this test vacuous: Zero._predict never touches test[features], so it could
    # not observe a train/test feature-name mismatch. "linear" runs Sklearn.predict, which indexes
    # test[features] directly and raises KeyError the moment train and test disagree on columns.
    cfg = {"target": "within_xy", "features": ["ndvi_z_peer", "quirky"], "arms": ["zero", "linear"],
           "splits": ["spatial"], "min_history": 2, "shrink": 0.0, "seed": 42, "tau": -0.5,
           "peer_key": "leaky", "lat_band": 20.0, "elev_band": 1000.0}

    out = run_experiment(cfg, df)  # must not raise KeyError
    assert len(out["folds"]) == 3
    for fold in out["folds"]:
        assert fold["arms"], f"fold {fold['fold']} was unexpectedly skipped"


def _mixed_geo_panel() -> pd.DataFrame:
    """Four clusters at lat_band=5.0: C0 (lat 0) and C2 (lat 1) floor into the SAME geo bucket, so
    each can borrow the other's reference when held out; C1 (lat 100) and C3 (lat 200) each sit
    alone in their own bucket and get none. Held out one at a time (spatial LOCO), this yields
    exactly the partial-skip shape Finding 3 measured on real data: 2 of 4 clusters scored, 2
    skipped — never all-or-nothing the way `_geo_panel` (one shared bucket) or `cluster_month`
    (every cluster always alone) are."""
    rng = np.random.default_rng(2)
    rows = []
    for cluster, lat in (("C0", 0.0), ("C1", 100.0), ("C2", 1.0), ("C3", 200.0)):
        for k in range(6):
            level = (k - 2.5) * 0.5
            for j in range(12):
                obs = pd.Timestamp(f"2025-{j + 1:02d}-01")
                rows.append({"site_id": f"{cluster}-S{k}", "cluster": cluster,
                             "latitude": lat, "elevation": 0.0,
                             "obs_date": obs.strftime("%Y-%m-%d"),
                             "label_date": (obs + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                             "ndvi": 0.5 + level * 0.1 + rng.normal(0, 0.05),
                             "rvi": 0.4 + level * 0.05 + rng.normal(0, 0.05),
                             "ndvi_z_peer": level + rng.normal(0, 0.3),
                             "rain_30": rng.normal(50, 10),
                             "forward_z": level + rng.normal(0, 0.3)})
    return pd.DataFrame(rows)


def test_a_partially_skipped_split_reports_the_fold_count_not_just_the_average():
    """Regression for Finding 3: a run that loses SOME folds must not silently average the
    survivors and report as if nothing were lost. C1 and C3 (isolated geo buckets) must be
    skipped; C0 and C2 (shared bucket) must be scored — and every arm's summary entry must carry
    how many of the 4 attempted folds actually scored, plus which ones were dropped and why."""
    cfg = {"target": "delta_z", "features": ["ndvi_z_peer", "rain_30"], "arms": ["zero", "linear"],
           "splits": ["spatial"], "min_history": 2, "shrink": 0.0, "seed": 42, "tau": -0.5,
           "peer_key": "geo_month", "lat_band": 5.0, "elev_band": 500.0}
    out = run_experiment(cfg, _mixed_geo_panel())

    scored = {f["fold"] for f in out["folds"] if f["arms"]}
    skipped = {f["fold"] for f in out["folds"] if f.get("skipped")}
    assert scored == {"C0", "C2"}, scored
    assert skipped == {"C1", "C3"}, skipped

    for arm in cfg["arms"]:
        s = out["summary"]["spatial"][arm]
        assert s["folds_scored"] == 2, s
        assert s["folds_attempted"] == 4, s
        assert {f["fold"] for f in s["skipped_folds"]} == {"C1", "C3"}
        assert all(f["reason"] for f in s["skipped_folds"]), \
            "each skipped fold must carry why, not just that it was dropped"


def test_delta_z_under_cluster_month_skips_rather_than_crashes(capsys):
    """Regression for BUG-2: under cluster_month a held-out cluster gets no peer reference at all
    (peer_coverage 0.0), so delta_z's ndvi_z_peer-derived ztilde is entirely NaN and build_target
    drops every row. That is a real finding (delta_z is undefined for an unseen region under an
    honest cluster-keyed reference), not a bug to paper over with a fabricated value — the fold
    must be skipped and named, not handed to sklearn as zero rows."""
    out = run_experiment({**GEO_CFG, "target": "delta_z", "peer_key": "cluster_month"}, _geo_panel())
    assert out["folds"], "no folds ran at all"
    for fold in out["folds"]:
        assert fold.get("skipped"), f"fold {fold['fold']} should have been skipped, was scored"
        assert not fold["arms"]
    err = capsys.readouterr().err
    assert "delta_z" in err and "cluster_month" in err, "the skip must name the target and peer_key"
