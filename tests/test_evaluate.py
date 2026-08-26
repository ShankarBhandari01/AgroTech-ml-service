"""Net benefit, and the arithmetic it has to reproduce.

The incumbent reports macro F1 and precision@25 and lets them disagree, leaving the operating point
to a constant nobody derived (docs/model-design.md section 9.1 D). Net benefit replaces that
argument with a curve, so these tests pin it against the closed form rather than against itself.
"""

from __future__ import annotations

import numpy as np

from argotech.lab.evaluate import (
    bootstrap_ci,
    decision_curve,
    net_benefit,
    precision_at_k,
    prob_event,
    spearman,
)


def test_net_benefit_matches_the_closed_form():
    # 10 cases, 4 events. Flag the 5 highest probabilities: 3 true positives, 2 false positives.
    y = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0])
    prob = np.array([.9, .8, .7, .65, .6, .2, .1, .1, .1, .1])
    t = 0.5
    # NB = TP/n - (FP/n) * (t / (1 - t))
    assert np.isclose(net_benefit(y, prob, t), 3 / 10 - (2 / 10) * (t / (1 - t)))


def test_visit_none_scores_exactly_zero():
    y = np.array([1, 0, 1, 0])
    assert net_benefit(y, np.zeros(4), 0.5) == 0.0


def test_visit_all_matches_the_prevalence_formula():
    y = np.array([1, 1, 0, 0, 0])
    t = 0.25
    curve = decision_curve(y, np.full(5, 0.99), [t])[0]
    assert np.isclose(curve["visit_all"], 0.4 - 0.6 * (t / (1 - t)))
    assert curve["visit_none"] == 0.0


def test_a_perfect_ranker_beats_visit_all_at_every_threshold():
    y = np.array([1] * 5 + [0] * 15)
    prob = np.concatenate([np.full(5, 0.95), np.full(15, 0.02)])
    for row in decision_curve(y, prob, [0.1, 0.3, 0.5, 0.7]):
        assert row["model"] >= row["visit_all"] and row["model"] >= row["visit_none"]


def test_prob_event_is_a_normal_cdf_at_the_threshold():
    # A prediction sitting exactly on tau has probability 0.5 of falling at or below it.
    assert np.allclose(prob_event(np.array([-1.0]), 0.8, tau=-1.0), 0.5)
    # A worse prediction (further below tau) must carry higher probability.
    assert prob_event(np.array([-2.0]), 0.8, -1.0)[0] > prob_event(np.array([0.0]), 0.8, -1.0)[0]


def test_precision_at_k_counts_events_in_the_top_k():
    y = np.array([1, 0, 1, 0, 1])
    score = np.array([.9, .8, .7, .6, .5])   # top 3 contains 2 events
    assert np.isclose(precision_at_k(y, score, 3), 2 / 3)


def test_precision_at_k_clamps_k_to_the_sample():
    y = np.array([1, 0])
    assert np.isclose(precision_at_k(y, np.array([.9, .1]), 25), 0.5)


def test_spearman_is_signed_correctly():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(truth, truth) > 0.99
    assert spearman(-truth, truth) < -0.99


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    vals = [0.40, 0.42, 0.45, 0.39, 0.44]
    lo, hi = bootstrap_ci(vals, seed=0)
    assert lo < np.mean(vals) < hi
    assert bootstrap_ci(vals, seed=0) == bootstrap_ci(vals, seed=0)
