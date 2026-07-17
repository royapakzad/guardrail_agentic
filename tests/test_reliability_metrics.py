"""Tests for reliability_metrics.py (Issue #50). Pure functions, no HTTP/LLM calls."""

import reliability_metrics as rm


def test_flip_rate_unanimous_and_split():
    runs = [
        {"crit_x": "A", "crit_y": "A"},
        {"crit_x": "A", "crit_y": "B"},
        {"crit_x": "A", "crit_y": "A"},
    ]
    stats = {s.criterion: s for s in rm.compute_flip_rates(runs)}
    assert stats["crit_x"].flip_rate == 0.0
    assert stats["crit_x"].modal_verdict == "A"
    assert stats["crit_y"].flip_rate == round(1 / 3, 4)
    assert stats["crit_y"].modal_verdict == "A"


def test_flip_rate_empty_runs_is_safe():
    assert rm.compute_flip_rates([]) == []


def test_flip_rate_sorts_criteria_and_handles_missing_key():
    runs = [
        {"b_crit": "COMPLIANT", "a_crit": "COMPLIANT"},
        {"b_crit": "COMPLIANT"},  # a_crit missing in this run
    ]
    stats = rm.compute_flip_rates(runs)
    assert [s.criterion for s in stats] == ["a_crit", "b_crit"]
    a_stats = stats[0]
    assert a_stats.verdicts == ["COMPLIANT", "MISSING"]
    assert a_stats.flip_rate == 0.5


def test_fleiss_kappa_perfect_agreement():
    # Two items, each unanimous across 4 raters, split evenly between two
    # categories -- textbook perfect agreement, kappa must be exactly 1.0.
    runs = [
        {"item1": "A", "item2": "B"},
        {"item1": "A", "item2": "B"},
        {"item1": "A", "item2": "B"},
        {"item1": "A", "item2": "B"},
    ]
    assert rm.fleiss_kappa(runs) == 1.0


def test_fleiss_kappa_below_chance_agreement():
    # Hand-computed: 3 raters, 2 items, categories A/B.
    #   item1: A, A, A            (unanimous)
    #   item2: A, B, A            (2-1 split)
    # p_A = 5/6, p_B = 1/6 -> P_e_bar = 26/36 = 13/18
    # P_i(item1) = (9-3)/6 = 1.0 ; P_i(item2) = (4+1-3)/6 = 1/3
    # P_bar = (1.0 + 1/3) / 2 = 2/3
    # kappa = (2/3 - 13/18) / (1 - 13/18) = (-1/18) / (5/18) = -0.2
    runs = [
        {"item1": "A", "item2": "A"},
        {"item1": "A", "item2": "B"},
        {"item1": "A", "item2": "A"},
    ]
    assert rm.fleiss_kappa(runs) == -0.2


def test_fleiss_kappa_undefined_cases():
    assert rm.fleiss_kappa([{"c": "A"}]) is None  # only 1 run
    assert rm.fleiss_kappa([]) is None
    # Only one category ever observed -- no variance, kappa undefined.
    runs = [{"c": "A"}, {"c": "A"}, {"c": "A"}]
    assert rm.fleiss_kappa(runs) is None


def test_metrics_are_domain_agnostic():
    """Same functions, totally different criterion/verdict vocabulary
    (humanitarian-style names) -- must behave identically to the
    cybersecurity-style example, since nothing here should be hardcoded
    to a specific policy's criteria or label set."""
    runs = [
        {"AID ORG LEGITIMACY": "COMPLIANT", "DISASTER CLAIM ACCURACY": "COMPLIANT"},
        {"AID ORG LEGITIMACY": "MINOR_ISSUE", "DISASTER CLAIM ACCURACY": "COMPLIANT"},
        {"AID ORG LEGITIMACY": "COMPLIANT", "DISASTER CLAIM ACCURACY": "COMPLIANT"},
    ]
    stats = {s.criterion: s for s in rm.compute_flip_rates(runs)}
    assert stats["AID ORG LEGITIMACY"].flip_rate == round(1 / 3, 4)
    assert stats["DISASTER CLAIM ACCURACY"].flip_rate == 0.0
    kappa = rm.fleiss_kappa(runs)
    assert kappa is not None


def test_build_reliability_report():
    runs = [
        {"c1": "COMPLIANT", "c2": "COMPLIANT"},
        {"c1": "COMPLIANT", "c2": "MINOR_ISSUE"},
    ]
    report = rm.build_reliability_report(runs, label="agentic")
    assert report.label == "agentic"
    assert report.n_runs == 2
    assert len(report.unstable_criteria()) == 1
    assert report.unstable_criteria()[0].criterion == "c2"
    text = report.format_report()
    assert "agentic" in text
    assert "UNSTABLE" in text
