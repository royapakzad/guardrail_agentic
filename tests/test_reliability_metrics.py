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


def test_collapse_runs_full_is_identity():
    runs = [{"c1": "COMPLIANT"}, {"c1": "MAJOR_ISSUE"}, {"c1": "CRITICAL"}]
    assert rm.collapse_runs(runs, "full") == runs


def test_collapse_runs_severity_tier_merges_major_and_critical_only():
    runs = [{"c1": "COMPLIANT"}, {"c1": "MINOR_ISSUE"}, {"c1": "MAJOR_ISSUE"}, {"c1": "CRITICAL"}]
    collapsed = rm.collapse_runs(runs, "severity_tier")
    assert collapsed == [{"c1": "COMPLIANT"}, {"c1": "MINOR"}, {"c1": "VIOLATION"}, {"c1": "VIOLATION"}]


def test_collapse_runs_harm_based():
    runs = [{"c1": "COMPLIANT"}, {"c1": "MINOR_ISSUE"}, {"c1": "MAJOR_ISSUE"}, {"c1": "CRITICAL"}]
    collapsed = rm.collapse_runs(runs, "harm_based")
    assert collapsed == [{"c1": "COMPLIANT"}, {"c1": "COMPLIANT"}, {"c1": "VIOLATION"}, {"c1": "VIOLATION"}]


def test_collapse_runs_unknown_scheme_raises():
    import pytest

    with pytest.raises(ValueError):
        rm.collapse_runs([{"c1": "COMPLIANT"}], "not-a-real-scheme")


def test_compare_collapse_schemes_matches_empirical_finding():
    """Regression test for Issue #54's core finding: every observed flip so
    far was COMPLIANT <-> MINOR_ISSUE. severity_tier (merges only MAJOR_ISSUE/
    CRITICAL) should NOT remove that instability, since it never touches the
    COMPLIANT/MINOR_ISSUE boundary. harm_based (merges COMPLIANT/MINOR_ISSUE)
    SHOULD remove it, because it merges away the exact boundary that was
    flip-flopping."""
    runs = [
        {"c1": "COMPLIANT", "c2": "COMPLIANT"},
        {"c1": "MINOR_ISSUE", "c2": "COMPLIANT"},
        {"c1": "COMPLIANT", "c2": "COMPLIANT"},
    ]
    comparison = rm.compare_collapse_schemes(runs, label="test")
    assert comparison["full"]["n_unstable_criteria"] == 1
    assert comparison["severity_tier"]["n_unstable_criteria"] == 1  # unchanged -- flip wasn't in MAJOR/CRITICAL
    assert comparison["harm_based"]["n_unstable_criteria"] == 0  # eliminated by merging the flipping boundary


def test_compare_tool_call_outputs_detects_identical_and_changed_evidence():
    runs_tool_logs = [
        [
            {"tool": "urlscan_check", "input": {"url": "https://x.example"}, "output_preview": "A"},
            {"tool": "aid_org_verify", "input": {"org_name": "Red Cross"}, "output_preview": "verified"},
        ],
        [
            {"tool": "urlscan_check", "input": {"url": "https://x.example"}, "output_preview": "A"},
            {"tool": "aid_org_verify", "input": {"org_name": "Red Cross"}, "output_preview": "verified"},
        ],
        [
            {"tool": "urlscan_check", "input": {"url": "https://x.example"}, "output_preview": "B"},
            # aid_org_verify not called this run -- only 2 calls total, still >= 2, still compared
        ],
    ]
    report = rm.compare_tool_call_outputs(runs_tool_logs)
    by_tool = {e["tool"]: e for e in report}

    assert by_tool["urlscan_check"]["n_calls"] == 3
    assert by_tool["urlscan_check"]["identical_output"] is False
    assert by_tool["urlscan_check"]["distinct_outputs"] == 2

    assert by_tool["aid_org_verify"]["n_calls"] == 2
    assert by_tool["aid_org_verify"]["identical_output"] is True
    assert by_tool["aid_org_verify"]["distinct_outputs"] == 1


def test_compare_tool_call_outputs_excludes_single_calls():
    runs_tool_logs = [
        [{"tool": "search_web", "input": {"query": "only once"}, "output_preview": "x"}],
        [],
    ]
    assert rm.compare_tool_call_outputs(runs_tool_logs) == []


def test_format_evidence_reproducibility_report_flags_changed_evidence():
    entries = [
        {"tool": "urlscan_check", "input": {"url": "https://x"}, "n_calls": 3, "identical_output": False, "distinct_outputs": 2}
    ]
    text = rm.format_evidence_reproducibility_report(entries)
    assert "EVIDENCE ITSELF CHANGED" in text

    assert "nothing to compare" in rm.format_evidence_reproducibility_report([])
