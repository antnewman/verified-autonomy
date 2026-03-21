"""
Tests for Layer 07: Adversarial Testing

Test naming convention: test_[subject]_[scenario]_[expected_result]

Requirements:
- 90% minimum coverage
- Edge cases at both boundaries
- Corner cases at intersections of multiple boundaries
- Failure mode tests (correct exceptions, messages, fallbacks)
- No mocking of the system under test
"""

from __future__ import annotations

from typing import Callable

import pytest

from adversarial_testing import (
    Finding,
    FindingSeverity,
    PolicyRule,
    RedTeamHarness,
    RedTeamReport,
    TestCase,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _rule(
    name: str = "test_rule",
    patterns: tuple[str, ...] = ("forbidden",),
    severity: FindingSeverity = FindingSeverity.HIGH,
    description: str = "Test rule",
) -> PolicyRule:
    return PolicyRule(
        name=name,
        description=description,
        forbidden_patterns=patterns,
        severity=severity,
    )


def _test_case(
    test_case_id: str = "tc-001",
    prompt: str = "hello",
    category: str = "baseline",
    description: str = "A test",
    target_rules: tuple[str, ...] = (),
) -> TestCase:
    return TestCase(
        test_case_id=test_case_id,
        prompt=prompt,
        category=category,
        description=description,
        target_rules=target_rules,
    )


def _echo_sut(prompt: str) -> str:
    """SUT that echoes the prompt — triggers rules based on prompt content."""
    return prompt


def _clean_sut(prompt: str) -> str:
    """SUT that always returns a clean response."""
    return "This is a safe and approved response."


def _failing_sut(prompt: str) -> str:
    """SUT that always returns text containing forbidden patterns."""
    return "forbidden content here"


def _raising_sut(prompt: str) -> str:
    """SUT that always raises an exception."""
    raise RuntimeError("SUT internal error")


def _empty_sut(prompt: str) -> str:
    """SUT that always returns an empty string."""
    return ""


def _make_harness(
    sut: Callable[[str], str] = _echo_sut,
    rules: list[PolicyRule] | None = None,
) -> RedTeamHarness:
    if rules is None:
        rules = [_rule()]
    return RedTeamHarness(system_under_test=sut, policy=rules)


# ---------------------------------------------------------------------------
# PolicyRule — frozen dataclass
# ---------------------------------------------------------------------------


class TestPolicyRule:
    def test_is_frozen(self) -> None:
        rule = _rule()
        with pytest.raises(Exception):
            rule.name = "changed"  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        rule = PolicyRule(
            name="my_rule",
            description="My description",
            forbidden_patterns=("bad", "evil"),
            severity=FindingSeverity.CRITICAL,
        )
        assert rule.name == "my_rule"
        assert rule.description == "My description"
        assert rule.forbidden_patterns == ("bad", "evil")
        assert rule.severity == FindingSeverity.CRITICAL

    def test_multiple_patterns_stored(self) -> None:
        rule = _rule(patterns=("alpha", "beta", "gamma"))
        assert len(rule.forbidden_patterns) == 3

    def test_empty_patterns_tuple(self) -> None:
        rule = _rule(patterns=())
        assert rule.forbidden_patterns == ()


# ---------------------------------------------------------------------------
# TestCase — frozen dataclass
# ---------------------------------------------------------------------------


class TestTestCase:
    def test_is_frozen(self) -> None:
        tc = _test_case()
        with pytest.raises(Exception):
            tc.prompt = "changed"  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        tc = TestCase(
            test_case_id="tc-42",
            prompt="some prompt",
            category="injection",
            description="A detailed description",
            target_rules=("rule_a", "rule_b"),
        )
        assert tc.test_case_id == "tc-42"
        assert tc.prompt == "some prompt"
        assert tc.category == "injection"
        assert tc.description == "A detailed description"
        assert tc.target_rules == ("rule_a", "rule_b")

    def test_target_rules_defaults_to_empty_tuple(self) -> None:
        tc = TestCase(
            test_case_id="tc-01",
            prompt="hello",
            category="baseline",
            description="test",
        )
        assert tc.target_rules == ()


# ---------------------------------------------------------------------------
# RedTeamHarness.evaluate_output
# ---------------------------------------------------------------------------


class TestEvaluateOutput:
    def test_detects_violation_when_forbidden_pattern_present(self) -> None:
        harness = _make_harness(rules=[_rule(patterns=("forbidden",))])
        violated = harness.evaluate_output("This contains forbidden content")
        assert len(violated) == 1

    def test_passes_when_no_forbidden_pattern_present(self) -> None:
        harness = _make_harness(rules=[_rule(patterns=("forbidden",))])
        violated = harness.evaluate_output("This is a perfectly safe response")
        assert violated == []

    def test_is_case_insensitive_uppercase_pattern(self) -> None:
        harness = _make_harness(rules=[_rule(patterns=("FORBIDDEN",))])
        violated = harness.evaluate_output("contains forbidden text")
        assert len(violated) == 1

    def test_is_case_insensitive_uppercase_output(self) -> None:
        harness = _make_harness(rules=[_rule(patterns=("forbidden",))])
        violated = harness.evaluate_output("CONTAINS FORBIDDEN TEXT")
        assert len(violated) == 1

    def test_is_case_insensitive_mixed_case(self) -> None:
        harness = _make_harness(rules=[_rule(patterns=("Forbidden",))])
        violated = harness.evaluate_output("FoRbIdDeN")
        assert len(violated) == 1

    def test_target_rules_filters_to_specified_rules_only(self) -> None:
        rule_a = _rule("rule_a", patterns=("bad_a",))
        rule_b = _rule("rule_b", patterns=("bad_b",))
        harness = RedTeamHarness(_echo_sut, [rule_a, rule_b])
        # Output contains bad_a but we only check rule_b
        violated = harness.evaluate_output("bad_a content", target_rules=("rule_b",))
        assert violated == []

    def test_target_rules_detects_when_specified_rule_fires(self) -> None:
        rule_a = _rule("rule_a", patterns=("bad_a",))
        rule_b = _rule("rule_b", patterns=("bad_b",))
        harness = RedTeamHarness(_echo_sut, [rule_a, rule_b])
        violated = harness.evaluate_output("bad_b here", target_rules=("rule_b",))
        assert len(violated) == 1
        assert violated[0].name == "rule_b"

    def test_no_target_rules_checks_all_rules(self) -> None:
        rule_a = _rule("rule_a", patterns=("alpha",))
        rule_b = _rule("rule_b", patterns=("beta",))
        harness = RedTeamHarness(_echo_sut, [rule_a, rule_b])
        violated = harness.evaluate_output("alpha and beta both present")
        assert len(violated) == 2

    def test_target_rules_nonexistent_name_returns_empty(self) -> None:
        harness = _make_harness(rules=[_rule("existing_rule", patterns=("forbidden",))])
        violated = harness.evaluate_output("forbidden", target_rules=("nonexistent",))
        assert violated == []

    def test_multiple_patterns_one_match_fires_rule(self) -> None:
        rule = _rule(patterns=("pattern_a", "pattern_b", "pattern_c"))
        harness = _make_harness(rules=[rule])
        violated = harness.evaluate_output("contains pattern_b only")
        assert len(violated) == 1

    def test_empty_output_passes_all_rules(self) -> None:
        harness = _make_harness(rules=[_rule(patterns=("forbidden",))])
        violated = harness.evaluate_output("")
        assert violated == []

    def test_empty_patterns_rule_never_fires(self) -> None:
        harness = _make_harness(rules=[_rule(patterns=())])
        violated = harness.evaluate_output("any content at all")
        assert violated == []

    def test_empty_policy_returns_empty(self) -> None:
        harness = RedTeamHarness(_echo_sut, policy=[])
        violated = harness.evaluate_output("forbidden content")
        assert violated == []

    def test_returns_list_of_policy_rules(self) -> None:
        harness = _make_harness(rules=[_rule(patterns=("forbidden",))])
        result = harness.evaluate_output("forbidden")
        assert all(isinstance(r, PolicyRule) for r in result)


# ---------------------------------------------------------------------------
# RedTeamHarness.run_test
# ---------------------------------------------------------------------------


class TestRunTest:
    def test_returns_finding_instance(self) -> None:
        harness = _make_harness(sut=_clean_sut)
        finding = harness.run_test(_test_case())
        assert isinstance(finding, Finding)

    def test_finding_has_correct_test_case_id(self) -> None:
        harness = _make_harness(sut=_clean_sut)
        finding = harness.run_test(_test_case(test_case_id="tc-999"))
        assert finding.test_case_id == "tc-999"

    def test_finding_has_correct_category(self) -> None:
        harness = _make_harness(sut=_clean_sut)
        finding = harness.run_test(_test_case(category="injection"))
        assert finding.category == "injection"

    def test_finding_captures_prompt(self) -> None:
        harness = _make_harness(sut=_clean_sut)
        finding = harness.run_test(_test_case(prompt="my test prompt"))
        assert finding.prompt == "my test prompt"

    def test_finding_captures_output_text(self) -> None:
        harness = _make_harness(sut=_clean_sut)
        finding = harness.run_test(_test_case())
        assert "safe" in finding.output.lower()

    def test_violated_rules_correctly_identified_on_failure(self) -> None:
        rule = _rule(patterns=("forbidden",))
        harness = RedTeamHarness(_failing_sut, [rule])
        finding = harness.run_test(_test_case())
        assert len(finding.violated_rules) == 1
        assert finding.violated_rules[0].name == rule.name

    def test_violated_rules_empty_on_clean_output(self) -> None:
        harness = RedTeamHarness(_clean_sut, [_rule(patterns=("forbidden",))])
        finding = harness.run_test(_test_case())
        assert finding.violated_rules == []

    def test_severity_is_highest_among_violated_rules(self) -> None:
        low_rule = _rule("low", patterns=("low_word",), severity=FindingSeverity.LOW)
        crit_rule = _rule("crit", patterns=("crit_word",), severity=FindingSeverity.CRITICAL)
        harness = RedTeamHarness(
            lambda _: "low_word and crit_word both present",
            [low_rule, crit_rule],
        )
        finding = harness.run_test(_test_case())
        assert finding.severity == FindingSeverity.CRITICAL

    def test_no_violation_produces_severity_none(self) -> None:
        harness = RedTeamHarness(_clean_sut, [_rule(patterns=("forbidden",))])
        finding = harness.run_test(_test_case())
        assert finding.severity == FindingSeverity.NONE

    def test_raising_sut_captured_as_error(self) -> None:
        harness = _make_harness(sut=_raising_sut)
        finding = harness.run_test(_test_case())
        assert finding.error is not None
        assert "RuntimeError" in finding.error

    def test_raising_sut_output_is_empty_string(self) -> None:
        harness = _make_harness(sut=_raising_sut)
        finding = harness.run_test(_test_case())
        assert finding.output == ""

    def test_raising_sut_violated_rules_is_empty(self) -> None:
        harness = _make_harness(sut=_raising_sut)
        finding = harness.run_test(_test_case())
        assert finding.violated_rules == []

    def test_empty_sut_output_passes_all_rules(self) -> None:
        harness = RedTeamHarness(_empty_sut, [_rule(patterns=("forbidden",))])
        finding = harness.run_test(_test_case())
        assert finding.severity == FindingSeverity.NONE
        assert finding.output == ""

    def test_error_field_is_none_on_clean_run(self) -> None:
        harness = _make_harness(sut=_clean_sut)
        finding = harness.run_test(_test_case())
        assert finding.error is None

    def test_target_rules_in_test_case_are_respected(self) -> None:
        rule_a = _rule("rule_a", patterns=("forbidden",))
        rule_b = _rule("rule_b", patterns=("forbidden",))
        harness = RedTeamHarness(_failing_sut, [rule_a, rule_b])
        # Only target rule_a — rule_b should not appear in violations
        tc = _test_case(target_rules=("rule_a",))
        finding = harness.run_test(tc)
        violated_names = {r.name for r in finding.violated_rules}
        assert violated_names == {"rule_a"}

    def test_nonexistent_target_rule_gives_no_violations(self) -> None:
        harness = RedTeamHarness(_failing_sut, [_rule("existing", patterns=("forbidden",))])
        tc = _test_case(target_rules=("nonexistent",))
        finding = harness.run_test(tc)
        assert finding.violated_rules == []
        assert finding.severity == FindingSeverity.NONE


# ---------------------------------------------------------------------------
# RedTeamHarness.run_suite
# ---------------------------------------------------------------------------


class TestRunSuite:
    def _two_rule_policy(self) -> list[PolicyRule]:
        return [
            _rule("rule_pass", patterns=("xyzzy_not_present",)),
            _rule("rule_fail", patterns=("forbidden",)),
        ]

    def test_returns_red_team_report(self) -> None:
        harness = _make_harness(sut=_clean_sut)
        report = harness.run_suite([_test_case()])
        assert isinstance(report, RedTeamReport)

    def test_total_tests_matches_input_count(self) -> None:
        harness = _make_harness(sut=_clean_sut)
        cases = [_test_case(f"tc-{i}") for i in range(5)]
        report = harness.run_suite(cases)
        assert report.total_tests == 5

    def test_findings_contains_only_violated_tests(self) -> None:
        harness = RedTeamHarness(_failing_sut, self._two_rule_policy())
        cases = [_test_case("tc-1"), _test_case("tc-2")]
        report = harness.run_suite(cases)
        # All fail because _failing_sut returns "forbidden content"
        assert len(report.findings) == 2
        assert all(len(f.violated_rules) > 0 for f in report.findings)

    def test_findings_excludes_clean_tests(self) -> None:
        harness = RedTeamHarness(_clean_sut, [_rule(patterns=("forbidden",))])
        cases = [_test_case(f"tc-{i}") for i in range(4)]
        report = harness.run_suite(cases)
        assert report.findings == []

    def test_pass_rate_all_passing(self) -> None:
        harness = _make_harness(sut=_clean_sut)
        cases = [_test_case(f"tc-{i}") for i in range(4)]
        report = harness.run_suite(cases)
        assert report.pass_rate == 1.0

    def test_pass_rate_all_failing(self) -> None:
        harness = RedTeamHarness(_failing_sut, [_rule(patterns=("forbidden",))])
        cases = [_test_case(f"tc-{i}") for i in range(4)]
        report = harness.run_suite(cases)
        assert report.pass_rate == 0.0

    def test_pass_rate_mixed(self) -> None:
        # First two cases have prompt "forbidden" → echo → fail
        # Last two have clean prompt → pass
        harness = RedTeamHarness(_echo_sut, [_rule(patterns=("forbidden",))])
        cases = [
            _test_case("tc-1", prompt="forbidden"),
            _test_case("tc-2", prompt="forbidden"),
            _test_case("tc-3", prompt="clean text"),
            _test_case("tc-4", prompt="clean text"),
        ]
        report = harness.run_suite(cases)
        assert report.pass_rate == 0.5

    def test_summary_by_severity_counts_are_correct(self) -> None:
        low_rule = _rule("low", patterns=("low_word",), severity=FindingSeverity.LOW)
        high_rule = _rule("high", patterns=("high_word",), severity=FindingSeverity.HIGH)
        harness = RedTeamHarness(_echo_sut, [low_rule, high_rule])
        cases = [
            _test_case("tc-1", prompt="low_word"),   # LOW finding
            _test_case("tc-2", prompt="high_word"),  # HIGH finding
            _test_case("tc-3", prompt="clean"),      # no finding
        ]
        report = harness.run_suite(cases)
        assert report.summary_by_severity[FindingSeverity.LOW] == 1
        assert report.summary_by_severity[FindingSeverity.HIGH] == 1
        assert report.summary_by_severity[FindingSeverity.NONE] == 0

    def test_summary_by_category_counts_are_correct(self) -> None:
        harness = RedTeamHarness(_failing_sut, [_rule(patterns=("forbidden",))])
        cases = [
            _test_case("tc-1", category="injection"),
            _test_case("tc-2", category="injection"),
            _test_case("tc-3", category="extraction"),
        ]
        report = harness.run_suite(cases)
        assert report.summary_by_category["injection"] == 2
        assert report.summary_by_category["extraction"] == 1

    def test_summary_by_severity_all_severities_present(self) -> None:
        """summary_by_severity contains a key for every FindingSeverity."""
        harness = _make_harness(sut=_clean_sut)
        report = harness.run_suite([_test_case()])
        for severity in FindingSeverity:
            assert severity in report.summary_by_severity

    def test_report_text_is_non_empty_string(self) -> None:
        harness = _make_harness(sut=_clean_sut)
        report = harness.run_suite([_test_case()])
        assert isinstance(report.report_text, str)
        assert len(report.report_text) > 0

    def test_report_text_contains_pass_rate(self) -> None:
        harness = _make_harness(sut=_clean_sut)
        report = harness.run_suite([_test_case()])
        assert "Pass rate" in report.report_text or "pass rate" in report.report_text.lower()

    def test_report_text_contains_total_tests(self) -> None:
        harness = _make_harness(sut=_clean_sut)
        cases = [_test_case(f"tc-{i}") for i in range(3)]
        report = harness.run_suite(cases)
        assert "3" in report.report_text

    def test_empty_test_suite_produces_clean_report(self) -> None:
        harness = _make_harness(sut=_clean_sut)
        report = harness.run_suite([])
        assert report.total_tests == 0
        assert report.findings == []
        assert report.pass_rate == 1.0
        assert isinstance(report.report_text, str)

    def test_raising_sut_included_in_findings(self) -> None:
        harness = _make_harness(sut=_raising_sut)
        cases = [_test_case()]
        report = harness.run_suite(cases)
        assert len(report.findings) == 1
        assert report.findings[0].error is not None

    def test_summary_by_category_empty_for_clean_suite(self) -> None:
        harness = _make_harness(sut=_clean_sut)
        report = harness.run_suite([_test_case()])
        assert report.summary_by_category == {}

    def test_single_finding_severity_tracked(self) -> None:
        harness = RedTeamHarness(_failing_sut, [_rule(patterns=("forbidden",), severity=FindingSeverity.MEDIUM)])
        report = harness.run_suite([_test_case()])
        assert report.summary_by_severity[FindingSeverity.MEDIUM] == 1


# ---------------------------------------------------------------------------
# FindingSeverity ordering
# ---------------------------------------------------------------------------


class TestFindingSeverityOrdering:
    def test_severity_values(self) -> None:
        assert FindingSeverity.NONE.value == "none"
        assert FindingSeverity.LOW.value == "low"
        assert FindingSeverity.MEDIUM.value == "medium"
        assert FindingSeverity.HIGH.value == "high"
        assert FindingSeverity.CRITICAL.value == "critical"

    def test_highest_severity_among_multiple_rules(self) -> None:
        rules = [
            _rule("a", patterns=("alpha",), severity=FindingSeverity.LOW),
            _rule("b", patterns=("beta",), severity=FindingSeverity.HIGH),
            _rule("c", patterns=("gamma",), severity=FindingSeverity.MEDIUM),
        ]
        harness = RedTeamHarness(
            lambda _: "alpha beta gamma",
            rules,
        )
        finding = harness.run_test(_test_case())
        assert finding.severity == FindingSeverity.HIGH

    def test_single_rule_severity_propagates(self) -> None:
        harness = RedTeamHarness(
            _failing_sut,
            [_rule(patterns=("forbidden",), severity=FindingSeverity.CRITICAL)],
        )
        finding = harness.run_test(_test_case())
        assert finding.severity == FindingSeverity.CRITICAL


# ---------------------------------------------------------------------------
# main() smoke tests
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_runs_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        from adversarial_testing import main
        main()

    def test_main_output_contains_adversarial_testing(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from adversarial_testing import main
        main()
        assert "Adversarial Testing" in capsys.readouterr().out

    def test_main_output_contains_severity_levels(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from adversarial_testing import main
        main()
        out = capsys.readouterr().out
        # At least one finding severity should appear
        assert any(
            sev in out.upper()
            for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "FAIL")
        )

    def test_main_output_contains_pass_rate(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from adversarial_testing import main
        main()
        assert "Pass rate" in capsys.readouterr().out

    def test_main_output_contains_pass(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from adversarial_testing import main
        main()
        assert "PASS" in capsys.readouterr().out
