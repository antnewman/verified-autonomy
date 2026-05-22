"""
Tests for Layer 05: Deterministic Guardrails

Test naming convention: test_[subject]_[scenario]_[expected_result]

Requirements:
- 90% minimum coverage
- Edge cases at both boundaries
- Corner cases at intersections of multiple boundaries
- Failure mode tests (correct exceptions, messages, fallbacks)
- No mocking of the system under test
"""

from __future__ import annotations

import pytest

from deterministic_guardrails import (
    GuardrailResult,
    Rule,
    RuleResult,
    Severity,
    Verdict,
    build_allowed_values_rule,
    build_range_rule,
    build_required_field_rule,
    evaluate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block_rule(name: str = "test_block", violated: bool = False) -> Rule:
    return Rule(
        name=name,
        description=f"Block rule: {name}",
        severity=Severity.BLOCK,
        condition=lambda _: violated,
    )


def _warn_rule(name: str = "test_warn", violated: bool = False) -> Rule:
    return Rule(
        name=name,
        description=f"Warn rule: {name}",
        severity=Severity.WARN,
        condition=lambda _: violated,
    )


def _info_rule(name: str = "test_info", violated: bool = False) -> Rule:
    return Rule(
        name=name,
        description=f"Info rule: {name}",
        severity=Severity.INFO,
        condition=lambda _: violated,
    )


# ---------------------------------------------------------------------------
# evaluate — verdict logic
# ---------------------------------------------------------------------------


class TestEvaluateVerdict:
    def test_all_rules_pass_returns_approved(self) -> None:
        rules = [_block_rule(violated=False), _warn_rule(violated=False)]
        result = evaluate({}, rules)
        assert result.verdict == Verdict.APPROVED

    def test_warn_fires_no_block_returns_flagged(self) -> None:
        rules = [_block_rule(violated=False), _warn_rule(violated=True)]
        result = evaluate({}, rules)
        assert result.verdict == Verdict.FLAGGED

    def test_block_fires_returns_rejected(self) -> None:
        rules = [_block_rule(violated=True), _warn_rule(violated=False)]
        result = evaluate({}, rules)
        assert result.verdict == Verdict.REJECTED

    def test_block_and_warn_both_fire_returns_rejected(self) -> None:
        rules = [_block_rule(violated=True), _warn_rule(violated=True)]
        result = evaluate({}, rules)
        assert result.verdict == Verdict.REJECTED

    def test_info_violation_alone_does_not_flag(self) -> None:
        rules = [_info_rule(violated=True)]
        result = evaluate({}, rules)
        assert result.verdict == Verdict.APPROVED

    def test_info_and_warn_violations_returns_flagged(self) -> None:
        rules = [_info_rule(violated=True), _warn_rule(violated=True)]
        result = evaluate({}, rules)
        assert result.verdict == Verdict.FLAGGED

    def test_empty_rules_list_returns_approved(self) -> None:
        result = evaluate({"amount": 100}, [])
        assert result.verdict == Verdict.APPROVED

    def test_empty_output_dict_all_rules_pass(self) -> None:
        rules = [_block_rule(violated=False)]
        result = evaluate({}, rules)
        assert result.verdict == Verdict.APPROVED

    def test_multiple_block_violations_returns_rejected(self) -> None:
        rules = [
            _block_rule("b1", violated=True),
            _block_rule("b2", violated=True),
            _block_rule("b3", violated=True),
        ]
        result = evaluate({}, rules)
        assert result.verdict == Verdict.REJECTED


# ---------------------------------------------------------------------------
# evaluate — violations and all_results
# ---------------------------------------------------------------------------


class TestEvaluateResults:
    def test_violations_contains_only_violated_rules(self) -> None:
        rules = [
            _block_rule("pass_block", violated=False),
            _warn_rule("fail_warn", violated=True),
        ]
        result = evaluate({}, rules)
        assert len(result.violations) == 1
        assert result.violations[0].rule_name == "fail_warn"

    def test_all_results_contains_every_rule(self) -> None:
        rules = [_block_rule("a"), _warn_rule("b"), _info_rule("c")]
        result = evaluate({}, rules)
        assert len(result.all_results) == 3
        names = {r.rule_name for r in result.all_results}
        assert names == {"a", "b", "c"}

    def test_all_results_order_matches_rules_order(self) -> None:
        rules = [_block_rule("first"), _warn_rule("second"), _info_rule("third")]
        result = evaluate({}, rules)
        assert [r.rule_name for r in result.all_results] == ["first", "second", "third"]

    def test_violations_empty_when_all_pass(self) -> None:
        rules = [_block_rule(violated=False), _warn_rule(violated=False)]
        result = evaluate({}, rules)
        assert result.violations == []

    def test_rule_result_violated_true_for_failed_rule(self) -> None:
        rules = [_block_rule("r", violated=True)]
        result = evaluate({}, rules)
        assert result.all_results[0].violated is True

    def test_rule_result_violated_false_for_passing_rule(self) -> None:
        rules = [_block_rule("r", violated=False)]
        result = evaluate({}, rules)
        assert result.all_results[0].violated is False

    def test_rule_result_severity_matches_rule_severity(self) -> None:
        rules = [_warn_rule("r")]
        result = evaluate({}, rules)
        assert result.all_results[0].severity == Severity.WARN

    def test_rule_result_description_matches_rule_description(self) -> None:
        rule = Rule("r", "Custom description", Severity.BLOCK, lambda _: False)
        result = evaluate({}, [rule])
        assert result.all_results[0].description == "Custom description"

    def test_returns_guardrail_result_instance(self) -> None:
        result = evaluate({}, [])
        assert isinstance(result, GuardrailResult)

    def test_violations_are_rule_result_instances(self) -> None:
        rules = [_block_rule(violated=True)]
        result = evaluate({}, rules)
        assert all(isinstance(r, RuleResult) for r in result.violations)

    def test_all_results_are_rule_result_instances(self) -> None:
        rules = [_block_rule(), _warn_rule()]
        result = evaluate({}, rules)
        assert all(isinstance(r, RuleResult) for r in result.all_results)

    def test_empty_rules_violations_is_empty_list(self) -> None:
        result = evaluate({}, [])
        assert result.violations == []

    def test_empty_rules_all_results_is_empty_list(self) -> None:
        result = evaluate({}, [])
        assert result.all_results == []


# ---------------------------------------------------------------------------
# evaluate — audit trail
# ---------------------------------------------------------------------------


class TestEvaluateAuditTrail:
    def test_audit_trail_is_string(self) -> None:
        result = evaluate({}, [_block_rule("r")])
        assert isinstance(result.audit_trail, str)

    def test_audit_trail_contains_all_rule_names(self) -> None:
        rules = [_block_rule("rule_a"), _warn_rule("rule_b"), _info_rule("rule_c")]
        result = evaluate({}, rules)
        for name in ("rule_a", "rule_b", "rule_c"):
            assert name in result.audit_trail

    def test_audit_trail_contains_pass_for_passing_rule(self) -> None:
        rules = [_block_rule("r", violated=False)]
        result = evaluate({}, rules)
        assert "PASS" in result.audit_trail

    def test_audit_trail_contains_fail_for_violated_rule(self) -> None:
        rules = [_block_rule("r", violated=True)]
        result = evaluate({}, rules)
        assert "FAIL" in result.audit_trail

    def test_audit_trail_contains_verdict(self) -> None:
        result = evaluate({}, [_block_rule(violated=False)])
        assert "APPROVED" in result.audit_trail.upper()

    def test_audit_trail_contains_rejected_on_block(self) -> None:
        result = evaluate({}, [_block_rule(violated=True)])
        assert "REJECTED" in result.audit_trail.upper()

    def test_audit_trail_contains_flagged_on_warn(self) -> None:
        result = evaluate({}, [_warn_rule(violated=True)])
        assert "FLAGGED" in result.audit_trail.upper()

    def test_audit_trail_names_violated_rules(self) -> None:
        rules = [_block_rule("culprit", violated=True)]
        result = evaluate({}, rules)
        assert "culprit" in result.audit_trail


# ---------------------------------------------------------------------------
# evaluate — validation
# ---------------------------------------------------------------------------


class TestEvaluateValidation:
    def test_duplicate_rule_names_raises_value_error(self) -> None:
        rules = [_block_rule("same_name"), _warn_rule("same_name")]
        with pytest.raises(ValueError, match="Duplicate rule name"):
            evaluate({}, rules)

    def test_three_rules_same_name_raises_value_error(self) -> None:
        rules = [_block_rule("x"), _warn_rule("x"), _info_rule("x")]
        with pytest.raises(ValueError):
            evaluate({}, rules)

    def test_unique_names_do_not_raise(self) -> None:
        rules = [_block_rule("a"), _warn_rule("b"), _info_rule("c")]
        evaluate({}, rules)  # must not raise


# ---------------------------------------------------------------------------
# build_range_rule
# ---------------------------------------------------------------------------


class TestBuildRangeRule:
    def test_neither_min_nor_max_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="At least one of"):
            build_range_rule("r", "x")

    def test_max_only_fires_when_exceeded(self) -> None:
        rule = build_range_rule("r", "amount", max_value=100.0)
        assert rule.condition({"amount": 150}) is True

    def test_max_only_passes_when_equal(self) -> None:
        rule = build_range_rule("r", "amount", max_value=100.0)
        assert rule.condition({"amount": 100.0}) is False

    def test_max_only_passes_when_below(self) -> None:
        rule = build_range_rule("r", "amount", max_value=100.0)
        assert rule.condition({"amount": 50}) is False

    def test_min_only_fires_when_below(self) -> None:
        rule = build_range_rule("r", "score", min_value=0.5)
        assert rule.condition({"score": 0.2}) is True

    def test_min_only_passes_when_equal(self) -> None:
        rule = build_range_rule("r", "score", min_value=0.5)
        assert rule.condition({"score": 0.5}) is False

    def test_min_only_passes_when_above(self) -> None:
        rule = build_range_rule("r", "score", min_value=0.5)
        assert rule.condition({"score": 0.9}) is False

    def test_both_fires_when_below_min(self) -> None:
        rule = build_range_rule("r", "x", min_value=1.0, max_value=10.0)
        assert rule.condition({"x": 0.5}) is True

    def test_both_fires_when_above_max(self) -> None:
        rule = build_range_rule("r", "x", min_value=1.0, max_value=10.0)
        assert rule.condition({"x": 11.0}) is True

    def test_both_passes_within_range(self) -> None:
        rule = build_range_rule("r", "x", min_value=1.0, max_value=10.0)
        assert rule.condition({"x": 5.0}) is False

    def test_value_at_exact_min_boundary_passes(self) -> None:
        rule = build_range_rule("r", "x", min_value=1.0, max_value=10.0)
        assert rule.condition({"x": 1.0}) is False

    def test_value_at_exact_max_boundary_passes(self) -> None:
        rule = build_range_rule("r", "x", min_value=1.0, max_value=10.0)
        assert rule.condition({"x": 10.0}) is False

    def test_field_absent_does_not_fire(self) -> None:
        rule = build_range_rule("r", "amount", max_value=100.0)
        assert rule.condition({}) is False
        assert rule.condition({"other_field": 200}) is False

    def test_custom_description_is_used(self) -> None:
        rule = build_range_rule("r", "x", max_value=5.0, description="Custom desc")
        assert rule.description == "Custom desc"

    def test_auto_description_generated_for_max_only(self) -> None:
        rule = build_range_rule("r", "amount", max_value=1000.0)
        assert "amount" in rule.description
        assert "1000" in rule.description

    def test_auto_description_generated_for_min_only(self) -> None:
        rule = build_range_rule("r", "score", min_value=0.0)
        assert "score" in rule.description

    def test_auto_description_generated_for_both(self) -> None:
        rule = build_range_rule("r", "x", min_value=1.0, max_value=10.0)
        assert "x" in rule.description

    def test_default_severity_is_block(self) -> None:
        rule = build_range_rule("r", "x", max_value=10.0)
        assert rule.severity == Severity.BLOCK

    def test_custom_severity_is_applied(self) -> None:
        rule = build_range_rule("r", "x", max_value=10.0, severity=Severity.WARN)
        assert rule.severity == Severity.WARN

    def test_returns_rule_instance(self) -> None:
        rule = build_range_rule("r", "x", max_value=10.0)
        assert isinstance(rule, Rule)

    def test_name_is_set(self) -> None:
        rule = build_range_rule("my_rule", "x", max_value=10.0)
        assert rule.name == "my_rule"


# ---------------------------------------------------------------------------
# build_required_field_rule
# ---------------------------------------------------------------------------


class TestBuildRequiredFieldRule:
    def test_field_present_with_value_passes(self) -> None:
        rule = build_required_field_rule("r", "reference")
        assert rule.condition({"reference": "REF-001"}) is False

    def test_field_missing_fires(self) -> None:
        rule = build_required_field_rule("r", "reference")
        assert rule.condition({}) is True

    def test_field_none_fires(self) -> None:
        rule = build_required_field_rule("r", "reference")
        assert rule.condition({"reference": None}) is True

    def test_field_empty_string_fires(self) -> None:
        rule = build_required_field_rule("r", "reference")
        assert rule.condition({"reference": ""}) is True

    def test_field_zero_does_not_fire(self) -> None:
        """Zero is a valid value — only None and empty string are rejected."""
        rule = build_required_field_rule("r", "amount")
        assert rule.condition({"amount": 0}) is False

    def test_field_false_does_not_fire(self) -> None:
        """False is a valid value — only None and empty string are rejected."""
        rule = build_required_field_rule("r", "active")
        assert rule.condition({"active": False}) is False

    def test_custom_description_is_used(self) -> None:
        rule = build_required_field_rule("r", "ref", description="Custom")
        assert rule.description == "Custom"

    def test_auto_description_contains_field_name(self) -> None:
        rule = build_required_field_rule("r", "my_field")
        assert "my_field" in rule.description

    def test_default_severity_is_block(self) -> None:
        rule = build_required_field_rule("r", "field")
        assert rule.severity == Severity.BLOCK

    def test_custom_severity_is_applied(self) -> None:
        rule = build_required_field_rule("r", "field", severity=Severity.WARN)
        assert rule.severity == Severity.WARN

    def test_returns_rule_instance(self) -> None:
        assert isinstance(build_required_field_rule("r", "f"), Rule)


# ---------------------------------------------------------------------------
# build_allowed_values_rule
# ---------------------------------------------------------------------------


class TestBuildAllowedValuesRule:
    def test_value_in_allowed_set_passes(self) -> None:
        rule = build_allowed_values_rule("r", "currency", {"GBP", "EUR"})
        assert rule.condition({"currency": "GBP"}) is False

    def test_value_not_in_allowed_set_fires(self) -> None:
        rule = build_allowed_values_rule("r", "currency", {"GBP", "EUR"})
        assert rule.condition({"currency": "BTC"}) is True

    def test_empty_allowed_set_always_fires_when_field_present(self) -> None:
        rule = build_allowed_values_rule("r", "currency", set())
        assert rule.condition({"currency": "GBP"}) is True

    def test_field_absent_does_not_fire(self) -> None:
        rule = build_allowed_values_rule("r", "currency", {"GBP"})
        assert rule.condition({}) is False

    def test_custom_description_is_used(self) -> None:
        rule = build_allowed_values_rule("r", "currency", {"GBP"}, description="Custom")
        assert rule.description == "Custom"

    def test_auto_description_contains_field_name(self) -> None:
        rule = build_allowed_values_rule("r", "my_field", {"A", "B"})
        assert "my_field" in rule.description

    def test_auto_description_contains_allowed_values(self) -> None:
        rule = build_allowed_values_rule("r", "currency", {"GBP", "EUR"})
        assert "EUR" in rule.description or "GBP" in rule.description

    def test_default_severity_is_block(self) -> None:
        rule = build_allowed_values_rule("r", "f", {"a"})
        assert rule.severity == Severity.BLOCK

    def test_custom_severity_is_applied(self) -> None:
        rule = build_allowed_values_rule("r", "f", {"a"}, severity=Severity.WARN)
        assert rule.severity == Severity.WARN

    def test_returns_rule_instance(self) -> None:
        assert isinstance(build_allowed_values_rule("r", "f", {"a"}), Rule)

    def test_numeric_value_not_in_allowed_fires(self) -> None:
        """Numeric values are coerced to str for comparison."""
        rule = build_allowed_values_rule("r", "code", {"1", "2", "3"})
        assert rule.condition({"code": "4"}) is True

    def test_single_allowed_value_passes(self) -> None:
        rule = build_allowed_values_rule("r", "status", {"active"})
        assert rule.condition({"status": "active"}) is False


# ---------------------------------------------------------------------------
# Custom rule with lambda condition
# ---------------------------------------------------------------------------


class TestCustomRule:
    def test_lambda_condition_fires_when_true(self) -> None:
        rule = Rule(
            name="no_negative",
            description="Amount must be non-negative",
            severity=Severity.BLOCK,
            condition=lambda o: o.get("amount", 0) < 0,
        )
        result = evaluate({"amount": -1}, [rule])
        assert result.verdict == Verdict.REJECTED

    def test_lambda_condition_passes_when_false(self) -> None:
        rule = Rule(
            name="no_negative",
            description="Amount must be non-negative",
            severity=Severity.BLOCK,
            condition=lambda o: o.get("amount", 0) < 0,
        )
        result = evaluate({"amount": 100}, [rule])
        assert result.verdict == Verdict.APPROVED

    def test_complex_cross_field_rule(self) -> None:
        """Rules can reference multiple fields simultaneously."""
        rule = Rule(
            name="high_value_usd_only",
            description="Transactions over 50,000 must use USD",
            severity=Severity.WARN,
            condition=lambda o: o.get("amount", 0) > 50_000 and o.get("currency") != "USD",
        )
        result = evaluate({"amount": 60_000, "currency": "GBP"}, [rule])
        assert result.verdict == Verdict.FLAGGED

    def test_rule_with_info_severity_in_evaluate(self) -> None:
        rule = Rule(
            name="note",
            description="Informational note",
            severity=Severity.INFO,
            condition=lambda _: True,
        )
        result = evaluate({}, [rule])
        assert result.verdict == Verdict.APPROVED
        assert len(result.violations) == 1
        assert result.violations[0].severity == Severity.INFO


# ---------------------------------------------------------------------------
# Integration: full policy evaluation
# ---------------------------------------------------------------------------


class TestIntegration:
    def _make_policy(self) -> list[Rule]:
        return [
            build_range_rule("max_amount", "amount", max_value=10_000.0),
            build_required_field_rule("required_ref", "reference"),
            build_allowed_values_rule("allowed_currency", "currency", {"GBP", "EUR", "USD"}),
            # Custom rule: fires when amount is large (> £5,000), used for flagging.
            # build_range_rule uses min_value to mean "minimum allowed" (fires when
            # too small), so a "flag large amounts" rule is expressed as a lambda.
            Rule(
                name="large_flag",
                description="Transactions over £5,000 require manual review",
                severity=Severity.WARN,
                condition=lambda o: o.get("amount", 0) > 5_000,
            ),
        ]

    def test_approved_output_passes_all_rules(self) -> None:
        policy = self._make_policy()
        output = {"amount": 500.0, "reference": "REF-001", "currency": "GBP"}
        result = evaluate(output, policy)
        assert result.verdict == Verdict.APPROVED
        assert result.violations == []

    def test_flagged_output_triggers_warn_only(self) -> None:
        policy = self._make_policy()
        output = {"amount": 7_500.0, "reference": "REF-002", "currency": "EUR"}
        result = evaluate(output, policy)
        assert result.verdict == Verdict.FLAGGED
        assert all(r.severity != Severity.BLOCK for r in result.violations)

    def test_rejected_output_triggers_block(self) -> None:
        policy = self._make_policy()
        output = {"amount": 20_000.0, "reference": "", "currency": "BTC"}
        result = evaluate(output, policy)
        assert result.verdict == Verdict.REJECTED

    def test_rejected_output_has_multiple_violations(self) -> None:
        policy = self._make_policy()
        output = {"amount": 20_000.0, "reference": "", "currency": "BTC"}
        result = evaluate(output, policy)
        assert len(result.violations) >= 2

    def test_audit_trail_documents_all_rules(self) -> None:
        policy = self._make_policy()
        output = {"amount": 500.0, "reference": "REF-001", "currency": "GBP"}
        result = evaluate(output, policy)
        for rule in policy:
            assert rule.name in result.audit_trail

    def test_empty_output_against_required_field_rules(self) -> None:
        rules = [
            build_required_field_rule("req_a", "field_a"),
            build_required_field_rule("req_b", "field_b"),
        ]
        result = evaluate({}, rules)
        assert result.verdict == Verdict.REJECTED
        assert len(result.violations) == 2

    def test_boundary_amount_exactly_at_max_is_approved(self) -> None:
        policy = [build_range_rule("max", "amount", max_value=10_000.0)]
        result = evaluate({"amount": 10_000.0}, policy)
        assert result.verdict == Verdict.APPROVED

    def test_amount_one_penny_over_max_is_rejected(self) -> None:
        policy = [build_range_rule("max", "amount", max_value=10_000.0)]
        result = evaluate({"amount": 10_000.01}, policy)
        assert result.verdict == Verdict.REJECTED


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TestEnumerations:
    def test_severity_values(self) -> None:
        assert Severity.BLOCK.value == "block"
        assert Severity.WARN.value == "warn"
        assert Severity.INFO.value == "info"

    def test_verdict_values(self) -> None:
        assert Verdict.APPROVED.value == "approved"
        assert Verdict.FLAGGED.value == "flagged"
        assert Verdict.REJECTED.value == "rejected"

    def test_severity_is_str_enum(self) -> None:
        assert isinstance(Severity.BLOCK, str)

    def test_verdict_is_str_enum(self) -> None:
        assert isinstance(Verdict.APPROVED, str)


# ---------------------------------------------------------------------------
# main() smoke tests
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_runs_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        from deterministic_guardrails import main
        main()

    def test_main_output_contains_approved(self, capsys: pytest.CaptureFixture[str]) -> None:
        from deterministic_guardrails import main
        main()
        assert "APPROVED" in capsys.readouterr().out

    def test_main_output_contains_flagged(self, capsys: pytest.CaptureFixture[str]) -> None:
        from deterministic_guardrails import main
        main()
        assert "FLAGGED" in capsys.readouterr().out

    def test_main_output_contains_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        from deterministic_guardrails import main
        main()
        assert "REJECTED" in capsys.readouterr().out

    def test_main_output_contains_guardrails(self, capsys: pytest.CaptureFixture[str]) -> None:
        from deterministic_guardrails import main
        main()
        assert "Guardrails" in capsys.readouterr().out
