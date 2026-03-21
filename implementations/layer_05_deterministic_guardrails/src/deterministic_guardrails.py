"""
Layer 05: Deterministic Guardrails

The constitutional court that holds final power.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

This module implements the "model proposes, rule decides" architecture.
A probabilistic model produces an output.  A deterministic rule layer
validates that output against a policy before it is permitted through.
The rule layer is transparent, reproducible, and auditable — every decision
traces back to a specific rule with a human-readable description.

Key properties:
- Every rule is evaluated independently for every output.
- The verdict is determined solely by the highest-severity violation.
- The audit trail documents every rule evaluation, pass or fail.
- Zero external dependencies; pure Python standard library only.

Rule severities and their effect on the verdict:
    BLOCK → REJECTED    (any blocking violation stops the output)
    WARN  → FLAGGED     (output proceeds but is flagged for review)
    INFO  → APPROVED*   (logged for audit only; no verdict effect)
    (* INFO violations do not change the verdict to FLAGGED or REJECTED.)

See layers/05-deterministic-guardrails.md for the full explanation,
production examples, and limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Severity level of a guardrail rule.

    Determines how a violation affects the overall verdict.

    Attributes:
        BLOCK: Hard stop — the output cannot proceed.
        WARN:  Output proceeds but is flagged for review.
        INFO:  Logged for audit only; no effect on the verdict.
    """

    BLOCK = "block"
    WARN = "warn"
    INFO = "info"


class Verdict(str, Enum):
    """Overall verdict produced by evaluating an output against a policy.

    Attributes:
        APPROVED: No blocking or warning rules fired.
        FLAGGED:  At least one WARN rule fired; no BLOCK rules fired.
        REJECTED: At least one BLOCK rule fired.
    """

    APPROVED = "approved"
    FLAGGED = "flagged"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """A single deterministic policy rule.

    Attributes:
        name:        Unique identifier for this rule (e.g. "max_amount").
        description: Human-readable description of what the rule checks
                     (e.g. "Transaction must not exceed £10,000").
        severity:    What happens when the rule is violated.
        condition:   Callable that receives the model output dict and returns
                     True when the rule is **violated**, False when it passes.
    """

    name: str
    description: str
    severity: Severity
    condition: Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class RuleResult:
    """The outcome of evaluating a single rule against a model output.

    Attributes:
        rule_name:   Name of the rule that was evaluated.
        severity:    Severity of the rule.
        violated:    True if the condition returned True (rule violated).
        description: Human-readable description copied from the rule.
    """

    rule_name: str
    severity: Severity
    violated: bool
    description: str


@dataclass
class GuardrailResult:
    """The complete result of evaluating a policy against a model output.

    Attributes:
        verdict:     APPROVED, FLAGGED, or REJECTED.
        violations:  RuleResult entries where violated=True only.
        all_results: RuleResult for every rule, pass or fail.
        audit_trail: Human-readable string documenting every rule evaluation.
    """

    verdict: Verdict
    violations: list[RuleResult]
    all_results: list[RuleResult]
    audit_trail: str


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


def evaluate(
    output: dict[str, Any],
    rules: list[Rule],
) -> GuardrailResult:
    """Evaluate a model output against a list of deterministic rules.

    Every rule is evaluated.  The verdict is determined by the highest
    severity violation: any BLOCK → REJECTED, any WARN (no BLOCK) → FLAGGED,
    otherwise APPROVED.  INFO violations do not affect the verdict.

    The audit_trail is a human-readable string documenting every rule
    evaluation — what was checked, whether it passed or failed, and why.
    This is the paper trail that makes the system auditable.

    Args:
        output: The model output as a dict of field names to values.
        rules:  Ordered list of Rule objects that constitute the policy.

    Returns:
        GuardrailResult with verdict, violations, all_results, and audit_trail.

    Raises:
        ValueError: If any two rules share the same name.
    """
    names = [r.name for r in rules]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(
                f"Duplicate rule name {name!r}. Each rule must have a unique name."
            )
        seen.add(name)

    all_results: list[RuleResult] = []
    violations: list[RuleResult] = []
    audit_lines: list[str] = []

    for rule in rules:
        violated = rule.condition(output)
        result = RuleResult(
            rule_name=rule.name,
            severity=rule.severity,
            violated=violated,
            description=rule.description,
        )
        all_results.append(result)
        if violated:
            violations.append(result)

        tag = "FAIL" if violated else "PASS"
        severity_tag = f" ({rule.severity.value.upper()})" if violated else ""
        audit_lines.append(
            f"  [{tag}] {rule.name}{severity_tag}: {rule.description}"
        )

    # Determine verdict from highest-severity violation
    has_block = any(r.severity == Severity.BLOCK for r in violations)
    has_warn = any(r.severity == Severity.WARN for r in violations)

    if has_block:
        verdict = Verdict.REJECTED
    elif has_warn:
        verdict = Verdict.FLAGGED
    else:
        verdict = Verdict.APPROVED

    verdict_line = f"Verdict: {verdict.value.upper()}"
    if violations:
        violation_summary = ", ".join(r.rule_name for r in violations)
        verdict_line += f" — violations: {violation_summary}"

    audit_trail = verdict_line + "\n" + "\n".join(audit_lines)

    return GuardrailResult(
        verdict=verdict,
        violations=violations,
        all_results=all_results,
        audit_trail=audit_trail,
    )


# ---------------------------------------------------------------------------
# Rule factories
# ---------------------------------------------------------------------------


def build_range_rule(
    name: str,
    field: str,
    min_value: float | None = None,
    max_value: float | None = None,
    severity: Severity = Severity.BLOCK,
    description: str | None = None,
) -> Rule:
    """Build a Rule that checks a numeric field falls within a given range.

    The condition fires (returns True, meaning violated) when the field value
    is below min_value or above max_value.  If the field is absent from the
    output the rule does not fire — pair with build_required_field_rule to
    enforce field presence.

    Args:
        name:        Rule identifier.
        field:       Key to look up in the model output dict.
        min_value:   Minimum acceptable value (inclusive).  None = no lower bound.
        max_value:   Maximum acceptable value (inclusive).  None = no upper bound.
        severity:    Severity of the violation.  Defaults to BLOCK.
        description: Human-readable description.  Auto-generated if None.

    Returns:
        A Rule whose condition returns True when the range is violated.

    Raises:
        ValueError: If neither min_value nor max_value is provided.
    """
    if min_value is None and max_value is None:
        raise ValueError(
            "At least one of min_value or max_value must be provided."
        )

    if description is None:
        parts: list[str] = []
        if min_value is not None:
            parts.append(f">= {min_value}")
        if max_value is not None:
            parts.append(f"<= {max_value}")
        description = f"{field} must be {' and '.join(parts)}"

    def condition(output: dict[str, Any]) -> bool:
        if field not in output:
            return False
        value = output[field]
        if min_value is not None and value < min_value:
            return True
        if max_value is not None and value > max_value:
            return True
        return False

    return Rule(name=name, description=description, severity=severity, condition=condition)


def build_required_field_rule(
    name: str,
    field: str,
    severity: Severity = Severity.BLOCK,
    description: str | None = None,
) -> Rule:
    """Build a Rule that checks a required field is present and non-empty.

    The condition fires when the field is absent from the output dict, is
    None, or is an empty string.

    Args:
        name:        Rule identifier.
        field:       Key that must be present in the model output dict.
        severity:    Severity of the violation.  Defaults to BLOCK.
        description: Human-readable description.  Auto-generated if None.

    Returns:
        A Rule whose condition returns True when the field is missing or empty.
    """
    if description is None:
        description = f"{field} is required and must not be empty"

    def condition(output: dict[str, Any]) -> bool:
        value = output.get(field)
        return value is None or value == ""

    return Rule(name=name, description=description, severity=severity, condition=condition)


def build_allowed_values_rule(
    name: str,
    field: str,
    allowed: set[str],
    severity: Severity = Severity.BLOCK,
    description: str | None = None,
) -> Rule:
    """Build a Rule that checks a field's value is within a permitted set.

    The condition fires when the field is present in the output but its value
    is not in the allowed set.  If the field is absent the rule does not fire
    — pair with build_required_field_rule to enforce field presence.

    Args:
        name:        Rule identifier.
        field:       Key to look up in the model output dict.
        allowed:     Set of permitted string values.
        severity:    Severity of the violation.  Defaults to BLOCK.
        description: Human-readable description.  Auto-generated if None.

    Returns:
        A Rule whose condition returns True when the value is not permitted.
    """
    if description is None:
        sorted_allowed = sorted(allowed)
        description = f"{field} must be one of: {', '.join(sorted_allowed)}"

    def condition(output: dict[str, Any]) -> bool:
        if field not in output:
            return False
        return str(output[field]) not in allowed

    return Rule(name=name, description=description, severity=severity, condition=condition)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    """Demonstrate the "model proposes, rule decides" pattern.

    A financial transaction validation policy with six rules runs three
    model outputs through the engine, showing APPROVED, FLAGGED, and
    REJECTED verdicts together with their full audit trails.
    """
    print("Layer 05: Deterministic Guardrails")
    print("=" * 50)
    print()
    print('The "model proposes, rule decides" pattern.')
    print()

    # -----------------------------------------------------------------------
    # Define the policy
    # -----------------------------------------------------------------------
    policy: list[Rule] = [
        build_range_rule(
            name="max_transaction_amount",
            field="amount",
            max_value=10_000.0,
            severity=Severity.BLOCK,
            description="Transaction must not exceed £10,000",
        ),
        build_required_field_rule(
            name="required_reference",
            field="reference",
            severity=Severity.BLOCK,
            description="Transaction reference is required",
        ),
        build_allowed_values_rule(
            name="allowed_currencies",
            field="currency",
            allowed={"GBP", "EUR", "USD"},
            severity=Severity.BLOCK,
            description="Currency must be GBP, EUR, or USD",
        ),
        Rule(
            name="large_transaction_flag",
            description="Transactions over £5,000 require manual review",
            severity=Severity.WARN,
            condition=lambda o: o.get("amount", 0) > 5_000,
        ),
        Rule(
            name="suspicious_round_amount",
            description="Exact round amounts of £1,000+ may indicate structuring",
            severity=Severity.INFO,
            condition=lambda o: (
                "amount" in o
                and isinstance(o["amount"], (int, float))
                and o["amount"] >= 1_000
                and o["amount"] % 1_000 == 0
            ),
        ),
        build_required_field_rule(
            name="required_account_id",
            field="account_id",
            severity=Severity.BLOCK,
            description="Destination account ID is required",
        ),
    ]

    print("Policy: Financial Transaction Validation")
    print("-" * 42)
    for rule in policy:
        print(f"  [{rule.severity.value.upper():<5}] {rule.name}: {rule.description}")
    print()

    # -----------------------------------------------------------------------
    # Scenario 1 — All rules pass → APPROVED
    # -----------------------------------------------------------------------
    print("Scenario 1 — Standard transaction (all rules pass)")
    print("-" * 51)
    output_1 = {
        "amount": 1_500.00,
        "currency": "GBP",
        "reference": "PAY-2024-001",
        "account_id": "GB29NWBK60161331926819",
    }
    print(f"  Model output: {output_1}")
    result_1 = evaluate(output_1, policy)
    print(f"  Verdict: {result_1.verdict.value.upper()}")
    print()
    print("  Audit trail:")
    for line in result_1.audit_trail.splitlines():
        print(f"  {line}")
    print()

    # -----------------------------------------------------------------------
    # Scenario 2 — Warning fires but no blocks → FLAGGED
    # -----------------------------------------------------------------------
    print("Scenario 2 — Large transaction (warning fires, no blocks)")
    print("-" * 57)
    output_2 = {
        "amount": 7_500.00,
        "currency": "EUR",
        "reference": "PAY-2024-002",
        "account_id": "DE89370400440532013000",
    }
    print(f"  Model output: {output_2}")
    result_2 = evaluate(output_2, policy)
    print(f"  Verdict: {result_2.verdict.value.upper()}")
    print()
    print("  Audit trail:")
    for line in result_2.audit_trail.splitlines():
        print(f"  {line}")
    print()

    # -----------------------------------------------------------------------
    # Scenario 3 — Blocking rules fire → REJECTED
    # -----------------------------------------------------------------------
    print("Scenario 3 — Invalid transaction (blocking rules fire)")
    print("-" * 54)
    output_3 = {
        "amount": 15_000.00,
        "currency": "BTC",
        "reference": "",
        "account_id": "GB29NWBK60161331926819",
    }
    print(f"  Model output: {output_3}")
    result_3 = evaluate(output_3, policy)
    print(f"  Verdict: {result_3.verdict.value.upper()}")
    print()
    print("  Audit trail:")
    for line in result_3.audit_trail.splitlines():
        print(f"  {line}")
    print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("Summary:")
    print(f"  {'Scenario':<40}  {'Verdict':>8}  {'Violations':>10}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*10}")
    for label, res in [
        ("Standard transaction", result_1),
        ("Large transaction (review required)", result_2),
        ("Invalid transaction (blocked)", result_3),
    ]:
        n = len(res.violations)
        print(f"  {label:<40}  {res.verdict.value.upper():>8}  {n:>10}")
    print()
    print("See: layers/05-deterministic-guardrails.md")


if __name__ == "__main__":  # pragma: no cover
    main()
