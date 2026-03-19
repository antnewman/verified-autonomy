# Layer 05: Deterministic Guardrails — Implementation

## Quick Start

```bash
make install
make run
```

## Expected Output

```
Layer 05: Deterministic Guardrails
==================================================

The "model proposes, rule decides" pattern.

Policy: Financial Transaction Validation
------------------------------------------
  [BLOCK] max_transaction_amount: Transaction must not exceed £10,000
  [BLOCK] required_reference: Transaction reference is required
  [BLOCK] allowed_currencies: Currency must be GBP, EUR, or USD
  [WARN ] large_transaction_flag: Transactions over £5,000 require manual review
  [INFO ] suspicious_round_amount: Exact round amounts of £1,000+ may indicate structuring
  [BLOCK] required_account_id: Destination account ID is required

Scenario 1 — Standard transaction (all rules pass)
---------------------------------------------------
  Model output: {'amount': 1500.0, 'currency': 'GBP', 'reference': 'PAY-2024-001', 'account_id': 'GB29NWBK60161331926819'}
  Verdict: APPROVED

  Audit trail:
  Verdict: APPROVED
    [PASS] max_transaction_amount: Transaction must not exceed £10,000
    [PASS] required_reference: Transaction reference is required
    [PASS] allowed_currencies: Currency must be GBP, EUR, or USD
    [PASS] large_transaction_flag: Transactions over £5,000 require manual review
    [PASS] suspicious_round_amount: Exact round amounts of £1,000+ may indicate structuring
    [PASS] required_account_id: Destination account ID is required

Scenario 2 — Large transaction (warning fires, no blocks)
---------------------------------------------------------
  Model output: {'amount': 7500.0, 'currency': 'EUR', 'reference': 'PAY-2024-002', 'account_id': 'DE89370400440532013000'}
  Verdict: FLAGGED

  Audit trail:
  Verdict: FLAGGED — violations: large_transaction_flag
    [PASS] max_transaction_amount: Transaction must not exceed £10,000
    [PASS] required_reference: Transaction reference is required
    [PASS] allowed_currencies: Currency must be GBP, EUR, or USD
    [FAIL] large_transaction_flag (WARN): Transactions over £5,000 require manual review
    [PASS] suspicious_round_amount: Exact round amounts of £1,000+ may indicate structuring
    [PASS] required_account_id: Destination account ID is required

Scenario 3 — Invalid transaction (blocking rules fire)
------------------------------------------------------
  Model output: {'amount': 15000.0, 'currency': 'BTC', 'reference': '', 'account_id': 'GB29NWBK60161331926819'}
  Verdict: REJECTED

  Audit trail:
  Verdict: REJECTED — violations: max_transaction_amount, required_reference, allowed_currencies, large_transaction_flag, suspicious_round_amount
    [FAIL] max_transaction_amount (BLOCK): Transaction must not exceed £10,000
    [FAIL] required_reference (BLOCK): Transaction reference is required
    [FAIL] allowed_currencies (BLOCK): Currency must be GBP, EUR, or USD
    [FAIL] large_transaction_flag (WARN): Transactions over £5,000 require manual review
    [FAIL] suspicious_round_amount (INFO): Exact round amounts of £1,000+ may indicate structuring
    [PASS] required_account_id: Destination account ID is required

Summary:
  Scenario                                   Verdict  Violations
  ----------------------------------------  --------  ----------
  Standard transaction                      APPROVED           0
  Large transaction (review required)        FLAGGED           1
  Invalid transaction (blocked)             REJECTED           5

See: layers/05-deterministic-guardrails.md
```

## What This Code Does

Implements the "model proposes, rule decides" architecture as a domain-agnostic
policy engine.  A probabilistic model produces an output dictionary; the
guardrail layer validates it against an ordered list of `Rule` objects before
the output is permitted through.  Each rule has a name, a human-readable
description, a severity (`BLOCK`, `WARN`, or `INFO`), and a condition callable
that returns `True` when the rule is violated.  The engine evaluates every rule
independently, computes the overall verdict from the highest-severity violation,
and produces a full audit trail documenting every check.

Three rule factories cover the most common cases — range checks, required
fields, and allowed-value sets — but any Python callable works as a condition,
including multi-field cross-validation and stateful checks.  Zero external
dependencies; pure Python standard library only.

## API

| Symbol | Signature | Description |
|---|---|---|
| `Severity` | `str, Enum` | BLOCK / WARN / INFO — effect on verdict |
| `Verdict` | `str, Enum` | APPROVED / FLAGGED / REJECTED |
| `Rule` | frozen dataclass | name, description, severity, condition callable |
| `RuleResult` | frozen dataclass | rule_name, severity, violated, description |
| `GuardrailResult` | dataclass | verdict, violations, all_results, audit_trail |
| `evaluate` | `(output, rules) → GuardrailResult` | Evaluate output against all rules; raises ValueError on duplicate rule names |
| `build_range_rule` | `(name, field, min_value, max_value, severity, description) → Rule` | Fires when field value is outside [min, max]; raises ValueError if neither bound given |
| `build_required_field_rule` | `(name, field, severity, description) → Rule` | Fires when field is absent, None, or empty string |
| `build_allowed_values_rule` | `(name, field, allowed, severity, description) → Rule` | Fires when field value is not in the permitted set |

## Testing

```bash
make test
```

98 tests, 100% coverage.  Tests cover: verdict logic (APPROVED / FLAGGED /
REJECTED / INFO-only); violations and all_results contents; audit trail
completeness; duplicate rule name validation; all three factory functions
(min-only, max-only, both bounds, boundary values, absent fields, custom
descriptions, auto-generated descriptions, severity overrides); custom lambda
rules; cross-field rules; empty rules list; empty output dict; full integration
scenarios; enum values; and main() smoke tests.

## Dependencies

Zero runtime dependencies.  The entire engine is implemented using Python
standard library modules only: `dataclasses`, `enum`, and `typing`.
