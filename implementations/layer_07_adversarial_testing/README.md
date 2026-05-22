# Layer 07: Adversarial Testing — Implementation

## Quick Start

```bash
make install
make run
```

## Expected Output

```
Layer 07: Adversarial Testing
==================================================

Pattern: model proposes, harness evaluates.
The system under test is a deterministic function, not an LLM.

Policy:
------------------------------
  [CRITICAL] no_instruction_bypass
  [HIGH    ] no_system_prompt_leakage
  [CRITICAL] no_credential_leakage
  [HIGH    ] no_pii_output

Test Results:
------------------------------------------------------------
  tc-001  PASS                baseline
  tc-002  PASS                baseline
  tc-003  FAIL [CRITICAL]     prompt_injection
             Rule: no_instruction_bypass
  tc-004  FAIL [HIGH]         prompt_injection
             Rule: no_system_prompt_leakage
  tc-005  FAIL [CRITICAL]     credential_extraction
             Rule: no_credential_leakage
  tc-006  FAIL [HIGH]         pii_extraction
             Rule: no_pii_output

Red Team Report
==================================================
Total tests:  6
Violations:   4
Pass rate:    33.3%

Findings by severity:
  HIGH       2
  CRITICAL   2

Findings by category:
  credential_extraction          1
  pii_extraction                 1
  prompt_injection               2

Findings:
  [CRITICAL] tc-003 (prompt_injection)
         Rule violated: no_instruction_bypass
  [HIGH] tc-004 (prompt_injection)
         Rule violated: no_system_prompt_leakage
  [CRITICAL] tc-005 (credential_extraction)
         Rule violated: no_credential_leakage
  [HIGH] tc-006 (pii_extraction)
         Rule violated: no_pii_output

See: layers/07-adversarial-testing.md
```

## What This Code Does

Implements a structured adversarial testing harness demonstrating the *pattern*
of red teaming — not the attacks themselves.  The system under test (SUT) is any
Python callable that accepts a prompt string and returns a response string; in
the demo it is a simple deterministic function that simulates a naive content
moderator with known failure modes (prompt injection bypass, system prompt
leakage, credential leakage).  No LLM calls are made.

A `PolicyRule` defines what the SUT must never output as a set of
forbidden-pattern strings; matching is case-insensitive substring search —
simple, transparent, and easy to audit.  Test cases are organised into
categories and can be scoped to specific rules via `target_rules`.
`RedTeamHarness.run_suite()` runs every test case, captures SUT exceptions
gracefully, computes per-severity and per-category summaries, and produces a
human-readable report.  The verdict on each test is determined by the highest
severity among violated rules.

## API

| Symbol | Signature | Description |
|---|---|---|
| `FindingSeverity` | `str, Enum` | NONE / LOW / MEDIUM / HIGH / CRITICAL |
| `PolicyRule` | frozen dataclass | name, description, forbidden_patterns, severity |
| `TestCase` | frozen dataclass | test_case_id, prompt, category, description, target_rules |
| `Finding` | dataclass | test_case_id, category, prompt, output, violated_rules, severity, error |
| `RedTeamReport` | dataclass | total_tests, findings, summary_by_severity, summary_by_category, pass_rate, report_text |
| `RedTeamHarness.__init__` | `(system_under_test, policy)` | Wraps a SUT callable with a policy |
| `RedTeamHarness.evaluate_output` | `(output, target_rules=()) → list[PolicyRule]` | Case-insensitive substring check; returns violated rules |
| `RedTeamHarness.run_test` | `(test_case) → Finding` | Run one test; catches SUT exceptions |
| `RedTeamHarness.run_suite` | `(test_cases) → RedTeamReport` | Run all tests; aggregate and report |

## Testing

```bash
make test
```

62 tests, 100% coverage.  Tests cover: frozen dataclass fields; evaluate_output
detection, case-insensitivity, target_rules filtering, empty output, empty
policy, non-existent rule names; run_test output capture, rule identification,
severity propagation, exception handling, empty-string SUT; run_suite totals,
findings contents, pass rate (0%, 50%, 100%), severity/category summaries,
report text, empty suite, raising SUT; FindingSeverity ordering; main() smoke
tests.

## Dependencies

Zero runtime dependencies.  Pure Python standard library: `dataclasses`, `enum`,
`traceback`, `typing`.
