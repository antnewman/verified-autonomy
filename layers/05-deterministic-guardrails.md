# Layer 05: Deterministic Guardrails

*The constitutional court that holds final power*

> **Primary author:** Ant Newman
>
> **Production examples:** Reference implementation; field examples from financial services, aviation, and live broadcast

---

## What It Is

The model proposes. The rule decides.

A deterministic guardrail is an architectural separation between a probabilistic model and a deterministic policy enforcement layer. The model generates an output — a classification, a recommendation, an extracted value. The guardrail layer then evaluates that output against a set of rules before it is permitted through. The model has no ability to override the rules. The rules have no ability to generate outputs. Each does what it is good at.

The rule layer has three properties that the model layer does not: it is **transparent** (every rule is readable as a plain-English statement of what it checks), **reproducible** (the same input always produces the same verdict), and **auditable** (every decision traces back to a specific rule that can be examined, explained, and updated by a non-specialist).

Each rule has four components: a **name** (unique identifier), a **description** (human-readable statement of what it checks), a **severity** that determines the effect of a violation, and a **condition** (a callable that returns True when the rule is violated). The severity determines the verdict:

| Severity | Effect | Example |
|---|---|---|
| **BLOCK** | Output is rejected. Hard stop. | Transaction exceeds £10,000 |
| **WARN** | Output proceeds but is flagged for review. | Transaction exceeds £5,000 |
| **INFO** | Logged for audit only. No effect on verdict. | Amount is a suspicious round number |

Every rule is evaluated for every output. The verdict is determined by the highest severity violation: any BLOCK produces REJECTED, any WARN (with no BLOCK) produces FLAGGED, otherwise APPROVED. INFO violations are recorded in the audit trail but do not change the verdict. The audit trail documents every rule evaluation — what was checked, whether it passed or failed, and why — creating the paper trail that makes the system explainable.

This is the simplest layer in the nine-layer architecture, and deliberately so. Deterministic systems are easy to understand, easy to test, and easy to change. When a regulation changes, you update a rule. When an incident occurs, the audit trail tells you exactly which rule permitted the output through. The model is a black box. The guardrail is not.

## Why It Matters

Consider a financial institution using an AI model to assess loan applications. The model produces a risk score and a recommendation — approve or reject. The model is sophisticated, trained on years of data, and generally accurate. But it has no concept of regulatory limits. It does not know that certain loan types require specific documentation. It does not know that the institution's risk appetite changed last quarter. It does not know about the fraud pattern that was identified last week.

If the model's recommendation goes directly to execution, every one of these constraints must be embedded in the model itself — retrained, revalidated, and redeployed every time a rule changes. This is slow, expensive, and opaque. Worse, it is impossible to prove which rule governed a specific decision after the fact, because the rule was learned, not specified.

If a deterministic guardrail sits between the model and execution, the constraints are specified explicitly, enforced independently, and auditable without understanding the model. The model recommends approval for a £15,000 loan. The guardrail checks the policy: maximum transaction amount is £10,000. REJECTED. The rule name, the description, the severity, and the verdict are all in the audit trail. A compliance officer can read it. A regulator can audit it. A developer can update it without retraining the model.

The same pattern applies across domains. In aviation maintenance, an AI system might recommend deferring a maintenance check based on sensor data and historical patterns. A deterministic guardrail validates the recommendation against the aircraft's approved maintenance programme — certain checks cannot be deferred regardless of what the model recommends. In live broadcast, a semantic content monitor might flag a segment for review; the guardrail layer enforces the hard rules about what can and cannot air, with graceful fallback to pre-approved content if a rule fires.

The failure mode is specific: **without a deterministic policy layer, the model's learned behaviour is the only control, and learned behaviour cannot be audited, explained, or updated without retraining.** In regulated environments, the ability to point to the specific rule that governed a decision — and to update that rule when the regulation changes — is not optional. It is the basis of compliance.

## Production Example

The implementation is open source in the Verified Autonomy repository at [`implementations/layer_05_deterministic_guardrails/`](https://github.com/antnewman/verified-autonomy/tree/main/implementations/layer_05_deterministic_guardrails).

**Run it yourself in under two minutes:**

```bash
cd implementations/layer_05_deterministic_guardrails
make install
make run
```

**Defining a policy:**

```python
from verified_autonomy.deterministic_guardrails import (
    evaluate, Rule, Severity, Verdict,
    build_range_rule, build_required_field_rule, build_allowed_values_rule,
)

policy = [
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
]
```

**Evaluating a model output:**

```python
output = {"amount": 15_000.0, "currency": "BTC", "reference": "", "account_id": "GB29..."}
result = evaluate(output, policy)

result.verdict      # Verdict.REJECTED
len(result.violations)  # 3 (amount, currency, reference)
print(result.audit_trail)
```

```
Verdict: REJECTED — violations: max_transaction_amount, required_reference, allowed_currencies
  [FAIL] max_transaction_amount (BLOCK): Transaction must not exceed £10,000
  [FAIL] required_reference (BLOCK): Transaction reference is required
  [FAIL] allowed_currencies (BLOCK): Currency must be GBP, EUR, or USD
  [FAIL] large_transaction_flag (WARN): Transactions over £5,000 require manual review
```

**The three verdict paths from `make run`:**

Scenario 1: a £1,500 GBP transaction with all required fields passes every rule. Verdict: APPROVED. All six rules listed as PASS in the audit trail.

Scenario 2: a £7,500 EUR transaction passes all BLOCK rules but triggers the large transaction warning. Verdict: FLAGGED. The output proceeds but a human reviewer is notified. The audit trail shows exactly which rule fired and why.

Scenario 3: a £15,000 BTC transaction with an empty reference. Three BLOCK rules fire (amount exceeds limit, currency not permitted, reference missing), plus the WARN and INFO rules. Verdict: REJECTED. The output does not proceed. The audit trail documents every rule evaluation.

**The rule factories (from the test suite, 98 tests, 100% coverage):** Three factory functions cover the most common validation patterns. `build_range_rule` checks a numeric field against minimum and/or maximum bounds — the tests verify min-only, max-only, both bounds, values at exact boundaries, and the behaviour when the field is absent (the rule does not fire — pair with `build_required_field_rule` to enforce presence). `build_required_field_rule` checks that a field exists and is not None or empty string. `build_allowed_values_rule` checks that a field's value is within a permitted set. All three generate human-readable descriptions automatically if none is provided.

Any Python callable works as a condition, including multi-field cross-validation rules and stateful checks. The `Rule` dataclass accepts any `Callable[[dict[str, Any]], bool]` — the factory functions are conveniences, not constraints.

**Duplicate rule names are rejected.** The `evaluate` function raises `ValueError` if any two rules share the same name. This prevents accidental shadowing where two rules with the same name might create ambiguity in the audit trail.

**Code ancestry:** This implementation is built from scratch using only the Python standard library. Zero runtime dependencies. The pattern is the same "policy engine" architecture used in financial compliance systems, clinical decision support guardrails, and content moderation pipelines — simplified to its essential mechanism.

## Limitations

**Rules can only govern anticipated scenarios.** A deterministic guardrail catches violations of rules that someone wrote. It does not catch violations of rules that nobody thought to write. If a novel fraud pattern emerges that is not covered by an existing rule, the guardrail will not catch it. The model might — learned behaviour can generalise to novel inputs in ways that specified rules cannot. The guardrail is a floor, not a ceiling. It guarantees certain violations are caught. It does not guarantee all violations are caught.

**Policy debt accumulates.** As the model is deployed in more contexts and handles more edge cases, the number of rules grows. Rules interact — a change to one rule's threshold might create a gap that another rule used to cover. Removing obsolete rules is risky because nobody is certain what they were protecting against. Over time, the policy becomes a legacy system in its own right, with the same maintenance burden, technical debt, and institutional knowledge problems as any long-lived codebase. Most teams underestimate this. The guardrail layer requires the same engineering discipline as the model layer — versioning, testing, review, and deprecation processes.

**Provides a floor, not a ceiling.** The guardrail prevents known-bad outputs. It does not improve outputs that are merely mediocre. A model that consistently produces low-quality but technically compliant outputs will pass every rule. The guardrail ensures "not wrong" but cannot ensure "good." Quality improvement requires techniques at other layers — confidence scoring (Layer 01), calibration (Layer 04), and explainability (Layer 06).

**Brittle under rapid change.** In fast-moving environments where rules change frequently (e.g. trading regulations, content moderation policies during breaking news), the lag between a rule change being identified and the guardrail being updated is a window of exposure. The deterministic layer is only as current as its last deployment. If the policy update pipeline is slower than the rate of regulatory change, the guardrail is enforcing yesterday's rules on today's decisions.

**Condition callables are opaque to static analysis.** While the rule descriptions are human-readable, the condition functions themselves are arbitrary Python callables — including lambdas and closures that cannot be meaningfully inspected at rest. A rule described as "Transaction must not exceed £10,000" could in principle contain a condition that checks something entirely different. The description is a convention, not a contract. In safety-critical deployments, conditions should be simple, well-tested, and reviewed with the same rigour as the model they guard.

## Reader Takeaway

Separate the model from the policy. The model proposes outputs; a deterministic rule layer decides whether each output is permitted. The rules are transparent, reproducible, and auditable — every decision traces back to a named rule with a human-readable description. This is the simplest layer in the nine-layer architecture and often the most immediately valuable in regulated environments. But treat the guardrail layer as a living system: policy debt accumulates, rules interact in unexpected ways, and the layer requires the same engineering discipline — testing, versioning, review — as the model it governs.

---

*Contributor note: Ant Newman wrote the primary draft and implementation. Zero external dependencies — pure Python standard library only.*
