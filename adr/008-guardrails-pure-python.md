# ADR 008: Pure Python Implementation for Deterministic Guardrails

## Status

Accepted

## Context

Layer 05 implements a deterministic policy engine — a rule layer that validates
AI model outputs before they are permitted through.  The core design question
is how to represent and evaluate rules.

Several external approaches exist:

- **Policy DSLs** (Rego/Open Policy Agent, Cedar, Ply): expressive, auditable,
  with dedicated tooling for policy management.
- **External rule engines** (Drools, PyKnow): structured inference engines with
  forward-chaining, conflict resolution, and priority ordering.
- **Schema validation libraries** (Pydantic, Cerberus): well-tested validation
  with declarative schemas, but primarily for data shape rather than policy.

All of these add a dependency and introduce a translation layer between the
policy intent and the executing code.  A rule expressed in Rego must be
understood in Rego syntax before it can be audited.  A Pydantic validator
requires understanding Pydantic's field validator protocol.

## Decision

Implement the guardrails layer in pure Python with zero external dependencies.
Each rule is a `Rule` dataclass with a `condition: Callable[[dict], bool]` that
returns `True` when the rule is violated.  The policy is a plain Python list of
`Rule` objects.  The engine is a single `evaluate()` function.

The code IS the policy.  There is no translation layer.

Three factory functions (`build_range_rule`, `build_required_field_rule`,
`build_allowed_values_rule`) cover the most common validation patterns without
requiring callers to write lambdas for routine cases.  Complex or cross-field
rules are expressed as ordinary Python callables.

## Consequences

**Benefits:**

- **Maximum transparency.** Every rule is a Python callable.  An auditor reads
  Python; they need no additional tooling or language knowledge to understand
  what a rule does.
- **Zero dependency overhead.** The engine installs in milliseconds.  No version
  conflicts, no CVE surface from rule-engine libraries.
- **Unlimited expressiveness.** Any Python expression can be a condition —
  cross-field checks, stateful lookups, probability thresholds, string pattern
  matching.  A DSL would require language extensions for the same.
- **Testable by construction.** Each condition is a callable; it can be unit
  tested directly by calling it with a dict.  No test doubles or policy
  simulation environments required.

**Trade-offs:**

- **Less structured than a DSL.** A dedicated policy language provides conflict
  detection, priority ordering, and formal semantics that plain Python lists do
  not.  For complex rule sets (hundreds of rules with priority interactions),
  a DSL would be more appropriate.
- **No GUI or policy management tooling.** External rule engines often ship with
  dashboards for non-technical stakeholders to view and manage rules.  With
  pure Python, the policy lives in source code and is managed through version
  control.

**Alternatives considered:**

- Open Policy Agent / Rego — rejected.  Adds a sidecar process or WASM runtime.
  For a field guide implementation, this creates more setup friction than value.
- Pydantic validators — rejected.  Pydantic is primarily a data-shape library,
  not a policy engine.  Severity levels, audit trails, and verdict aggregation
  would require wrapping Pydantic in a custom layer anyway.
- PyKnow / Drools-style forward-chaining — rejected.  Forward-chaining inference
  is more powerful than required here and adds significant conceptual overhead
  for what is essentially a linear scan of independent conditions.
