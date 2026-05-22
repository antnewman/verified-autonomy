# ADR 002: Inverse Confidence Weighting Over Averaging

## Status

Accepted

## Context

When aggregating confidence scores across multiple model outputs, the standard approach is to average. This produces misleading results when one field is highly uncertain. A single uncertain field in nine confident ones produces an 87% overall score — a number that looks acceptable and hides the problem entirely.

## Decision

Weight toward the weakest signal (inverse confidence weighting) rather than averaging. This is a values choice about honesty over tidiness. A high aggregate score should not be achievable when any individual field is uncertain; the system should surface the problem, not conceal it.

## Consequences

**Benefits:**
- Surfaces the thing that matters — the field the system is least sure about.
- Routes uncertain cases to human review rather than allowing them to pass silently.

**Trade-offs:**
- Increases escalation rate, which reduces throughput.
- Assumes fields carry roughly equal importance unless additionally weighted. Deployments where fields have very different stakes may need to combine inverse confidence weighting with domain-specific importance weights.

**Alternatives considered:**
- Simple averaging — rejected because it masks single-field uncertainty behind a comfortable aggregate.
- Weighted averaging by field importance — a valid complement but does not address the fundamental honesty argument. A low-confidence field is still dangerous regardless of its nominal importance weight.
