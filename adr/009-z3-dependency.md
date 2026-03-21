# ADR 009: Allow z3-solver as a Runtime Dependency for Layer 09

Status: Accepted

## Context

Layer 09 demonstrates formal verification of neural network properties. Without
an actual SMT solver, the implementation would be either a brute-force
enumeration (not formal verification) or a simulation (not a proof). Z3 is the
standard SMT solver used in academic and production formal verification.

## Decision

Allow z3-solver as a runtime dependency for Layer 09.

## Consequences

**Benefits:**
- Real formal verification with real proofs
- Counterexamples are genuine — they are actual inputs that violate the property
- The state-space explosion is demonstrable with real timing data
- Z3 is the industry standard tool for this problem

**Trade-offs:**
- z3-solver is a compiled package with platform-specific binaries
- Slower to install than pure-Python packages
- Less widely installed than numpy or scikit-learn

**Mitigation:**
- z3-solver is pip-installable on all major platforms (Windows, macOS, Linux)
- The package is well-maintained and widely used in the formal methods community
- Install time is a one-off cost documented in the README
