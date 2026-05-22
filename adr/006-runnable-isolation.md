# ADR 006: Runnable Isolation for All Code Examples

## Status

Accepted

## Context

Most technical guides either describe techniques abstractly — with no runnable code at all — or provide code that requires cloning an entire repository, resolving a complex shared dependency tree, and navigating an unfamiliar project structure before a single example can be run. Both approaches create friction. Friction prevents adoption. A reader who cannot run the code in two minutes will not run it at all.

## Decision

Every layer's code example must run with a single command from its own directory, with no dependency on any other layer's code:

```bash
cd implementations/layer_01_confidence_weighting
make install
make run
```

Each implementation has its own `pyproject.toml` with minimal, self-contained dependencies; its own `Makefile` with standardised targets; and its own `README.md` with expected output documented. The reader goes from the field guide to a working terminal in under two minutes. This is non-negotiable.

## Consequences

**Benefits:**
- Dramatically lower barrier to trying a technique. A reader can evaluate any single layer without understanding the rest of the framework.
- Each layer is independently adoptable. A practitioner can lift a single implementation into their own codebase without taking a dependency on the full project.
- The Makefile contract (`install`, `test`, `run`, `lint`, `clean`) is predictable across all nine layers, reducing cognitive load for contributors.

**Trade-offs:**
- Some code duplication across layers. Shared utility logic cannot be factored into a common library without creating cross-layer dependencies, which violates the isolation principle.
- Each layer must maintain its own dependency set. A dependency update in one layer does not automatically propagate to others.

**Alternatives considered:**
- Shared utility package imported by all layers — rejected because it creates an implicit cross-layer dependency. A reader running layer 03 in isolation would also need to install the shared utility package, adding setup steps and defeating the two-minute goal.
- Monolithic implementation with a single entry point per layer — rejected because it requires the reader to navigate the full repository structure before running anything.
