# ADR 001: Nine-Layer Defence-in-Depth Structure

## Status

Accepted

## Context

The field guide needed a structure that was simultaneously comprehensive enough to become a white paper, practical enough to be useful to a practitioner, and distinctive enough to stand alone as a reference.

Alternatives considered: companion guide, practical toolkit, debate piece, state-of-the-field survey. Each was rejected for being either too narrow in scope, too abstract to be actionable, or too transient to serve as a lasting reference.

## Decision

Organise the guide around nine independent layers that build on each other:

- **Layers 1–3:** How a system reports its own uncertainty
- **Layers 4–6:** The architecture around the model
- **Layers 7–9:** Verification, provenance, and proof

The central argument is that trust is an emergent property of the complete architecture — no single layer is sufficient.

## Consequences

**Benefits:**
- Each layer can be read independently, contributed to independently, and updated independently.
- The layered structure maps naturally to a video series and a white paper structure.
- Contributors can own a single layer without needing to understand the full framework.

**Trade-offs:**
- The nine-layer count is somewhat arbitrary; real systems may need more or fewer layers.
- The structure may need to evolve as the field matures.

**Alternatives considered:**
- Companion guide — too dependent on a primary resource that does not yet exist.
- Practical toolkit — too narrow; misses the conceptual framing that makes the techniques coherent.
- Debate piece — too transient; argumentation ages poorly.
- State-of-the-field survey — too passive; does not give practitioners actionable guidance.
