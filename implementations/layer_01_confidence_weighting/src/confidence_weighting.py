"""
Layer 01: Inverse Confidence Weighting

Surface your weakest signal, not your average.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

This module provides a domain-agnostic implementation of inverse confidence
weighting. See the field guide (layers/01-confidence-weighting.md) for the
full explanation, production examples, and limitations.

Generalised from pda-platform/packages/agent-task-planning.
"""

from __future__ import annotations


def inverse_confidence_weight(field_confidences: dict[str, float]) -> float:
    """Compute an aggregate confidence score using inverse weighting.

    Fields with lower confidence are weighted more heavily, forcing the
    aggregate to reflect the weakest signal rather than masking it in a
    plain average.  A system that is 95% confident in nine fields but only
    30% confident in one critical field should *not* report 89% overall.

    The weight for each field is: ``weight = 2.0 - confidence``.
    This maps confidence in [0, 1] to weight in [1, 2] — a field with
    confidence 0.0 gets weight 2.0 (maximum influence) whilst a field with
    confidence 1.0 gets weight 1.0 (minimum influence).

    Args:
        field_confidences: Mapping of field name to confidence score.
            Each confidence value must be in the range [0.0, 1.0].

    Returns:
        Weighted aggregate confidence in [0.0, 1.0].
        Returns 0.0 for an empty mapping.

    Raises:
        ValueError: If any confidence value is outside [0.0, 1.0].

    Examples:
        >>> # All fields equally confident — returns the common value
        >>> inverse_confidence_weight({"a": 0.8, "b": 0.8, "c": 0.8})
        0.8

        >>> # One weak field drags the aggregate below a simple mean
        >>> result = inverse_confidence_weight({"strong": 0.9, "weak": 0.3})
        >>> result < (0.9 + 0.3) / 2  # True: aggregate < plain mean
        True

        >>> # Empty mapping returns 0.0
        >>> inverse_confidence_weight({})
        0.0
    """
    if not field_confidences:
        return 0.0

    for field, conf in field_confidences.items():
        if not (0.0 <= conf <= 1.0):
            raise ValueError(
                f"Confidence for field '{field}' must be in [0.0, 1.0], got {conf!r}."
            )

    weighted_sum = 0.0
    weight_sum = 0.0
    for conf in field_confidences.values():
        weight = 2.0 - conf  # Inverse weighting: low confidence → higher weight
        weighted_sum += conf * weight
        weight_sum += weight

    return weighted_sum / weight_sum if weight_sum > 0.0 else 0.0


def main() -> None:
    """Run a self-contained demonstration of inverse confidence weighting."""
    print("Layer 01: Inverse Confidence Weighting")
    print("=" * 40)
    print()

    # --- Scenario 1: All fields are equally confident ---
    uniform = {"field_a": 0.8, "field_b": 0.8, "field_c": 0.8}
    result_uniform = inverse_confidence_weight(uniform)
    print("Scenario 1 — Uniform confidence (0.8 across three fields):")
    print(f"  Aggregate: {result_uniform:.4f}  (expected ~ 0.8000)")
    print()

    # --- Scenario 2: One weak field among strong fields ---
    mixed = {
        "date":     0.95,
        "location": 0.92,
        "amount":   0.88,
        "category": 0.30,  # Low-confidence field
    }
    plain_mean = sum(mixed.values()) / len(mixed)
    result_mixed = inverse_confidence_weight(mixed)
    print("Scenario 2 — One weak field (category = 0.30):")
    for field, conf in mixed.items():
        print(f"  {field:>10}: {conf:.2f}")
    print(f"  Plain mean:              {plain_mean:.4f}")
    print(f"  Inverse-weighted result: {result_mixed:.4f}  (lower than mean)")
    print()

    # --- Scenario 3: All fields minimally confident ---
    low = {"x": 0.1, "y": 0.2, "z": 0.15}
    result_low = inverse_confidence_weight(low)
    print("Scenario 3 — Low confidence across all fields:")
    print(f"  Aggregate: {result_low:.4f}")
    print()

    # --- Scenario 4: Empty input ---
    result_empty = inverse_confidence_weight({})
    print("Scenario 4 — Empty input:")
    print(f"  Aggregate: {result_empty:.4f}  (expected 0.0)")
    print()

    print("See: layers/01-confidence-weighting.md")


if __name__ == "__main__":  # pragma: no cover
    main()
