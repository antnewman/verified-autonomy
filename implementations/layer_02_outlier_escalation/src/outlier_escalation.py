"""
Layer 02: Outlier Detection as Hard Escalation

Statistical disagreement is a signal, not noise.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

This module provides a domain-agnostic implementation of IQR-based outlier
detection with four-tier hard escalation routing.  See the field guide
(layers/02-outlier-escalation.md) for the full explanation, production
examples, and limitations.

Generalised from pda-platform/packages/agent-task-planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EscalationLevel(str, Enum):
    """Four-tier escalation routing.

    Ordered from lowest to highest urgency.  The escalation gate uses
    an OR condition: if *any* trigger fires the tier is activated.
    This means the system fails safe — it escalates when in doubt.
    """

    NONE = "none"
    """Confidence is high and no outliers detected.  No human review needed."""

    SPOT_CHECK = "spot_check"
    """Confidence is moderate (0.6 ≤ conf < 0.8).  Periodic sampling sufficient."""

    DETAILED_REVIEW = "detailed_review"
    """Confidence is low (0.4 ≤ conf < 0.6).  A human should read every output."""

    EXPERT_REQUIRED = "expert_required"
    """Confidence is very low (conf < 0.4) OR statistical outliers detected.
    Route to a domain expert before the output is acted upon."""


@dataclass(frozen=True)
class OutlierFlag:
    """A single field flagged as a statistical outlier."""

    label: str
    """Human-readable label identifying the flagged value."""

    value: float
    """The flagged numeric value."""

    consensus: float
    """The group consensus (median) against which the value was compared."""

    divergence: float
    """Normalised divergence: ``|value - consensus| / value_range``.
    Ranges from 0.0 (identical to consensus) to 1.0 (maximum spread)."""

    reason: str
    """Plain-English description of why the value was flagged."""


@dataclass
class DetectionResult:
    """Aggregated result of an outlier detection run."""

    escalation_level: EscalationLevel
    """Routing decision based on outlier presence and overall confidence."""

    consensus: float
    """Median of all input values — the agreed reference point."""

    confidence: float
    """The overall confidence score passed in by the caller."""

    outliers: list[OutlierFlag] = field(default_factory=list)
    """Flags for every value identified as a statistical outlier."""

    @property
    def has_outliers(self) -> bool:
        """True if at least one outlier was detected."""
        return len(self.outliers) > 0


def _compute_iqr(sorted_values: list[float]) -> tuple[float, float, float]:
    """Compute Q1, Q3, and IQR for a pre-sorted list of floats.

    Uses the lower-half / upper-half split method (inclusive median
    excluded from both halves when n is odd).

    Args:
        sorted_values: A list of floats, sorted ascending.  Must contain
            at least two elements.

    Returns:
        ``(q1, q3, iqr)`` where ``iqr = q3 - q1``.
    """
    n = len(sorted_values)
    mid = n // 2

    lower_half = sorted_values[:mid]
    upper_half = sorted_values[mid + (n % 2):]  # Skip median when n is odd

    def _median(vals: list[float]) -> float:
        k = len(vals)
        if k == 0:
            return 0.0
        half = k // 2
        if k % 2 == 1:
            return vals[half]
        return (vals[half - 1] + vals[half]) / 2.0

    q1 = _median(lower_half)
    q3 = _median(upper_half)
    return q1, q3, q3 - q1


def detect_outliers(
    values: list[float],
    labels: list[str] | None = None,
    confidence: float = 1.0,
) -> DetectionResult:
    """Detect statistical outliers and route to an appropriate escalation tier.

    Uses Tukey's IQR fence method: values outside ``[Q1 - 1.5×IQR,
    Q3 + 1.5×IQR]`` are flagged as outliers.  The escalation level is
    determined by an OR condition — outliers *or* low confidence each
    independently trigger escalation.

    Args:
        values: Numeric values to analyse.  Must contain at least two
            elements for outlier detection to be meaningful.
        labels: Optional human-readable labels for each value, used in
            the ``OutlierFlag.reason`` strings.  If omitted, positions
            are used (e.g. ``"value[0]"``).
        confidence: Overall confidence score in [0.0, 1.0].  Used to
            determine the baseline escalation tier independent of
            outlier detection.

    Returns:
        A :class:`DetectionResult` with the escalation level, consensus,
        and any flagged outliers.

    Raises:
        ValueError: If ``confidence`` is outside [0.0, 1.0].
        ValueError: If ``labels`` is provided but has a different length
            than ``values``.

    Examples:
        >>> result = detect_outliers([10.0, 11.0, 10.5, 50.0], confidence=0.9)
        >>> result.escalation_level
        <EscalationLevel.EXPERT_REQUIRED: 'expert_required'>
        >>> len(result.outliers)
        1

        >>> result = detect_outliers([10.0, 11.0, 10.5], confidence=0.9)
        >>> result.escalation_level
        <EscalationLevel.NONE: 'none'>
    """
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(
            f"confidence must be in [0.0, 1.0], got {confidence!r}."
        )
    if labels is not None and len(labels) != len(values):
        raise ValueError(
            f"labels length ({len(labels)}) must match values length ({len(values)})."
        )

    effective_labels = labels if labels is not None else [f"value[{i}]" for i in range(len(values))]

    # --- Consensus ---
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n == 0:
        consensus = 0.0
    elif n % 2 == 1:
        consensus = sorted_vals[mid]
    else:
        consensus = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

    # --- Outlier detection ---
    flagged: list[OutlierFlag] = []

    if n >= 2:
        q1, q3, iqr = _compute_iqr(sorted_vals)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        value_range = sorted_vals[-1] - sorted_vals[0]

        for label, value in zip(effective_labels, values):
            if value < lower_bound or value > upper_bound:
                if value_range > 0.0:
                    divergence = abs(value - consensus) / value_range
                else:
                    divergence = 0.0

                if value < lower_bound:
                    reason = (
                        f"{label} ({value:.4g}) is below the lower fence "
                        f"({lower_bound:.4g}); consensus is {consensus:.4g}"
                    )
                else:
                    reason = (
                        f"{label} ({value:.4g}) is above the upper fence "
                        f"({upper_bound:.4g}); consensus is {consensus:.4g}"
                    )

                flagged.append(
                    OutlierFlag(
                        label=label,
                        value=value,
                        consensus=consensus,
                        divergence=round(divergence, 6),
                        reason=reason,
                    )
                )

    # --- Escalation routing (OR condition — fails safe) ---
    if flagged or confidence < 0.4:
        level = EscalationLevel.EXPERT_REQUIRED
    elif confidence < 0.6:
        level = EscalationLevel.DETAILED_REVIEW
    elif confidence < 0.8:
        level = EscalationLevel.SPOT_CHECK
    else:
        level = EscalationLevel.NONE

    return DetectionResult(
        escalation_level=level,
        consensus=consensus,
        confidence=confidence,
        outliers=flagged,
    )


def main() -> None:
    """Run a self-contained demonstration of IQR outlier detection."""
    print("Layer 02: Outlier Detection as Hard Escalation")
    print("=" * 48)
    print()

    # --- Scenario 1: Clean data, high confidence → NONE ---
    clean = [10.2, 10.5, 10.1, 10.4, 10.3]
    r1 = detect_outliers(clean, confidence=0.92)
    print("Scenario 1 — Clean data, high confidence:")
    print(f"  Values:     {clean}")
    print(f"  Consensus:  {r1.consensus:.2f}")
    print(f"  Escalation: {r1.escalation_level.value}")
    print(f"  Outliers:   {len(r1.outliers)}")
    print()

    # --- Scenario 2: One outlier → EXPERT_REQUIRED ---
    # Six values required: with only 5, the outlier inflates Q3 and widens
    # the IQR fence past itself.  Six identical + one extreme guarantees IQR=0.
    with_outlier = [10.1, 10.2, 10.0, 10.1, 10.0, 80.0]
    labels = ["agent_A", "agent_B", "agent_C", "agent_D", "agent_E", "agent_F"]
    r2 = detect_outliers(with_outlier, labels=labels, confidence=0.85)
    print("Scenario 2 — One outlier, otherwise high confidence:")
    for lbl, val in zip(labels, with_outlier):
        marker = " <-- OUTLIER" if any(o.label == lbl for o in r2.outliers) else ""
        print(f"  {lbl}: {val}{marker}")
    print(f"  Consensus:  {r2.consensus:.2f}")
    print(f"  Escalation: {r2.escalation_level.value}  (outlier alone triggers EXPERT_REQUIRED)")
    print()

    # --- Scenario 3: Low confidence alone → EXPERT_REQUIRED ---
    r3 = detect_outliers([10.0, 10.5, 11.0], confidence=0.35)
    print("Scenario 3 — Clean data, very low confidence (0.35):")
    print(f"  Escalation: {r3.escalation_level.value}  (conf < 0.4 triggers EXPERT_REQUIRED)")
    print()

    # --- Scenario 4: Moderate confidence → DETAILED_REVIEW ---
    r4 = detect_outliers([1.0, 2.0, 1.5], confidence=0.55)
    print("Scenario 4 — Moderate confidence (0.55):")
    print(f"  Escalation: {r4.escalation_level.value}")
    print()

    print("See: layers/02-outlier-escalation.md")


if __name__ == "__main__":  # pragma: no cover
    main()
