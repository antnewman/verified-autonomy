"""
Layer 03: Making Failures Visible

The potential hallucinations list.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

This module provides a domain-agnostic implementation of the potential
hallucinations list — a technique for making model failures explicit and
visible to downstream consumers.  See the field guide
(layers/03-visible-failures.md) for the full explanation, production
examples, and limitations.

Generalised from pda-platform/packages/agent-task-planning.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityScore:
    """Quality assessment for a single model-generated sample.

    Attributes:
        coherence: How internally consistent the output is (0.0–1.0).
        relevance: How on-topic the output is relative to the prompt (0.0–1.0).
        completeness: How fully the output addresses the request (0.0–1.0).
        semantic_entropy: A measure of output unpredictability across samples.
            Higher values indicate more variance — a proxy for hallucination
            risk.  Typical range 0.0–1.0, though not strictly bounded.
        overall: Weighted composite quality score after entropy penalty.
        passed_threshold: True if ``overall`` meets the minimum quality bar.
    """

    coherence: float
    relevance: float
    completeness: float
    semantic_entropy: float
    overall: float
    passed_threshold: bool


def compute_quality_score(
    coherence: float,
    relevance: float,
    completeness: float,
    semantic_entropy: float,
    threshold: float = 0.6,
) -> QualityScore:
    """Compute a quality score for a single model-generated sample.

    The composite score weights coherence and relevance at 40% each and
    completeness at 20%, then subtracts an entropy penalty for high-variance
    outputs:

    .. code-block::

        entropy_penalty = max(0, semantic_entropy - 0.5) * 0.5
        overall = (coherence×0.4 + relevance×0.4 + completeness×0.2)
                  - entropy_penalty

    The entropy penalty activates only above 0.5 — a modest amount of
    semantic variation is tolerated.  At maximum entropy (1.0), the penalty
    is 0.25, sufficient to push a borderline output below the default
    threshold of 0.6.

    Args:
        coherence: Internal consistency of the output.  Must be in [0.0, 1.0].
        relevance: On-topic quality of the output.  Must be in [0.0, 1.0].
        completeness: How fully the request was addressed.  Must be in [0.0, 1.0].
        semantic_entropy: Cross-sample variance proxy.  Values above 0.5
            incur a quality penalty.
        threshold: Minimum ``overall`` score to pass.  Defaults to 0.6.

    Returns:
        A :class:`QualityScore` with the computed overall score and pass/fail
        result.

    Raises:
        ValueError: If ``coherence``, ``relevance``, or ``completeness`` are
            outside [0.0, 1.0], or if ``semantic_entropy`` is negative.

    Examples:
        >>> score = compute_quality_score(0.8, 0.9, 0.7, 0.3)
        >>> score.passed_threshold
        True

        >>> # High entropy penalises an otherwise acceptable output
        >>> score = compute_quality_score(0.7, 0.7, 0.6, 0.9)
        >>> score.overall < 0.6
        True
    """
    for name, val in [("coherence", coherence), ("relevance", relevance), ("completeness", completeness)]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"'{name}' must be in [0.0, 1.0], got {val!r}.")
    if semantic_entropy < 0.0:
        raise ValueError(f"'semantic_entropy' must be non-negative, got {semantic_entropy!r}.")

    weighted = coherence * 0.4 + relevance * 0.4 + completeness * 0.2
    entropy_penalty = max(0.0, semantic_entropy - 0.5) * 0.5
    overall = weighted - entropy_penalty

    return QualityScore(
        coherence=coherence,
        relevance=relevance,
        completeness=completeness,
        semantic_entropy=semantic_entropy,
        overall=round(overall, 6),
        passed_threshold=overall >= threshold,
    )


@dataclass(frozen=True)
class Sample:
    """A single model-generated sample with identity and quality information."""

    id: str
    """Unique identifier for this sample (e.g. a UUID or sequence number)."""

    quality: QualityScore
    """Quality assessment for this sample."""


def flag_potential_hallucinations(
    samples: list[Sample],
    entropy_threshold: float = 0.7,
) -> list[str]:
    """Return the IDs of samples at elevated hallucination risk.

    A sample is flagged when its ``semantic_entropy`` exceeds
    ``entropy_threshold``.  High entropy indicates the model produced
    markedly different outputs across repeated runs for the same prompt —
    a reliable proxy for confabulation.

    This list is intended to be published alongside model outputs, not
    to silently suppress them.  Callers should surface the flag to
    downstream consumers rather than filtering the sample out.

    Args:
        samples: The candidate samples to inspect.
        entropy_threshold: Entropy above this value triggers a flag.
            Defaults to 0.7.

    Returns:
        List of ``Sample.id`` values for samples at hallucination risk.
        Returns an empty list if no samples are flagged.

    Raises:
        ValueError: If ``entropy_threshold`` is not in [0.0, 1.0].

    Examples:
        >>> from verified_autonomy.visible_failures import (
        ...     Sample, QualityScore, flag_potential_hallucinations
        ... )
        >>> score = QualityScore(0.8, 0.8, 0.8, 0.9, 0.72, True)
        >>> s = Sample(id="abc", quality=score)
        >>> flag_potential_hallucinations([s])
        ['abc']
    """
    if not (0.0 <= entropy_threshold <= 1.0):
        raise ValueError(
            f"entropy_threshold must be in [0.0, 1.0], got {entropy_threshold!r}."
        )
    return [s.id for s in samples if s.quality.semantic_entropy > entropy_threshold]


def compute_pass_rate(samples: list[Sample]) -> float:
    """Compute the fraction of samples that passed the quality threshold.

    Args:
        samples: The evaluated samples.

    Returns:
        Pass rate in [0.0, 1.0].  Returns 0.0 for an empty list.

    Examples:
        >>> # Build two samples, one passing and one failing
        >>> passing = compute_quality_score(0.9, 0.9, 0.9, 0.1)
        >>> failing = compute_quality_score(0.2, 0.2, 0.2, 0.9)
        >>> s1 = Sample(id="s1", quality=passing)
        >>> s2 = Sample(id="s2", quality=failing)
        >>> compute_pass_rate([s1, s2])
        0.5
    """
    if not samples:
        return 0.0
    passed = sum(1 for s in samples if s.quality.passed_threshold)
    return passed / len(samples)


def main() -> None:
    """Run a self-contained demonstration of the potential hallucinations list."""
    print("Layer 03: Making Failures Visible")
    print("=" * 36)
    print()

    samples_raw = [
        # (id, coherence, relevance, completeness, entropy)
        ("sample_01", 0.90, 0.88, 0.85, 0.20),  # Clean
        ("sample_02", 0.75, 0.80, 0.70, 0.45),  # Borderline entropy
        ("sample_03", 0.60, 0.65, 0.55, 0.75),  # High entropy → hallucination risk
        ("sample_04", 0.85, 0.82, 0.78, 0.30),  # Clean
        ("sample_05", 0.40, 0.35, 0.50, 0.92),  # High entropy + low quality
    ]

    samples = [
        Sample(id=sid, quality=compute_quality_score(coh, rel, com, ent))
        for sid, coh, rel, com, ent in samples_raw
    ]

    print("Sample quality scores:")
    print(f"  {'ID':<12} {'Overall':>7}  {'Pass?':>5}  {'Entropy':>7}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*5}  {'-'*7}")
    for s in samples:
        q = s.quality
        print(
            f"  {s.id:<12}  {q.overall:>7.4f}  {'yes' if q.passed_threshold else 'no':>5}  {q.semantic_entropy:>7.2f}"
        )
    print()

    pass_rate = compute_pass_rate(samples)
    print(f"Quality pass rate: {pass_rate:.0%}  ({sum(1 for s in samples if s.quality.passed_threshold)}/{len(samples)} samples)")
    print()

    flagged = flag_potential_hallucinations(samples)
    if flagged:
        print(f"Potential hallucinations ({len(flagged)} flagged):")
        for fid in flagged:
            print(f"  - {fid}")
    else:
        print("No hallucinations flagged.")
    print()

    print("See: layers/03-visible-failures.md")


if __name__ == "__main__":  # pragma: no cover
    main()
