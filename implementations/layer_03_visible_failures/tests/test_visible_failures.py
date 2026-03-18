"""
Tests for Layer 03: Making Failures Visible

Test naming convention: test_[function]_[scenario]_[expected_result]
Example: test_flag_hallucinations_returns_empty_list_when_all_outputs_pass_threshold

Requirements (see CONTRIBUTING.md):
- 90% minimum coverage
- Edge cases at both boundaries
- Corner cases at intersections of multiple boundaries
- Failure mode tests (correct exceptions, messages, fallbacks)
- No mocking of the system under test
"""

from __future__ import annotations

import math

import pytest

from visible_failures import (
    QualityScore,
    Sample,
    compute_pass_rate,
    compute_quality_score,
    flag_potential_hallucinations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(sid: str, coherence: float, relevance: float, completeness: float, entropy: float) -> Sample:
    return Sample(id=sid, quality=compute_quality_score(coherence, relevance, completeness, entropy))


# ---------------------------------------------------------------------------
# compute_quality_score
# ---------------------------------------------------------------------------

class TestComputeQualityScoreBasic:
    """Basic correctness of the weighted formula and entropy penalty."""

    def test_high_quality_no_entropy_passes_threshold(self) -> None:
        score = compute_quality_score(0.9, 0.9, 0.9, 0.0)
        assert score.passed_threshold is True
        assert score.overall > 0.6

    def test_low_quality_fails_threshold(self) -> None:
        score = compute_quality_score(0.2, 0.2, 0.2, 0.0)
        assert score.passed_threshold is False

    def test_entropy_penalty_activates_above_half(self) -> None:
        """Entropy at 0.5 incurs no penalty; entropy at 1.0 incurs 0.25."""
        score_low_entropy = compute_quality_score(0.8, 0.8, 0.8, 0.5)
        score_high_entropy = compute_quality_score(0.8, 0.8, 0.8, 1.0)
        assert score_low_entropy.overall > score_high_entropy.overall

    def test_entropy_below_half_incurs_no_penalty(self) -> None:
        """entropy=0.3 and entropy=0.5 should produce the same overall score."""
        s1 = compute_quality_score(0.7, 0.7, 0.7, 0.3)
        s2 = compute_quality_score(0.7, 0.7, 0.7, 0.5)
        assert s1.overall == pytest.approx(s2.overall)

    def test_max_entropy_penalty_is_quarter(self) -> None:
        """At entropy=1.0: penalty = max(0, 1.0-0.5)*0.5 = 0.25."""
        s_no_ent = compute_quality_score(0.8, 0.8, 0.8, 0.0)
        s_max_ent = compute_quality_score(0.8, 0.8, 0.8, 1.0)
        assert s_no_ent.overall - s_max_ent.overall == pytest.approx(0.25)

    def test_formula_coherence_relevance_completeness_weights(self) -> None:
        """overall = coherence*0.4 + relevance*0.4 + completeness*0.2 (no penalty)."""
        expected = 0.6 * 0.4 + 0.7 * 0.4 + 0.8 * 0.2
        score = compute_quality_score(0.6, 0.7, 0.8, 0.0)
        assert score.overall == pytest.approx(expected)

    def test_returned_fields_match_inputs(self) -> None:
        score = compute_quality_score(0.5, 0.6, 0.7, 0.3)
        assert score.coherence == pytest.approx(0.5)
        assert score.relevance == pytest.approx(0.6)
        assert score.completeness == pytest.approx(0.7)
        assert score.semantic_entropy == pytest.approx(0.3)

    def test_result_is_frozen_dataclass(self) -> None:
        score = compute_quality_score(0.8, 0.8, 0.8, 0.2)
        with pytest.raises((AttributeError, TypeError)):
            score.overall = 99.0  # type: ignore[misc]


class TestComputeQualityScoreBoundary:
    """Edge cases at the boundary of the default threshold (0.6)."""

    def test_overall_exactly_at_threshold_passes(self) -> None:
        """A score of exactly 0.6 must pass."""
        # Need overall == 0.6 with entropy=0
        # 0.6 = c*0.4 + r*0.4 + m*0.2; with c=r=m=0.6 → 0.6
        score = compute_quality_score(0.6, 0.6, 0.6, 0.0)
        assert score.passed_threshold is True
        assert score.overall == pytest.approx(0.6)

    def test_overall_just_below_threshold_fails(self) -> None:
        score = compute_quality_score(0.59, 0.59, 0.59, 0.0)
        assert score.passed_threshold is False

    def test_high_entropy_pushes_borderline_below_threshold(self) -> None:
        # Without entropy: overall ≈ 0.65, but with entropy=1.0: overall ≈ 0.40
        score = compute_quality_score(0.7, 0.7, 0.6, 0.9)
        assert score.passed_threshold is False

    def test_custom_threshold_respected(self) -> None:
        score = compute_quality_score(0.5, 0.5, 0.5, 0.0, threshold=0.4)
        assert score.passed_threshold is True

    def test_custom_threshold_higher_than_default(self) -> None:
        score = compute_quality_score(0.7, 0.7, 0.7, 0.0, threshold=0.75)
        assert score.passed_threshold is False


class TestComputeQualityScoreValidation:
    """Failure mode: invalid inputs raise ValueError."""

    def test_coherence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="coherence"):
            compute_quality_score(1.01, 0.5, 0.5, 0.0)

    def test_coherence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="coherence"):
            compute_quality_score(-0.01, 0.5, 0.5, 0.0)

    def test_relevance_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="relevance"):
            compute_quality_score(0.5, 1.5, 0.5, 0.0)

    def test_completeness_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="completeness"):
            compute_quality_score(0.5, 0.5, -0.1, 0.0)

    def test_negative_entropy_raises(self) -> None:
        with pytest.raises(ValueError, match="semantic_entropy"):
            compute_quality_score(0.5, 0.5, 0.5, -0.1)

    def test_zero_entropy_is_valid(self) -> None:
        score = compute_quality_score(0.8, 0.8, 0.8, 0.0)
        assert isinstance(score, QualityScore)

    def test_overall_is_finite(self) -> None:
        score = compute_quality_score(0.5, 0.5, 0.5, 0.5)
        assert math.isfinite(score.overall)


# ---------------------------------------------------------------------------
# flag_potential_hallucinations
# ---------------------------------------------------------------------------

class TestFlagPotentialHallucinations:
    """Core behaviour of the hallucination flagger."""

    def test_empty_samples_returns_empty_list(self) -> None:
        assert flag_potential_hallucinations([]) == []

    def test_all_samples_below_threshold_returns_empty(self) -> None:
        samples = [_make_sample("s1", 0.8, 0.8, 0.8, 0.3), _make_sample("s2", 0.7, 0.7, 0.7, 0.5)]
        assert flag_potential_hallucinations(samples) == []

    def test_sample_above_threshold_is_flagged(self) -> None:
        samples = [_make_sample("risky", 0.7, 0.7, 0.7, 0.8)]
        assert flag_potential_hallucinations(samples) == ["risky"]

    def test_only_high_entropy_samples_flagged(self) -> None:
        samples = [
            _make_sample("clean", 0.9, 0.9, 0.9, 0.2),
            _make_sample("risky", 0.6, 0.6, 0.6, 0.75),
        ]
        result = flag_potential_hallucinations(samples)
        assert "risky" in result
        assert "clean" not in result

    def test_multiple_high_entropy_samples_all_flagged(self) -> None:
        samples = [
            _make_sample("a", 0.7, 0.7, 0.7, 0.8),
            _make_sample("b", 0.6, 0.6, 0.6, 0.9),
            _make_sample("c", 0.8, 0.8, 0.8, 0.3),
        ]
        result = flag_potential_hallucinations(samples)
        assert set(result) == {"a", "b"}

    def test_returns_sample_ids_not_objects(self) -> None:
        samples = [_make_sample("id_42", 0.5, 0.5, 0.5, 0.8)]
        result = flag_potential_hallucinations(samples)
        assert result == ["id_42"]
        assert all(isinstance(r, str) for r in result)


class TestFlagPotentialHallucinationsBoundary:
    """Boundary: threshold is exclusive (> not >=)."""

    def test_entropy_exactly_at_threshold_is_not_flagged(self) -> None:
        """Entropy equal to threshold does not trigger the flag."""
        samples = [_make_sample("s", 0.7, 0.7, 0.7, 0.7)]
        assert flag_potential_hallucinations(samples, entropy_threshold=0.7) == []

    def test_entropy_just_above_threshold_is_flagged(self) -> None:
        samples = [_make_sample("s", 0.7, 0.7, 0.7, 0.701)]
        result = flag_potential_hallucinations(samples, entropy_threshold=0.7)
        assert "s" in result

    def test_custom_threshold_zero_flags_any_nonzero_entropy(self) -> None:
        samples = [_make_sample("s", 0.8, 0.8, 0.8, 0.01)]
        result = flag_potential_hallucinations(samples, entropy_threshold=0.0)
        assert "s" in result

    def test_custom_threshold_one_flags_nothing_in_normal_range(self) -> None:
        """With threshold=1.0 and entropy ≤ 1.0, nothing is flagged."""
        samples = [_make_sample("s", 0.5, 0.5, 0.5, 1.0)]
        result = flag_potential_hallucinations(samples, entropy_threshold=1.0)
        assert result == []


class TestFlagPotentialHallucinationsValidation:
    """Failure mode: invalid entropy_threshold raises ValueError."""

    def test_threshold_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="entropy_threshold"):
            flag_potential_hallucinations([], entropy_threshold=1.01)

    def test_threshold_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="entropy_threshold"):
            flag_potential_hallucinations([], entropy_threshold=-0.01)

    def test_threshold_exactly_zero_is_valid(self) -> None:
        result = flag_potential_hallucinations([], entropy_threshold=0.0)
        assert result == []

    def test_threshold_exactly_one_is_valid(self) -> None:
        result = flag_potential_hallucinations([], entropy_threshold=1.0)
        assert result == []


# ---------------------------------------------------------------------------
# compute_pass_rate
# ---------------------------------------------------------------------------

class TestComputePassRate:
    """Pass rate calculation."""

    def test_empty_list_returns_zero(self) -> None:
        assert compute_pass_rate([]) == 0.0

    def test_all_pass_returns_one(self) -> None:
        samples = [_make_sample(f"s{i}", 0.9, 0.9, 0.9, 0.1) for i in range(5)]
        assert compute_pass_rate(samples) == pytest.approx(1.0)

    def test_all_fail_returns_zero(self) -> None:
        samples = [_make_sample(f"s{i}", 0.1, 0.1, 0.1, 0.9) for i in range(5)]
        assert compute_pass_rate(samples) == pytest.approx(0.0)

    def test_half_pass_returns_half(self) -> None:
        passing = _make_sample("p", 0.9, 0.9, 0.9, 0.1)
        failing = _make_sample("f", 0.1, 0.1, 0.1, 0.9)
        assert compute_pass_rate([passing, failing]) == pytest.approx(0.5)

    def test_single_passing_sample_returns_one(self) -> None:
        sample = _make_sample("s", 0.9, 0.9, 0.9, 0.1)
        assert compute_pass_rate([sample]) == pytest.approx(1.0)

    def test_single_failing_sample_returns_zero(self) -> None:
        sample = _make_sample("s", 0.1, 0.1, 0.1, 0.9)
        assert compute_pass_rate([sample]) == pytest.approx(0.0)

    def test_return_value_is_between_zero_and_one(self) -> None:
        samples = [
            _make_sample("a", 0.9, 0.9, 0.9, 0.1),
            _make_sample("b", 0.5, 0.5, 0.5, 0.4),
            _make_sample("c", 0.1, 0.1, 0.1, 0.9),
        ]
        rate = compute_pass_rate(samples)
        assert 0.0 <= rate <= 1.0

    def test_result_is_float(self) -> None:
        samples = [_make_sample("s", 0.8, 0.8, 0.8, 0.2)]
        assert isinstance(compute_pass_rate(samples), float)


class TestMain:
    """The main() demonstration function runs without error."""

    def test_main_runs_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        from visible_failures import main
        main()
        captured = capsys.readouterr()
        assert "Layer 03" in captured.out
        assert "Making Failures Visible" in captured.out

    def test_main_output_shows_pass_rate(self, capsys: pytest.CaptureFixture[str]) -> None:
        from visible_failures import main
        main()
        captured = capsys.readouterr()
        assert "pass rate" in captured.out.lower()

    def test_main_output_shows_flagged_samples(self, capsys: pytest.CaptureFixture[str]) -> None:
        from visible_failures import main
        main()
        captured = capsys.readouterr()
        assert "hallucination" in captured.out.lower()
