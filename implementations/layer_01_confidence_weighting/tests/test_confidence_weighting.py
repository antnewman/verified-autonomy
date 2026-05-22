"""
Tests for Layer 01: Inverse Confidence Weighting

Test naming convention: test_[function]_[scenario]_[expected_result]
Example: test_inverse_confidence_weight_single_field_returns_that_value

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

from confidence_weighting import inverse_confidence_weight


class TestInverseConfidenceWeightEmptyAndSingle:
    """Edge cases: empty input and single-field inputs."""

    def test_empty_dict_returns_zero(self) -> None:
        assert inverse_confidence_weight({}) == 0.0

    def test_single_field_confidence_zero_returns_zero(self) -> None:
        result = inverse_confidence_weight({"field": 0.0})
        assert result == pytest.approx(0.0)

    def test_single_field_confidence_one_returns_one(self) -> None:
        result = inverse_confidence_weight({"field": 1.0})
        assert result == pytest.approx(1.0)

    def test_single_field_confidence_half_returns_half(self) -> None:
        result = inverse_confidence_weight({"field": 0.5})
        assert result == pytest.approx(0.5)

    def test_single_field_arbitrary_confidence_returns_that_value(self) -> None:
        # With one field, weighted_sum/weight_sum = conf * w / w = conf
        for conf in [0.1, 0.3, 0.73, 0.99]:
            result = inverse_confidence_weight({"x": conf})
            assert result == pytest.approx(conf), f"Failed for conf={conf}"


class TestInverseConfidenceWeightUniformInputs:
    """If all fields share the same confidence, the aggregate equals that value."""

    def test_uniform_confidence_returns_common_value(self) -> None:
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            fields = {f"f{i}": conf for i in range(5)}
            result = inverse_confidence_weight(fields)
            assert result == pytest.approx(conf), f"Failed for conf={conf}"

    def test_uniform_low_confidence_returns_low_value(self) -> None:
        result = inverse_confidence_weight({"a": 0.1, "b": 0.1, "c": 0.1})
        assert result == pytest.approx(0.1)

    def test_uniform_high_confidence_returns_high_value(self) -> None:
        result = inverse_confidence_weight({"a": 0.9, "b": 0.9})
        assert result == pytest.approx(0.9)


class TestInverseConfidenceWeightWeakFieldEffect:
    """The aggregate should be pulled toward weak (low-confidence) fields."""

    def test_one_weak_field_pulls_aggregate_below_plain_mean(self) -> None:
        fields = {"strong_1": 0.9, "strong_2": 0.9, "weak": 0.3}
        plain_mean = sum(fields.values()) / len(fields)
        result = inverse_confidence_weight(fields)
        assert result < plain_mean

    def test_very_weak_field_dominates_aggregate(self) -> None:
        """A near-zero confidence field should drag the aggregate well below the mean."""
        fields = {"a": 0.95, "b": 0.92, "c": 0.01}
        plain_mean = sum(fields.values()) / len(fields)
        result = inverse_confidence_weight(fields)
        # The result should be noticeably below the plain mean
        assert result < plain_mean - 0.05

    def test_aggregate_is_bounded_by_min_and_max_input(self) -> None:
        """The weighted result must lie within the range of inputs."""
        fields = {"low": 0.2, "mid": 0.6, "high": 0.9}
        result = inverse_confidence_weight(fields)
        assert min(fields.values()) <= result <= max(fields.values())

    def test_two_fields_equal_weight_when_both_same_confidence(self) -> None:
        result = inverse_confidence_weight({"x": 0.7, "y": 0.7})
        assert result == pytest.approx(0.7)

    def test_weak_field_receives_higher_weight_than_strong(self) -> None:
        """Verify the weight formula: weight = 2.0 - conf."""
        # conf=0.2 → weight=1.8;  conf=0.8 → weight=1.2
        # weighted aggregate = (0.2*1.8 + 0.8*1.2) / (1.8 + 1.2)
        expected = (0.2 * 1.8 + 0.8 * 1.2) / (1.8 + 1.2)
        result = inverse_confidence_weight({"low": 0.2, "high": 0.8})
        assert result == pytest.approx(expected)


class TestInverseConfidenceWeightBoundaryValues:
    """Corner cases at the exact boundaries of the [0, 1] confidence range."""

    def test_all_fields_at_zero_returns_zero(self) -> None:
        result = inverse_confidence_weight({"a": 0.0, "b": 0.0})
        assert result == pytest.approx(0.0)

    def test_all_fields_at_one_returns_one(self) -> None:
        result = inverse_confidence_weight({"a": 1.0, "b": 1.0, "c": 1.0})
        assert result == pytest.approx(1.0)

    def test_mix_of_zero_and_one_pulls_toward_zero(self) -> None:
        """zero conf gets weight 2.0, one conf gets weight 1.0; aggregate < 0.5."""
        result = inverse_confidence_weight({"zero": 0.0, "one": 1.0})
        # weighted = (0*2 + 1*1) / 3 = 1/3
        assert result == pytest.approx(1.0 / 3.0)

    def test_result_is_non_negative(self) -> None:
        result = inverse_confidence_weight({"a": 0.0, "b": 0.01, "c": 0.02})
        assert result >= 0.0

    def test_result_does_not_exceed_one(self) -> None:
        result = inverse_confidence_weight({"a": 0.99, "b": 1.0})
        assert result <= 1.0


class TestInverseConfidenceWeightReturnType:
    """The return value must always be a plain Python float."""

    def test_returns_float(self) -> None:
        result = inverse_confidence_weight({"a": 0.5})
        assert isinstance(result, float)

    def test_returns_float_for_empty_input(self) -> None:
        result = inverse_confidence_weight({})
        assert isinstance(result, float)

    def test_result_is_finite(self) -> None:
        result = inverse_confidence_weight({"a": 0.5, "b": 0.7})
        assert math.isfinite(result)


class TestInverseConfidenceWeightValidation:
    """Failure mode: invalid confidence values raise ValueError."""

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="field_x"):
            inverse_confidence_weight({"field_x": 1.01})

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="field_y"):
            inverse_confidence_weight({"field_y": -0.01})

    def test_confidence_exactly_negative_one_raises(self) -> None:
        with pytest.raises(ValueError):
            inverse_confidence_weight({"bad": -1.0})

    def test_valid_field_before_invalid_still_raises(self) -> None:
        """All fields are validated — bad fields mid-dict are not silently skipped."""
        with pytest.raises(ValueError):
            inverse_confidence_weight({"ok": 0.8, "bad": 1.5})

    def test_error_message_includes_field_name(self) -> None:
        with pytest.raises(ValueError, match="my_field"):
            inverse_confidence_weight({"my_field": 2.0})


class TestMain:
    """The main() demonstration function runs without error."""

    def test_main_runs_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        from confidence_weighting import main
        main()
        captured = capsys.readouterr()
        assert "Layer 01" in captured.out
        assert "Inverse Confidence Weighting" in captured.out

    def test_main_output_contains_all_scenarios(self, capsys: pytest.CaptureFixture[str]) -> None:
        from confidence_weighting import main
        main()
        captured = capsys.readouterr()
        assert "Scenario 1" in captured.out
        assert "Scenario 2" in captured.out
        assert "Scenario 3" in captured.out
        assert "Scenario 4" in captured.out

    def test_main_output_shows_empty_result_as_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        from confidence_weighting import main
        main()
        captured = capsys.readouterr()
        assert "0.0000" in captured.out


class TestInverseConfidenceWeightLargeInputs:
    """Regression: large numbers of fields should not overflow or underflow."""

    def test_100_uniform_fields_returns_correct_value(self) -> None:
        fields = {f"f{i}": 0.6 for i in range(100)}
        result = inverse_confidence_weight(fields)
        assert result == pytest.approx(0.6, abs=1e-9)

    def test_gradient_across_many_fields(self) -> None:
        """Fields spanning [0, 1] — result should be pulled below the mean."""
        n = 20
        fields = {f"f{i}": i / (n - 1) for i in range(n)}
        plain_mean = sum(fields.values()) / len(fields)  # ≈ 0.5
        result = inverse_confidence_weight(fields)
        # Low-confidence fields are weighted more, so result < plain_mean
        assert result < plain_mean
