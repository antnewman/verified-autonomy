"""
Tests for Layer 02: Outlier Detection as Hard Escalation

Test naming convention: test_[function]_[scenario]_[expected_result]
Example: test_detect_outliers_single_outlier_escalates_to_expert_required

Requirements (see CONTRIBUTING.md):
- 90% minimum coverage
- Edge cases at both boundaries
- Corner cases at intersections of multiple boundaries
- Failure mode tests (correct exceptions, messages, fallbacks)
- No mocking of the system under test
"""

from __future__ import annotations

import pytest

from outlier_escalation import (
    DetectionResult,
    EscalationLevel,
    OutlierFlag,
    _compute_iqr,
    detect_outliers,
)


class TestComputeIQR:
    """Unit tests for the internal IQR computation helper."""

    def test_simple_five_element_list(self) -> None:
        # [1,2,3,4,5] → lower=[1,2], upper=[4,5]; Q1=1.5, Q3=4.5, IQR=3.0
        q1, q3, iqr = _compute_iqr([1.0, 2.0, 3.0, 4.0, 5.0])
        assert q1 == pytest.approx(1.5)
        assert q3 == pytest.approx(4.5)
        assert iqr == pytest.approx(3.0)

    def test_four_element_list(self) -> None:
        # [1,2,3,4] → lower=[1,2], upper=[3,4]; Q1=1.5, Q3=3.5, IQR=2.0
        q1, q3, iqr = _compute_iqr([1.0, 2.0, 3.0, 4.0])
        assert q1 == pytest.approx(1.5)
        assert q3 == pytest.approx(3.5)
        assert iqr == pytest.approx(2.0)

    def test_two_element_list(self) -> None:
        q1, q3, iqr = _compute_iqr([0.0, 10.0])
        assert q1 == pytest.approx(0.0)
        assert q3 == pytest.approx(10.0)
        assert iqr == pytest.approx(10.0)

    def test_identical_values_returns_zero_iqr(self) -> None:
        q1, q3, iqr = _compute_iqr([5.0, 5.0, 5.0, 5.0])
        assert iqr == pytest.approx(0.0)

    def test_iqr_equals_q3_minus_q1(self) -> None:
        q1, q3, iqr = _compute_iqr([2.0, 4.0, 6.0, 8.0, 10.0])
        assert iqr == pytest.approx(q3 - q1)


class TestDetectOutliersCleanData:
    """No outliers present — escalation driven purely by confidence."""

    def test_high_confidence_clean_data_returns_none(self) -> None:
        result = detect_outliers([10.0, 10.5, 10.2, 10.3], confidence=0.9)
        assert result.escalation_level == EscalationLevel.NONE
        assert not result.has_outliers

    def test_spot_check_tier_for_moderate_confidence(self) -> None:
        result = detect_outliers([1.0, 2.0, 1.5], confidence=0.75)
        assert result.escalation_level == EscalationLevel.SPOT_CHECK
        assert not result.has_outliers

    def test_detailed_review_tier_for_low_confidence(self) -> None:
        result = detect_outliers([1.0, 2.0, 1.5], confidence=0.55)
        assert result.escalation_level == EscalationLevel.DETAILED_REVIEW
        assert not result.has_outliers

    def test_expert_required_for_very_low_confidence(self) -> None:
        result = detect_outliers([1.0, 2.0, 1.5], confidence=0.35)
        assert result.escalation_level == EscalationLevel.EXPERT_REQUIRED

    def test_confidence_exactly_at_lower_tier_boundary(self) -> None:
        # Exactly 0.4 → DETAILED_REVIEW (not EXPERT_REQUIRED)
        r1 = detect_outliers([1.0, 1.5, 2.0], confidence=0.4)
        assert r1.escalation_level == EscalationLevel.DETAILED_REVIEW

    def test_confidence_exactly_at_middle_tier_boundary(self) -> None:
        # Exactly 0.6 → SPOT_CHECK (not DETAILED_REVIEW)
        r2 = detect_outliers([1.0, 1.5, 2.0], confidence=0.6)
        assert r2.escalation_level == EscalationLevel.SPOT_CHECK

    def test_confidence_exactly_at_upper_tier_boundary(self) -> None:
        # Exactly 0.8 → NONE (not SPOT_CHECK)
        r3 = detect_outliers([1.0, 1.5, 2.0], confidence=0.8)
        assert r3.escalation_level == EscalationLevel.NONE


class TestDetectOutliersOutlierDetection:
    """Outlier presence always escalates to EXPERT_REQUIRED regardless of confidence.

    Note on test data: Tukey's IQR fence requires the outlier to NOT be included
    in Q3 computation to produce a tight fence.  This needs at least six values
    where the outlier(s) are a minority, so Q1/Q3 reflect the normal cluster.
    With fewer values, the outlier inflates Q3 and widens the fences past the
    outlier itself.  All datasets here use six or more values.
    """

    def test_single_outlier_triggers_expert_required(self) -> None:
        # 7 values: tight cluster 10.0–10.2 + one at 100.0
        # Q1≈10.0, Q3≈10.2, IQR≈0.2, upper fence≈10.5 → 100.0 flagged
        values = [10.0, 10.2, 10.1, 10.0, 10.1, 10.2, 100.0]
        result = detect_outliers(values, confidence=0.95)
        assert result.escalation_level == EscalationLevel.EXPERT_REQUIRED
        assert result.has_outliers

    def test_outlier_overrides_perfect_confidence(self) -> None:
        """Even confidence=1.0 cannot prevent EXPERT_REQUIRED when an outlier exists.

        With six identical values + one outlier: Q1=Q3=1.0, IQR=0, upper_fence=1.0.
        The outlier (100.0) is above the fence and must be flagged.
        """
        result = detect_outliers([1.0, 1.0, 1.0, 1.0, 1.0, 100.0], confidence=1.0)
        assert result.escalation_level == EscalationLevel.EXPERT_REQUIRED

    def test_low_outlier_also_flagged(self) -> None:
        # Six identical + one very low: IQR=0, lower_fence=1.0, -100.0 < 1.0 → flagged
        result = detect_outliers([1.0, 1.0, 1.0, 1.0, 1.0, -100.0], confidence=0.9)
        assert result.has_outliers
        flagged_vals = [o.value for o in result.outliers]
        assert -100.0 in flagged_vals

    def test_multiple_outliers_all_flagged(self) -> None:
        # 10 values: eight at 1.0, two extreme outliers
        # Q1=Q3=1.0, IQR=0, upper_fence=1.0 → both 1000 and 2000 flagged
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0, 2000.0]
        result = detect_outliers(values, confidence=0.9)
        assert result.has_outliers
        flagged_vals = {o.value for o in result.outliers}
        assert 1000.0 in flagged_vals
        assert 2000.0 in flagged_vals

    def test_outlier_divergence_is_between_zero_and_one(self) -> None:
        values = [10.0, 10.1, 10.0, 10.1, 10.0, 100.0]
        result = detect_outliers(values, confidence=0.9)
        for outlier in result.outliers:
            assert 0.0 <= outlier.divergence <= 1.0


class TestDetectOutliersLabels:
    """Label handling — custom labels vs. auto-generated positional labels."""

    def test_custom_labels_appear_in_outlier_flags(self) -> None:
        # 6 values, tight cluster + one outlier with known label
        values = [10.0, 10.1, 10.0, 10.1, 10.0, 100.0]
        labels = ["alpha", "beta", "gamma", "delta", "epsilon", "outlier"]
        result = detect_outliers(values, labels=labels, confidence=0.9)
        assert result.has_outliers
        flagged_labels = [o.label for o in result.outliers]
        assert "outlier" in flagged_labels

    def test_no_labels_generates_positional_labels(self) -> None:
        # 6 values with zero IQR → positional labels used
        result = detect_outliers([1.0, 1.0, 1.0, 1.0, 1.0, 100.0], confidence=0.9)
        assert result.has_outliers
        for o in result.outliers:
            assert o.label.startswith("value[")

    def test_labels_appear_in_reason_string(self) -> None:
        # 6 values with zero IQR + named outlier
        values = [10.0, 10.0, 10.0, 10.0, 10.0, 500.0]
        result = detect_outliers(values, labels=["a", "b", "c", "d", "e", "huge"], confidence=0.9)
        assert result.has_outliers
        reasons = [o.reason for o in result.outliers]
        assert any("huge" in r for r in reasons)


class TestDetectOutliersConsensus:
    """Consensus is the median of input values."""

    def test_odd_count_consensus_is_middle_value(self) -> None:
        result = detect_outliers([1.0, 2.0, 3.0, 4.0, 5.0], confidence=0.9)
        assert result.consensus == pytest.approx(3.0)

    def test_even_count_consensus_is_mean_of_two_middles(self) -> None:
        result = detect_outliers([1.0, 2.0, 3.0, 4.0], confidence=0.9)
        assert result.consensus == pytest.approx(2.5)

    def test_consensus_preserved_in_result(self) -> None:
        values = [5.0, 6.0, 7.0, 8.0, 9.0]
        result = detect_outliers(values, confidence=0.9)
        assert result.consensus == pytest.approx(7.0)


class TestDetectOutliersReturnType:
    """Return type and attribute correctness."""

    def test_returns_detection_result(self) -> None:
        result = detect_outliers([1.0, 2.0, 3.0], confidence=0.9)
        assert isinstance(result, DetectionResult)

    def test_outliers_is_list(self) -> None:
        result = detect_outliers([1.0, 2.0, 3.0], confidence=0.9)
        assert isinstance(result.outliers, list)

    def test_outlier_flag_fields_present(self) -> None:
        # 6 identical values + 1 outlier → IQR=0, 100.0 flagged
        result = detect_outliers([1.0, 1.0, 1.0, 1.0, 1.0, 100.0], confidence=0.9)
        assert result.has_outliers
        flag = result.outliers[0]
        assert isinstance(flag, OutlierFlag)
        assert flag.label
        assert isinstance(flag.value, float)
        assert isinstance(flag.consensus, float)
        assert isinstance(flag.divergence, float)
        assert flag.reason

    def test_confidence_stored_in_result(self) -> None:
        result = detect_outliers([1.0, 2.0, 3.0], confidence=0.72)
        assert result.confidence == pytest.approx(0.72)


class TestDetectOutliersEdgeCases:
    """Edge cases: minimal inputs and identical values."""

    def test_two_values_no_outlier_high_confidence(self) -> None:
        result = detect_outliers([5.0, 5.0], confidence=0.9)
        assert result.escalation_level == EscalationLevel.NONE
        assert not result.has_outliers

    def test_two_identical_values_no_outlier(self) -> None:
        result = detect_outliers([7.0, 7.0], confidence=0.85)
        assert not result.has_outliers

    def test_all_identical_values_no_outlier(self) -> None:
        result = detect_outliers([3.0, 3.0, 3.0, 3.0], confidence=0.9)
        assert not result.has_outliers

    def test_negative_values_handled_correctly(self) -> None:
        result = detect_outliers([-5.0, -4.8, -5.2, -5.0], confidence=0.9)
        assert result.escalation_level == EscalationLevel.NONE

    def test_zero_in_values_does_not_raise(self) -> None:
        result = detect_outliers([0.0, 1.0, 1.5, 2.0], confidence=0.9)
        assert isinstance(result, DetectionResult)

    def test_single_value_no_outlier_detection(self) -> None:
        """With only one value, IQR detection cannot run — no outliers expected."""
        result = detect_outliers([42.0], confidence=0.9)
        assert not result.has_outliers


class TestDetectOutliersValidation:
    """Failure mode: invalid inputs raise ValueError."""

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            detect_outliers([1.0, 2.0], confidence=1.01)

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            detect_outliers([1.0, 2.0], confidence=-0.01)

    def test_mismatched_labels_length_raises(self) -> None:
        with pytest.raises(ValueError, match="labels length"):
            detect_outliers([1.0, 2.0, 3.0], labels=["a", "b"], confidence=0.9)

    def test_empty_labels_list_with_values_raises(self) -> None:
        with pytest.raises(ValueError, match="labels length"):
            detect_outliers([1.0, 2.0], labels=[], confidence=0.9)

    def test_confidence_exactly_zero_is_valid(self) -> None:
        result = detect_outliers([1.0, 2.0], confidence=0.0)
        assert result.escalation_level == EscalationLevel.EXPERT_REQUIRED

    def test_confidence_exactly_one_is_valid(self) -> None:
        result = detect_outliers([1.0, 2.0], confidence=1.0)
        assert isinstance(result, DetectionResult)


class TestMain:
    """The main() demonstration function runs without error."""

    def test_main_runs_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        from outlier_escalation import main
        main()
        captured = capsys.readouterr()
        assert "Layer 02" in captured.out
        assert "Outlier Detection" in captured.out

    def test_main_output_contains_all_scenarios(self, capsys: pytest.CaptureFixture[str]) -> None:
        from outlier_escalation import main
        main()
        captured = capsys.readouterr()
        assert "Scenario 1" in captured.out
        assert "Scenario 2" in captured.out
        assert "Scenario 3" in captured.out
        assert "Scenario 4" in captured.out


class TestEscalationLevelEnum:
    """EscalationLevel enum correctness."""

    def test_enum_values(self) -> None:
        assert EscalationLevel.NONE.value == "none"
        assert EscalationLevel.SPOT_CHECK.value == "spot_check"
        assert EscalationLevel.DETAILED_REVIEW.value == "detailed_review"
        assert EscalationLevel.EXPERT_REQUIRED.value == "expert_required"

    def test_enum_members_are_strings(self) -> None:
        for member in EscalationLevel:
            assert isinstance(member.value, str)

    def test_has_outliers_property(self) -> None:
        result = detect_outliers([1.0, 2.0, 3.0], confidence=0.9)
        assert result.has_outliers is False
        # 6 identical + 1 outlier with zero IQR → outlier is flagged
        result_with = detect_outliers([1.0, 1.0, 1.0, 1.0, 1.0, 100.0], confidence=0.9)
        assert result_with.has_outliers is True
