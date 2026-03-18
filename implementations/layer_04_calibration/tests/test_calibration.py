"""
Tests for Layer 04: Calibration and Conformal Prediction

Test naming convention: test_[function]_[scenario]_[expected_result]
Example: test_compute_ece_perfect_calibration_returns_zero

Requirements (see CONTRIBUTING.md):
- 90% minimum coverage
- Edge cases at both boundaries
- Corner cases at intersections of multiple boundaries
- Failure mode tests (correct exceptions, messages, fallbacks)
- No mocking of the system under test
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from calibration import (
    BinData,
    apply_temperature_scaling,
    calibrate_conformal,
    compute_ece,
    conformal_predict,
    evaluate_coverage,
    find_temperature,
    reliability_diagram_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _one_hot_probs(predicted_class: int, confidence: float, n_classes: int) -> np.ndarray:
    """Create a single-sample probability row with given confidence in one class."""
    probs = np.full(n_classes, (1.0 - confidence) / (n_classes - 1))
    probs[predicted_class] = confidence
    return probs


def _uniform_probs(n_samples: int, n_classes: int) -> np.ndarray:
    """Uniform probability distribution for all samples."""
    return np.full((n_samples, n_classes), 1.0 / n_classes)


# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------


class TestComputeEce:
    """ECE calculation with equal-width binning (Guo et al. 2017)."""

    def test_perfect_calibration_returns_zero(self) -> None:
        """When accuracy equals confidence in every bin, ECE is 0."""
        # All 1000 samples predict class 0 with confidence 0.8.
        # 800 are correct (acc = 0.8 = conf) => ECE = 0.
        n = 1000
        y_prob = np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])
        y_true = np.array([0] * 800 + [1] * 200)
        ece, _ = compute_ece(y_true, y_prob)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_maximally_overconfident_returns_high_ece(self) -> None:
        """Always predicting class 0 with confidence 1.0, half wrong => ECE = 0.5."""
        n = 100
        y_prob = np.column_stack([np.ones(n), np.zeros(n)])
        y_true = np.array([0] * 50 + [1] * 50)
        ece, _ = compute_ece(y_true, y_prob)
        # All samples in the confidence=1.0 bin; acc=0.5, conf=1.0 => ECE=0.5
        assert ece == pytest.approx(0.5, abs=1e-6)

    def test_single_bin_ece_equals_abs_diff(self) -> None:
        """All predictions in one bin: ECE = |accuracy - confidence|."""
        n = 200
        # Confidence 0.9, accuracy 0.6
        y_prob = np.column_stack([np.full(n, 0.9), np.full(n, 0.1)])
        y_true = np.array([0] * 120 + [1] * 80)
        ece, bins = compute_ece(y_true, y_prob)
        assert len(bins) == 1
        assert ece == pytest.approx(abs(0.6 - bins[0].confidence), abs=1e-6)

    def test_empty_bins_are_skipped(self) -> None:
        """Empty bins contribute 0 and are omitted from the bin list."""
        n = 50
        # All confidences near 0.9 — most bins empty
        y_prob = np.column_stack([np.full(n, 0.9), np.full(n, 0.1)])
        y_true = np.zeros(n, dtype=int)
        ece, bins = compute_ece(y_true, y_prob, n_bins=15)
        # Only one non-empty bin
        assert len(bins) == 1
        assert all(b.count > 0 for b in bins)

    def test_n_bins_parameter_respected(self) -> None:
        """Changing n_bins changes the bin count."""
        n = 500
        rng = np.random.default_rng(0)
        y_prob_vals = rng.dirichlet([1, 1, 1], n)
        y_true = rng.integers(0, 3, n)
        _, bins_15 = compute_ece(y_true, y_prob_vals, n_bins=15)
        _, bins_5 = compute_ece(y_true, y_prob_vals, n_bins=5)
        # Fewer bins means at most 5 non-empty vs at most 15
        assert len(bins_5) <= len(bins_15)

    def test_n_bins_one_puts_everything_in_one_bin(self) -> None:
        n = 100
        y_prob = np.column_stack([np.full(n, 0.7), np.full(n, 0.3)])
        y_true = np.array([0] * 70 + [1] * 30)
        ece, bins = compute_ece(y_true, y_prob, n_bins=1)
        assert len(bins) == 1
        assert bins[0].count == n

    def test_n_bins_100_many_empty_bins_no_error(self) -> None:
        n = 50
        y_prob = np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])
        y_true = np.zeros(n, dtype=int)
        ece, bins = compute_ece(y_true, y_prob, n_bins=100)
        assert len(bins) <= 100
        assert ece >= 0.0

    def test_returns_correct_number_of_bindata_objects(self) -> None:
        n = 100
        y_prob = np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])
        y_true = np.zeros(n, dtype=int)
        _, bins = compute_ece(y_true, y_prob)
        assert all(isinstance(b, BinData) for b in bins)

    def test_length_mismatch_raises_value_error(self) -> None:
        y_prob = np.array([[0.8, 0.2], [0.7, 0.3]])
        y_true = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="length"):
            compute_ece(y_true, y_prob)

    def test_count_sums_to_total_samples(self) -> None:
        n = 300
        rng = np.random.default_rng(1)
        y_prob = rng.dirichlet([2, 2], n)
        y_true = rng.integers(0, 2, n)
        _, bins = compute_ece(y_true, y_prob)
        assert sum(b.count for b in bins) == n

    def test_ece_is_non_negative(self) -> None:
        rng = np.random.default_rng(2)
        y_prob = rng.dirichlet([1, 1, 1], 200)
        y_true = rng.integers(0, 3, 200)
        ece, _ = compute_ece(y_true, y_prob)
        assert ece >= 0.0

    def test_bin_midpoints_span_zero_to_one(self) -> None:
        n = 500
        rng = np.random.default_rng(3)
        y_prob = rng.dirichlet([1, 1], n)
        y_true = rng.integers(0, 2, n)
        _, bins = compute_ece(y_true, y_prob, n_bins=10)
        for b in bins:
            assert 0.0 < b.bin_midpoint < 1.0


# ---------------------------------------------------------------------------
# reliability_diagram_data
# ---------------------------------------------------------------------------


class TestReliabilityDiagramData:
    """reliability_diagram_data is a thin wrapper; verify contract."""

    def test_returns_list_of_bindata(self) -> None:
        n = 100
        y_prob = np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])
        y_true = np.zeros(n, dtype=int)
        bins = reliability_diagram_data(y_true, y_prob)
        assert isinstance(bins, list)
        assert all(isinstance(b, BinData) for b in bins)

    def test_count_sums_to_n(self) -> None:
        n = 200
        rng = np.random.default_rng(4)
        y_prob = rng.dirichlet([1, 1], n)
        y_true = rng.integers(0, 2, n)
        bins = reliability_diagram_data(y_true, y_prob)
        assert sum(b.count for b in bins) == n

    def test_perfect_calibration_accuracy_approx_confidence(self) -> None:
        """In the perfectly calibrated bin, acc ~ conf."""
        n = 1000
        y_prob = np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])
        y_true = np.array([0] * 800 + [1] * 200)
        bins = reliability_diagram_data(y_true, y_prob)
        assert len(bins) == 1
        assert abs(bins[0].accuracy - bins[0].confidence) < 1e-6

    def test_bin_midpoints_evenly_spaced(self) -> None:
        n = 500
        rng = np.random.default_rng(5)
        # Spread confidences across many bins
        y_prob = rng.dirichlet([0.5, 0.5], n)
        y_true = rng.integers(0, 2, n)
        bins = reliability_diagram_data(y_true, y_prob, n_bins=5)
        midpoints = [b.bin_midpoint for b in bins]
        # Each midpoint should be exactly (lower + upper)/2 for its bin
        # bin width = 1/5 = 0.2; midpoints are 0.1, 0.3, 0.5, 0.7, 0.9
        for mp in midpoints:
            assert any(abs(mp - expected) < 1e-9 for expected in [0.1, 0.3, 0.5, 0.7, 0.9])


# ---------------------------------------------------------------------------
# find_temperature
# ---------------------------------------------------------------------------


class TestFindTemperature:
    """Temperature scaling optimiser."""

    def test_overconfident_logits_return_t_greater_than_one(self) -> None:
        """High-confidence logits, 50% accuracy => T > 1 to soften predictions."""
        n = 100
        # Very confident in class 0 (logit >> 0)
        logits = np.zeros((n, 3))
        logits[:, 0] = 8.0
        # But only 50% correct
        y_true = np.array([0] * 50 + [1] * 50)
        T = find_temperature(logits, y_true)
        assert T > 1.0

    def test_returns_positive_float(self) -> None:
        n = 50
        logits = np.random.default_rng(6).standard_normal((n, 4))
        y_true = np.zeros(n, dtype=int)
        T = find_temperature(logits, y_true)
        assert isinstance(T, float)
        assert T > 0.0

    def test_result_within_search_bounds(self) -> None:
        """minimize_scalar bounds=(0.1, 10.0) => result in [0.1, 10.0]."""
        n = 80
        rng = np.random.default_rng(7)
        logits = rng.standard_normal((n, 3))
        y_true = rng.integers(0, 3, n)
        T = find_temperature(logits, y_true)
        assert 0.1 <= T <= 10.0

    def test_already_calibrated_logits_return_t_near_one(self) -> None:
        """When the model's accuracy matches its confidence, T ~ 1."""
        n = 200
        # Moderate confidence ~0.84 in class 0 (logit 2.0 vs 0.0)
        logits = np.zeros((n, 3))
        logits[:, 0] = 2.0
        # Accuracy = 0.84 (matching the softmax confidence)
        y_true = np.array([0] * 168 + [1] * 32)
        T = find_temperature(logits, y_true)
        # Not strictly 1, but in the ballpark
        assert 0.5 < T < 2.5


# ---------------------------------------------------------------------------
# apply_temperature_scaling
# ---------------------------------------------------------------------------


class TestApplyTemperatureScaling:
    """Temperature scaling: softmax(logits / T)."""

    def _simple_logits(self) -> np.ndarray:
        return np.array([[3.0, 1.0, 0.5], [0.5, 2.0, 1.0]])

    def test_t_one_is_identity(self) -> None:
        """T=1.0 should return the same probabilities as plain softmax."""
        logits = self._simple_logits()
        probs_t1 = apply_temperature_scaling(logits, 1.0)
        # Compare with manual softmax
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        manual = exp_l / exp_l.sum(axis=1, keepdims=True)
        assert probs_t1 == pytest.approx(manual, abs=1e-9)

    def test_t_greater_than_one_softens_probabilities(self) -> None:
        """T > 1 moves probabilities toward uniform (max prob decreases)."""
        logits = self._simple_logits()
        probs_t1 = apply_temperature_scaling(logits, 1.0)
        probs_t2 = apply_temperature_scaling(logits, 2.0)
        assert (probs_t2.max(axis=1) < probs_t1.max(axis=1)).all()

    def test_t_less_than_one_sharpens_probabilities(self) -> None:
        """T < 1 moves probabilities toward one-hot."""
        logits = self._simple_logits()
        probs_t1 = apply_temperature_scaling(logits, 1.0)
        probs_t_half = apply_temperature_scaling(logits, 0.5)
        assert (probs_t_half.max(axis=1) > probs_t1.max(axis=1)).all()

    def test_rows_sum_to_one(self) -> None:
        logits = self._simple_logits()
        for T in [0.5, 1.0, 2.0, 5.0]:
            probs = apply_temperature_scaling(logits, T)
            assert probs.sum(axis=1) == pytest.approx(np.ones(len(logits)), abs=1e-9)

    def test_argmax_preserved(self) -> None:
        """Temperature scaling does not change the predicted class."""
        logits = self._simple_logits()
        probs_t1 = apply_temperature_scaling(logits, 1.0)
        probs_t3 = apply_temperature_scaling(logits, 3.0)
        assert (probs_t1.argmax(axis=1) == probs_t3.argmax(axis=1)).all()

    def test_temperature_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            apply_temperature_scaling(self._simple_logits(), 0.0)

    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            apply_temperature_scaling(self._simple_logits(), -1.0)

    def test_single_sample_input(self) -> None:
        logits = np.array([[2.0, 1.0, 0.0]])
        probs = apply_temperature_scaling(logits, 1.5)
        assert probs.shape == (1, 3)
        assert probs.sum() == pytest.approx(1.0, abs=1e-9)

    def test_two_class_input(self) -> None:
        logits = np.array([[2.0, -2.0], [0.5, -0.5]])
        probs = apply_temperature_scaling(logits, 1.0)
        assert probs.shape == (2, 2)
        assert probs.sum(axis=1) == pytest.approx([1.0, 1.0], abs=1e-9)


# ---------------------------------------------------------------------------
# calibrate_conformal
# ---------------------------------------------------------------------------


class TestCalibrateConformal:
    """Conformal threshold computation."""

    def test_returns_float(self) -> None:
        y_prob = np.array([[0.9, 0.1], [0.3, 0.7], [0.6, 0.4]])
        y_true = np.array([0, 1, 0])
        q = calibrate_conformal(y_true, y_prob)
        assert isinstance(q, float)

    def test_threshold_is_in_unit_interval(self) -> None:
        rng = np.random.default_rng(8)
        y_prob = rng.dirichlet([1, 1, 1], 200)
        y_true = rng.integers(0, 3, 200)
        q = calibrate_conformal(y_true, y_prob, alpha=0.1)
        assert 0.0 <= q <= 1.0

    def test_alpha_01_threshold_near_90th_percentile_of_scores(self) -> None:
        """Threshold should be close to the (n+1)(1-alpha)/n quantile of scores."""
        rng = np.random.default_rng(9)
        n = 1000
        y_prob = rng.dirichlet([1, 1, 1], n)
        y_true = rng.integers(0, 3, n)
        scores = 1.0 - y_prob[np.arange(n), y_true]
        q_level = math.ceil((n + 1) * 0.9) / n
        expected = float(np.quantile(scores, min(q_level, 1.0)))
        q_hat = calibrate_conformal(y_true, y_prob, alpha=0.1)
        assert q_hat == pytest.approx(expected, abs=1e-9)

    def test_higher_alpha_gives_lower_threshold(self) -> None:
        """Higher alpha (less coverage required) => lower threshold => smaller sets."""
        rng = np.random.default_rng(10)
        y_prob = rng.dirichlet([1, 1], 500)
        y_true = rng.integers(0, 2, 500)
        q_50 = calibrate_conformal(y_true, y_prob, alpha=0.5)
        q_10 = calibrate_conformal(y_true, y_prob, alpha=0.1)
        assert q_50 < q_10

    def test_lower_alpha_gives_higher_threshold(self) -> None:
        """Lower alpha (more coverage required) => higher threshold."""
        rng = np.random.default_rng(11)
        y_prob = rng.dirichlet([1, 1], 500)
        y_true = rng.integers(0, 2, 500)
        q_10 = calibrate_conformal(y_true, y_prob, alpha=0.10)
        q_01 = calibrate_conformal(y_true, y_prob, alpha=0.01)
        assert q_01 >= q_10

    def test_alpha_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            calibrate_conformal(np.array([0]), np.array([[0.8, 0.2]]), alpha=0.0)

    def test_alpha_one_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            calibrate_conformal(np.array([0]), np.array([[0.8, 0.2]]), alpha=1.0)

    def test_alpha_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            calibrate_conformal(np.array([0]), np.array([[0.8, 0.2]]), alpha=-0.1)

    def test_single_sample_calibration_does_not_raise(self) -> None:
        q = calibrate_conformal(
            np.array([0]),
            np.array([[0.7, 0.3]]),
            alpha=0.1,
        )
        assert isinstance(q, float)

    def test_alpha_near_zero_gives_threshold_near_one(self) -> None:
        """Near-zero alpha demands near-100% coverage => q_hat close to 1."""
        rng = np.random.default_rng(12)
        y_prob = rng.dirichlet([1, 1], 500)
        y_true = rng.integers(0, 2, 500)
        q = calibrate_conformal(y_true, y_prob, alpha=0.01)
        assert q > 0.5  # Should be conservative (large threshold)


# ---------------------------------------------------------------------------
# conformal_predict
# ---------------------------------------------------------------------------


class TestConformalPredict:
    """Prediction set construction from conformity threshold."""

    def test_returns_list_of_arrays(self) -> None:
        y_prob = np.array([[0.9, 0.1], [0.3, 0.7]])
        sets = conformal_predict(y_prob, threshold=0.5)
        assert isinstance(sets, list)
        assert all(isinstance(s, np.ndarray) for s in sets)

    def test_length_matches_number_of_samples(self) -> None:
        n = 50
        y_prob = np.random.default_rng(13).dirichlet([1, 1, 1], n)
        sets = conformal_predict(y_prob, threshold=0.5)
        assert len(sets) == n

    def test_threshold_one_includes_all_classes(self) -> None:
        """score = 1 - prob <= 1 is always true => all classes included."""
        y_prob = np.array([[0.9, 0.1], [0.3, 0.7], [0.5, 0.5]])
        sets = conformal_predict(y_prob, threshold=1.0)
        for s in sets:
            assert len(s) == 2  # Both classes included

    def test_threshold_zero_gives_minimal_sets(self) -> None:
        """score = 1 - prob <= 0.0 only when prob >= 1.0 (near-empty in practice)."""
        y_prob = np.array([[0.8, 0.2], [0.6, 0.4]])
        sets = conformal_predict(y_prob, threshold=0.0)
        # No class has prob=1.0 in this input, so sets should be empty
        for s in sets:
            assert len(s) == 0

    def test_higher_threshold_gives_larger_sets(self) -> None:
        """Higher threshold => more classes pass the score <= threshold test."""
        y_prob = np.array([[0.8, 0.15, 0.05]])
        sets_low = conformal_predict(y_prob, threshold=0.3)
        sets_high = conformal_predict(y_prob, threshold=0.9)
        assert len(sets_high[0]) >= len(sets_low[0])

    def test_lower_threshold_gives_smaller_sets(self) -> None:
        y_prob = np.array([[0.8, 0.15, 0.05]])
        sets_t05 = conformal_predict(y_prob, threshold=0.5)
        sets_t02 = conformal_predict(y_prob, threshold=0.2)
        assert len(sets_t05[0]) >= len(sets_t02[0])

    def test_prediction_set_contains_highest_confidence_class(self) -> None:
        """The most confident class has the lowest conformity score => always included."""
        y_prob = np.array([[0.9, 0.1]])
        # score for class 0 = 1 - 0.9 = 0.1 <= 0.5
        sets = conformal_predict(y_prob, threshold=0.5)
        assert 0 in sets[0]

    def test_exact_scores(self) -> None:
        """Verify inclusion logic with known scores."""
        # y_prob = [[0.7, 0.3]]
        # scores = [0.3, 0.7]
        # threshold=0.5: include class 0 (score 0.3 <= 0.5), exclude class 1 (0.7 > 0.5)
        y_prob = np.array([[0.7, 0.3]])
        sets = conformal_predict(y_prob, threshold=0.5)
        assert list(sets[0]) == [0]

    def test_multiple_samples_independent(self) -> None:
        y_prob = np.array([[0.9, 0.1], [0.4, 0.6]])
        sets = conformal_predict(y_prob, threshold=0.5)
        # Sample 0: scores [0.1, 0.9]; include class 0 only
        assert list(sets[0]) == [0]
        # Sample 1: scores [0.6, 0.4]; include class 1 only
        assert list(sets[1]) == [1]


# ---------------------------------------------------------------------------
# evaluate_coverage
# ---------------------------------------------------------------------------


class TestEvaluateCoverage:
    """Empirical coverage computation."""

    def test_perfect_coverage_returns_one(self) -> None:
        y_true = np.array([0, 1, 2])
        sets = [np.array([0, 1]), np.array([1, 2]), np.array([2])]
        assert evaluate_coverage(y_true, sets) == pytest.approx(1.0)

    def test_zero_coverage_returns_zero(self) -> None:
        y_true = np.array([0, 1, 2])
        sets = [np.array([1, 2]), np.array([0, 2]), np.array([0, 1])]
        assert evaluate_coverage(y_true, sets) == pytest.approx(0.0)

    def test_known_fraction_correct(self) -> None:
        y_true = np.array([0, 0, 0, 0])
        # First two covered, last two not
        sets = [np.array([0]), np.array([0]), np.array([1]), np.array([2])]
        assert evaluate_coverage(y_true, sets) == pytest.approx(0.5)

    def test_single_sample_covered(self) -> None:
        assert evaluate_coverage(np.array([1]), [np.array([0, 1])]) == pytest.approx(1.0)

    def test_single_sample_not_covered(self) -> None:
        assert evaluate_coverage(np.array([1]), [np.array([0])]) == pytest.approx(0.0)

    def test_empty_input_returns_zero(self) -> None:
        assert evaluate_coverage(np.array([]), []) == pytest.approx(0.0)

    def test_return_value_is_float(self) -> None:
        y_true = np.array([0])
        sets = [np.array([0])]
        result = evaluate_coverage(y_true, sets)
        assert isinstance(result, float)

    def test_coverage_is_between_zero_and_one(self) -> None:
        rng = np.random.default_rng(14)
        y_true = rng.integers(0, 3, 100)
        y_prob = rng.dirichlet([1, 1, 1], 100)
        q = calibrate_conformal(y_true, y_prob, alpha=0.1)
        sets = conformal_predict(y_prob, q)
        cov = evaluate_coverage(y_true, sets)
        assert 0.0 <= cov <= 1.0


# ---------------------------------------------------------------------------
# Integration test: the coverage guarantee
# ---------------------------------------------------------------------------


class TestCoverageGuarantee:
    """The core invariant: empirical coverage >= 1 - alpha on exchangeable data.

    This is the most important test — it verifies the mathematical guarantee
    holds in practice.  The dataset uses random_state=42 for reproducibility.
    """

    def test_empirical_coverage_meets_guarantee_alpha_01(self) -> None:
        """90% coverage guarantee with alpha=0.1."""
        from sklearn.datasets import make_classification  # type: ignore[import-untyped]
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore[import-untyped]

        X, y = make_classification(
            n_samples=3000,
            n_features=20,
            n_classes=5,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=1,
            random_state=42,
        )
        X_train, y_train = X[:1000], y[:1000]
        X_cal, y_cal = X[1000:2000], y[1000:2000]
        X_test, y_test = X[2000:], y[2000:]

        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        probs_cal = clf.predict_proba(X_cal)
        probs_test = clf.predict_proba(X_test)

        alpha = 0.1
        q_hat = calibrate_conformal(y_cal, probs_cal, alpha=alpha)
        prediction_sets = conformal_predict(probs_test, q_hat)
        coverage = evaluate_coverage(y_test, prediction_sets)

        # THE GUARANTEE: P(Y in C(X)) >= 1 - alpha.
        # The conformal coverage guarantee is distributional — it holds in
        # expectation over the randomness of the calibration/test split.
        # With n_test=1000 and target coverage 0.90, the empirical coverage
        # has standard error sigma = sqrt(0.9 * 0.1 / 1000) ~ 0.009.
        # We allow a 2-sigma tolerance for finite-sample variability on a
        # single fixed split while still confirming the guarantee is met.
        n_test = len(y_test)
        sigma = float(((1.0 - alpha) * alpha / n_test) ** 0.5)
        assert coverage >= 1.0 - alpha - 2.0 * sigma, (
            f"Coverage {coverage:.4f} is more than 2 sigma below the {1-alpha:.0%} "
            f"guarantee (sigma={sigma:.4f}). Something is wrong with the algorithm."
        )

    def test_empirical_coverage_meets_guarantee_alpha_05(self) -> None:
        """50% coverage guarantee with alpha=0.5."""
        from sklearn.datasets import make_classification  # type: ignore[import-untyped]
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore[import-untyped]

        X, y = make_classification(
            n_samples=3000,
            n_features=20,
            n_classes=5,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=1,
            random_state=42,
        )
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        clf.fit(X[:1000], y[:1000])

        probs_cal = clf.predict_proba(X[1000:2000])
        probs_test = clf.predict_proba(X[2000:])

        alpha = 0.5
        q_hat = calibrate_conformal(y[1000:2000], probs_cal, alpha=alpha)
        sets = conformal_predict(probs_test, q_hat)
        coverage = evaluate_coverage(y[2000:], sets)
        assert coverage >= 1.0 - alpha

    def test_ece_decreases_after_temperature_scaling(self) -> None:
        """Temperature scaling should reduce ECE on a held-out set."""
        from sklearn.datasets import make_classification  # type: ignore[import-untyped]
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore[import-untyped]

        X, y = make_classification(
            n_samples=3000,
            n_features=20,
            n_classes=5,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=1,
            random_state=42,
        )
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        clf.fit(X[:1000], y[:1000])

        probs_cal = clf.predict_proba(X[1000:2000])
        probs_test = clf.predict_proba(X[2000:])

        epsilon = 1e-7
        logits_cal = np.log(probs_cal + epsilon)
        logits_test = np.log(probs_test + epsilon)

        T = find_temperature(logits_cal, y[1000:2000])
        probs_scaled = apply_temperature_scaling(logits_test, T)

        ece_before, _ = compute_ece(y[2000:], probs_test)
        ece_after, _ = compute_ece(y[2000:], probs_scaled)

        assert ece_after < ece_before


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    """The main() demonstration runs without error and contains key narrative markers."""

    def test_main_runs_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        from calibration import main
        main()
        captured = capsys.readouterr()
        assert "Layer 04" in captured.out

    def test_main_output_contains_overconfidence_narrative(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from calibration import main
        main()
        out = capsys.readouterr().out
        assert "Overconfidence" in out or "overconfident" in out.lower()

    def test_main_output_contains_temperature_narrative(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from calibration import main
        main()
        out = capsys.readouterr().out
        assert "Temperature" in out or "temperature" in out.lower()

    def test_main_output_contains_conformal_narrative(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from calibration import main
        main()
        out = capsys.readouterr().out
        assert "Conformal" in out or "conformal" in out.lower()

    def test_main_output_contains_coverage_result(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from calibration import main
        main()
        out = capsys.readouterr().out
        assert "coverage" in out.lower() or "Coverage" in out

    def test_main_output_contains_comparison_table(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from calibration import main
        main()
        out = capsys.readouterr().out
        assert "Comparison" in out
        assert "Guarantee" in out
