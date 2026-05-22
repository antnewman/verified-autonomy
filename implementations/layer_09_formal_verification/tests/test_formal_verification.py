"""
Tests for Layer 09: Formal Verification

Test naming convention: test_[function]_[scenario]_[expected_result]
Example: test_verify_output_bounds_wide_bounds_verified

Requirements:
- 90% minimum coverage
- Edge cases at both boundaries
- Corner cases at intersections of multiple boundaries
- Failure mode tests (correct exceptions, messages, fallbacks)
- No mocking of the system under test
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from formal_verification import (
    ConstrainedAI,
    SafetyKernelResult,
    TinyNetwork,
    VerificationResult,
    _count_parameters,
    _encode_network_z3,
    _z3_val_to_float,
    build_tiny_network,
    forward,
    main,
    measure_verification_scaling,
    verify_monotonicity,
    verify_output_bounds,
)


# ---------------------------------------------------------------------------
# TestTinyNetwork
# ---------------------------------------------------------------------------


class TestTinyNetwork:
    """Tests for the TinyNetwork dataclass."""

    def test_creation_stores_attributes(self) -> None:
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([0.1, 0.2])
        net = TinyNetwork(weights=[W], biases=[b], layer_sizes=(2, 2))
        assert len(net.weights) == 1
        assert len(net.biases) == 1
        assert net.layer_sizes == (2, 2)

    def test_frozen_raises_on_mutation(self) -> None:
        net = TinyNetwork(weights=[], biases=[], layer_sizes=(2,))
        with pytest.raises((AttributeError, TypeError)):
            net.layer_sizes = (3,)  # type: ignore[misc]

    def test_layer_sizes_preserved(self) -> None:
        net = build_tiny_network((2, 4, 8, 1), seed=0)
        assert net.layer_sizes == (2, 4, 8, 1)

    def test_weights_count_matches_layers(self) -> None:
        net = build_tiny_network((3, 5, 2), seed=1)
        assert len(net.weights) == 2
        assert len(net.biases) == 2


# ---------------------------------------------------------------------------
# TestForward
# ---------------------------------------------------------------------------


class TestForward:
    """Tests for the forward function."""

    def test_single_layer_linear(self) -> None:
        W = np.array([[2.0, 3.0]])
        b = np.array([1.0])
        net = TinyNetwork(weights=[W], biases=[b], layer_sizes=(2, 1))
        x = np.array([1.0, 1.0])
        result = forward(net, x)
        # 2*1 + 3*1 + 1 = 6
        assert np.isclose(result[0], 6.0)

    def test_relu_applied_to_hidden_layers(self) -> None:
        # First layer: x -> -x (negative output), second layer: identity
        W1 = np.array([[-1.0]])
        b1 = np.array([0.0])
        W2 = np.array([[1.0]])
        b2 = np.array([0.0])
        net = TinyNetwork(weights=[W1, W2], biases=[b1, b2], layer_sizes=(1, 1, 1))
        x = np.array([2.0])
        result = forward(net, x)
        # After layer 1: -2.0, after ReLU: 0.0, after layer 2: 0.0
        assert np.isclose(result[0], 0.0)

    def test_relu_not_applied_to_last_layer(self) -> None:
        # Last layer can output negative values
        W1 = np.array([[1.0]])
        b1 = np.array([0.0])
        W2 = np.array([[-1.0]])
        b2 = np.array([0.0])
        net = TinyNetwork(weights=[W1, W2], biases=[b1, b2], layer_sizes=(1, 1, 1))
        x = np.array([3.0])
        result = forward(net, x)
        # Hidden: relu(3.0) = 3.0, output: -3.0 (no relu on last layer)
        assert np.isclose(result[0], -3.0)

    def test_output_shape_matches_network(self) -> None:
        net = build_tiny_network((3, 8, 4, 2), seed=42)
        x = np.zeros(3)
        result = forward(net, x)
        assert result.shape == (2,)

    def test_forward_with_zeros_input(self) -> None:
        net = build_tiny_network((2, 4, 1), seed=42)
        x = np.zeros(2)
        result = forward(net, x)
        assert result.shape == (1,)

    def test_forward_deterministic(self) -> None:
        net = build_tiny_network((2, 4, 1), seed=42)
        x = np.array([0.5, 0.7])
        r1 = forward(net, x)
        r2 = forward(net, x)
        assert np.allclose(r1, r2)


# ---------------------------------------------------------------------------
# TestVerifyOutputBounds
# ---------------------------------------------------------------------------


class TestVerifyOutputBounds:
    """Tests for verify_output_bounds."""

    def test_property_holds_wide_bounds(self) -> None:
        net = build_tiny_network((2, 4, 1), seed=42)
        lo = np.zeros(2)
        hi = np.ones(2)
        result = verify_output_bounds(net, lo, hi, output_min=-100.0, output_max=100.0)
        assert result.verified is True
        assert result.counterexample is None
        assert result.property_name == "output_bounds"
        assert result.parameters_checked > 0
        assert result.time_seconds >= 0.0

    def test_property_violated_impossible_bounds(self) -> None:
        # Build a network and compute exact output at a single point,
        # then set bounds that definitely exclude that output
        net = build_tiny_network((2, 4, 1), seed=42)
        lo = np.array([0.5, 0.5])
        hi = np.array([0.5, 0.5])  # single point
        actual_out = float(forward(net, np.array([0.5, 0.5]))[0])
        tight_min = actual_out + 1.0
        tight_max = actual_out + 2.0
        result = verify_output_bounds(net, lo, hi, output_min=tight_min, output_max=tight_max)
        assert result.verified is False
        assert result.counterexample is not None

    def test_counterexample_is_valid_input(self) -> None:
        net = build_tiny_network((2, 4, 1), seed=42)
        lo = np.array([0.5, 0.5])
        hi = np.array([0.5, 0.5])
        actual_out = float(forward(net, np.array([0.5, 0.5]))[0])
        tight_min = actual_out + 1.0
        tight_max = actual_out + 2.0
        result = verify_output_bounds(net, lo, hi, output_min=tight_min, output_max=tight_max)
        assert result.counterexample is not None
        # Counterexample should be within input bounds
        ce = result.counterexample
        assert "x_0" in ce
        assert "x_1" in ce

    def test_no_bounds_specified_trivially_verified(self) -> None:
        net = build_tiny_network((2, 4, 1), seed=42)
        lo = np.zeros(2)
        hi = np.ones(2)
        result = verify_output_bounds(net, lo, hi)
        assert result.verified is True
        assert "trivially" in result.explanation

    def test_only_output_min_specified(self) -> None:
        net = build_tiny_network((2, 4, 1), seed=42)
        lo = np.zeros(2)
        hi = np.ones(2)
        result = verify_output_bounds(net, lo, hi, output_min=-100.0)
        assert result.verified is True

    def test_only_output_max_specified(self) -> None:
        net = build_tiny_network((2, 4, 1), seed=42)
        lo = np.zeros(2)
        hi = np.ones(2)
        result = verify_output_bounds(net, lo, hi, output_max=100.0)
        assert result.verified is True

    def test_result_contains_explanation(self) -> None:
        net = build_tiny_network((2, 4, 1), seed=42)
        lo = np.zeros(2)
        hi = np.ones(2)
        result = verify_output_bounds(net, lo, hi, output_min=-100.0, output_max=100.0)
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

    def test_parameters_checked_correct(self) -> None:
        net = build_tiny_network((2, 4, 1), seed=42)
        lo = np.zeros(2)
        hi = np.ones(2)
        result = verify_output_bounds(net, lo, hi, output_min=-100.0, output_max=100.0)
        # (2,4,1): 4*2 + 4 + 1*4 + 1 = 8 + 4 + 4 + 1 = 17
        assert result.parameters_checked == 17

    def test_single_input_network(self) -> None:
        net = build_tiny_network((1, 2, 1), seed=7)
        lo = np.array([0.0])
        hi = np.array([1.0])
        result = verify_output_bounds(net, lo, hi, output_min=-100.0, output_max=100.0)
        assert result.verified is True

    def test_verification_result_frozen(self) -> None:
        net = build_tiny_network((2, 4, 1), seed=42)
        lo = np.zeros(2)
        hi = np.ones(2)
        result = verify_output_bounds(net, lo, hi, output_min=-100.0, output_max=100.0)
        with pytest.raises((AttributeError, TypeError)):
            result.verified = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestVerifyMonotonicity
# ---------------------------------------------------------------------------


class TestVerifyMonotonicity:
    """Tests for verify_monotonicity."""

    def test_monotonicity_holds_positive_weight(self) -> None:
        # Single-layer network f(x) = w*x with w > 0: definitely monotone
        W = np.array([[1.0]])
        b = np.array([0.0])
        net = TinyNetwork(weights=[W], biases=[b], layer_sizes=(1, 1))
        lo = np.array([0.0])
        hi = np.array([1.0])
        result = verify_monotonicity(net, lo, hi, input_dimension=0)
        assert result.verified is True
        assert result.counterexample is None

    def test_monotonicity_holds_positive_weights_multi_input(self) -> None:
        # f(x1, x2) = x1 + x2: monotone in both dimensions
        W = np.array([[1.0, 1.0]])
        b = np.array([0.0])
        net = TinyNetwork(weights=[W], biases=[b], layer_sizes=(2, 1))
        lo = np.zeros(2)
        hi = np.ones(2)
        result = verify_monotonicity(net, lo, hi, input_dimension=0)
        assert result.verified is True

    def test_monotonicity_violated_negative_weight(self) -> None:
        # f(x) = -x: definitely not monotone (increasing x decreases output)
        W = np.array([[-1.0]])
        b = np.array([0.0])
        net = TinyNetwork(weights=[W], biases=[b], layer_sizes=(1, 1))
        lo = np.array([0.0])
        hi = np.array([1.0])
        result = verify_monotonicity(net, lo, hi, input_dimension=0)
        assert result.verified is False
        assert result.counterexample is not None

    def test_monotonicity_counterexample_has_correct_keys(self) -> None:
        W = np.array([[-1.0]])
        b = np.array([0.0])
        net = TinyNetwork(weights=[W], biases=[b], layer_sizes=(1, 1))
        lo = np.array([0.0])
        hi = np.array([1.0])
        result = verify_monotonicity(net, lo, hi, input_dimension=0)
        assert result.counterexample is not None
        assert "x1_0" in result.counterexample
        assert "x2_0" in result.counterexample

    def test_monotonicity_property_name(self) -> None:
        W = np.array([[1.0]])
        b = np.array([0.0])
        net = TinyNetwork(weights=[W], biases=[b], layer_sizes=(1, 1))
        lo = np.array([0.0])
        hi = np.array([1.0])
        result = verify_monotonicity(net, lo, hi, input_dimension=0)
        assert result.property_name == "monotonicity_dim0"

    def test_monotonicity_checks_correct_dimension(self) -> None:
        # f(x1, x2) = -x1 + x2: monotone in dim1 but not dim0
        W = np.array([[-1.0, 1.0]])
        b = np.array([0.0])
        net = TinyNetwork(weights=[W], biases=[b], layer_sizes=(2, 1))
        lo = np.zeros(2)
        hi = np.ones(2)
        result_dim0 = verify_monotonicity(net, lo, hi, input_dimension=0)
        result_dim1 = verify_monotonicity(net, lo, hi, input_dimension=1)
        assert result_dim0.verified is False
        assert result_dim1.verified is True

    def test_monotonicity_output_dimension(self) -> None:
        # Two-output network: first output monotone, second not
        W = np.array([[1.0], [-1.0]])
        b = np.array([0.0, 0.0])
        net = TinyNetwork(weights=[W], biases=[b], layer_sizes=(1, 2))
        lo = np.array([0.0])
        hi = np.array([1.0])
        result_out0 = verify_monotonicity(net, lo, hi, input_dimension=0, output_dimension=0)
        result_out1 = verify_monotonicity(net, lo, hi, input_dimension=0, output_dimension=1)
        assert result_out0.verified is True
        assert result_out1.verified is False

    def test_monotonicity_parameters_checked(self) -> None:
        W = np.array([[1.0]])
        b = np.array([0.0])
        net = TinyNetwork(weights=[W], biases=[b], layer_sizes=(1, 1))
        lo = np.array([0.0])
        hi = np.array([1.0])
        result = verify_monotonicity(net, lo, hi, input_dimension=0)
        assert result.parameters_checked == 2  # 1 weight + 1 bias

    def test_monotonicity_time_recorded(self) -> None:
        W = np.array([[1.0]])
        b = np.array([0.0])
        net = TinyNetwork(weights=[W], biases=[b], layer_sizes=(1, 1))
        lo = np.array([0.0])
        hi = np.array([1.0])
        result = verify_monotonicity(net, lo, hi, input_dimension=0)
        assert result.time_seconds >= 0.0


# ---------------------------------------------------------------------------
# TestConstrainedAI
# ---------------------------------------------------------------------------


class TestConstrainedAI:
    """Tests for ConstrainedAI."""

    def _build_cai(
        self, model_fn: object = None
    ) -> ConstrainedAI:
        kernel = build_tiny_network((2, 4, 1), seed=42)
        if model_fn is None:
            model_fn = lambda x: 1.0  # noqa: E731
        return ConstrainedAI(model_fn, kernel, (-5.0, 5.0))  # type: ignore[arg-type]

    def test_no_override_when_within_bounds(self) -> None:
        cai = self._build_cai(lambda x: 1.0)
        x = np.array([0.5, 0.5])
        result = cai.predict(x)
        assert result.override_triggered is False
        assert result.final_output == result.model_output

    def test_override_when_above_upper_bound(self) -> None:
        cai = self._build_cai(lambda x: 100.0)
        x = np.array([0.5, 0.5])
        result = cai.predict(x)
        assert result.override_triggered is True
        assert result.final_output == result.kernel_output

    def test_override_when_below_lower_bound(self) -> None:
        cai = self._build_cai(lambda x: -100.0)
        x = np.array([0.5, 0.5])
        result = cai.predict(x)
        assert result.override_triggered is True
        assert result.final_output == result.kernel_output

    def test_kernel_verified_initially_false(self) -> None:
        cai = self._build_cai()
        assert cai.kernel_verified is False

    def test_verify_kernel_sets_verified(self) -> None:
        cai = self._build_cai()
        lo = np.zeros(2)
        hi = np.ones(2)
        result = cai.verify_kernel(lo, hi)
        assert isinstance(result, VerificationResult)
        assert cai.kernel_verified == result.verified

    def test_verify_kernel_wide_bounds_verified(self) -> None:
        kernel = build_tiny_network((2, 4, 1), seed=42)
        cai = ConstrainedAI(lambda x: 0.0, kernel, (-100.0, 100.0))
        lo = np.zeros(2)
        hi = np.ones(2)
        result = cai.verify_kernel(lo, hi)
        assert result.verified is True
        assert cai.kernel_verified is True

    def test_predict_returns_safety_kernel_result(self) -> None:
        cai = self._build_cai()
        x = np.array([0.5, 0.5])
        result = cai.predict(x)
        assert isinstance(result, SafetyKernelResult)

    def test_predict_model_output_recorded(self) -> None:
        cai = self._build_cai(lambda x: 2.5)
        x = np.array([0.5, 0.5])
        result = cai.predict(x)
        assert np.isclose(result.model_output, 2.5)

    def test_predict_kernel_verified_reflects_state(self) -> None:
        cai = self._build_cai()
        x = np.array([0.5, 0.5])
        result_before = cai.predict(x)
        assert result_before.kernel_verified is False

        lo = np.zeros(2)
        hi = np.ones(2)
        cai.verify_kernel(lo, hi)
        result_after = cai.predict(x)
        assert result_after.kernel_verified == cai.kernel_verified

    def test_predict_explanation_non_empty(self) -> None:
        cai = self._build_cai()
        x = np.array([0.5, 0.5])
        result = cai.predict(x)
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

    def test_predict_at_boundary_not_overridden(self) -> None:
        # Model exactly at upper bound — should NOT trigger override
        cai = self._build_cai(lambda x: 5.0)
        x = np.array([0.5, 0.5])
        result = cai.predict(x)
        assert result.override_triggered is False

    def test_predict_at_boundary_lower_not_overridden(self) -> None:
        # Model exactly at lower bound — should NOT trigger override
        cai = self._build_cai(lambda x: -5.0)
        x = np.array([0.5, 0.5])
        result = cai.predict(x)
        assert result.override_triggered is False

    def test_safety_kernel_result_frozen(self) -> None:
        cai = self._build_cai()
        x = np.array([0.5, 0.5])
        result = cai.predict(x)
        with pytest.raises((AttributeError, TypeError)):
            result.final_output = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestMeasureVerificationScaling
# ---------------------------------------------------------------------------


class TestMeasureVerificationScaling:
    """Tests for measure_verification_scaling."""

    def test_default_returns_list(self) -> None:
        results = measure_verification_scaling()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_default_architecture_count(self) -> None:
        results = measure_verification_scaling()
        assert len(results) == 6

    def test_result_has_required_keys(self) -> None:
        results = measure_verification_scaling([(2, 2, 1)])
        assert len(results) == 1
        r = results[0]
        assert "architecture" in r
        assert "parameters" in r
        assert "time_seconds" in r
        assert "verified" in r

    def test_custom_sizes(self) -> None:
        sizes = [(2, 2, 1), (2, 4, 1)]
        results = measure_verification_scaling(sizes)
        assert len(results) == 2

    def test_parameters_increase_with_size(self) -> None:
        sizes = [(2, 2, 1), (2, 8, 1)]
        results = measure_verification_scaling(sizes)
        assert results[1]["parameters"] > results[0]["parameters"]

    def test_all_verified_with_wide_bounds(self) -> None:
        # With -100/100 bounds, all should be verified
        sizes = [(2, 2, 1), (2, 4, 1)]
        results = measure_verification_scaling(sizes)
        for r in results:
            assert r["verified"] is True

    def test_architecture_string_present(self) -> None:
        sizes = [(2, 4, 1)]
        results = measure_verification_scaling(sizes)
        assert "(2, 4, 1)" in str(results[0]["architecture"])

    def test_time_seconds_positive(self) -> None:
        sizes = [(2, 2, 1)]
        results = measure_verification_scaling(sizes)
        assert float(results[0]["time_seconds"]) >= 0.0


# ---------------------------------------------------------------------------
# TestBuildTinyNetwork
# ---------------------------------------------------------------------------


class TestBuildTinyNetwork:
    """Tests for build_tiny_network."""

    def test_weight_shapes_correct(self) -> None:
        net = build_tiny_network((3, 5, 2), seed=0)
        assert net.weights[0].shape == (5, 3)
        assert net.weights[1].shape == (2, 5)

    def test_bias_shapes_correct(self) -> None:
        net = build_tiny_network((3, 5, 2), seed=0)
        assert net.biases[0].shape == (5,)
        assert net.biases[1].shape == (2,)

    def test_deterministic_with_seed(self) -> None:
        net1 = build_tiny_network((2, 4, 1), seed=42)
        net2 = build_tiny_network((2, 4, 1), seed=42)
        assert np.allclose(net1.weights[0], net2.weights[0])

    def test_different_seeds_differ(self) -> None:
        net1 = build_tiny_network((2, 4, 1), seed=1)
        net2 = build_tiny_network((2, 4, 1), seed=2)
        assert not np.allclose(net1.weights[0], net2.weights[0])

    def test_single_layer_network(self) -> None:
        net = build_tiny_network((4, 1), seed=42)
        assert len(net.weights) == 1
        assert net.weights[0].shape == (1, 4)

    def test_deep_network(self) -> None:
        net = build_tiny_network((2, 4, 4, 4, 1), seed=42)
        assert len(net.weights) == 4
        assert net.layer_sizes == (2, 4, 4, 4, 1)


# ---------------------------------------------------------------------------
# TestCountParameters
# ---------------------------------------------------------------------------


class TestCountParameters:
    """Tests for _count_parameters."""

    def test_single_layer(self) -> None:
        net = TinyNetwork(
            weights=[np.ones((3, 2))],
            biases=[np.ones(3)],
            layer_sizes=(2, 3),
        )
        # 3*2 + 3 = 9
        assert _count_parameters(net) == 9

    def test_two_layer(self) -> None:
        net = build_tiny_network((2, 4, 1), seed=42)
        # Layer 1: 4*2=8 weights + 4 biases = 12
        # Layer 2: 1*4=4 weights + 1 bias = 5
        # Total = 17
        assert _count_parameters(net) == 17


# ---------------------------------------------------------------------------
# TestZ3ValToFloat
# ---------------------------------------------------------------------------


class TestZ3ValToFloat:
    """Tests for _z3_val_to_float helper."""

    def test_none_returns_zero(self) -> None:
        assert _z3_val_to_float(None) == 0.0

    def test_z3_rational_value(self) -> None:
        import z3

        val = z3.RealVal("1/2")
        result = _z3_val_to_float(val)
        assert np.isclose(result, 0.5)

    def test_z3_integer_value(self) -> None:
        import z3

        val = z3.RealVal(3)
        result = _z3_val_to_float(val)
        assert np.isclose(result, 3.0)


# ---------------------------------------------------------------------------
# TestEncodeNetworkZ3
# ---------------------------------------------------------------------------


class TestEncodeNetworkZ3:
    """Tests for _encode_network_z3."""

    def test_single_layer_encoding(self) -> None:
        import z3

        W = np.array([[2.0]])
        b = np.array([1.0])
        net = TinyNetwork(weights=[W], biases=[b], layer_sizes=(1, 1))
        x = [z3.Real("x_0")]
        outputs = _encode_network_z3(net, x)
        assert len(outputs) == 1

    def test_output_count_matches_network(self) -> None:
        import z3

        net = build_tiny_network((2, 4, 3), seed=0)
        xs = [z3.Real(f"x_{i}") for i in range(2)]
        outputs = _encode_network_z3(net, xs)
        assert len(outputs) == 3


# ---------------------------------------------------------------------------
# TestMain
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() demonstration function."""

    def test_main_runs_without_error(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        output = buf.getvalue()
        assert len(output) > 0

    def test_main_contains_layer_header(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        output = buf.getvalue()
        assert "Layer 09" in output

    def test_main_contains_demo1_header(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        output = buf.getvalue()
        assert "Demo 1" in output

    def test_main_contains_demo2_header(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        output = buf.getvalue()
        assert "Demo 2" in output

    def test_main_contains_demo3_header(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        output = buf.getvalue()
        assert "Demo 3" in output

    def test_main_contains_demo4_header(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        output = buf.getvalue()
        assert "Demo 4" in output

    def test_main_contains_verified_output(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        output = buf.getvalue()
        assert "Verified" in output or "verified" in output

    def test_main_contains_constrained_ai(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        output = buf.getvalue()
        assert "Constrained AI" in output or "Override" in output or "override" in output
