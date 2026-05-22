"""
Layer 09: Formal Verification

The gold standard nobody can fully use yet.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

This module implements three demonstrations of formal verification for neural
networks:

1. SMT-based verification using Z3: encode a small network as real-valued
   constraints with ReLU as conditional expressions, then search for a
   counterexample to a stated property. If no counterexample exists, the
   property is PROVEN for ALL inputs in the range.

2. The state-space explosion: verify networks of increasing size and measure
   how quickly verification time grows.

3. The Constrained AI pattern: a small, formally verified safety kernel that
   monitors and can override a larger unverified model.

Architecture
------------
- ``TinyNetwork``:          A small ReLU network stored as numpy arrays.
- ``VerificationResult``:   Outcome of a Z3 verification query.
- ``SafetyKernelResult``:   Outcome of a ConstrainedAI.predict() call.
- ``ConstrainedAI``:        Verified kernel monitoring an unverified model.
- ``build_tiny_network``:   Construct a network with He-initialised weights.
- ``forward``:              Pure-numpy forward pass with ReLU activations.
- ``verify_output_bounds``: Prove output stays within bounds for all inputs.
- ``verify_monotonicity``:  Prove monotonicity in one input dimension.
- ``measure_verification_scaling``: Demonstrate the state-space explosion.

See layers/09-formal-verification.md for the full explanation, production
examples, and limitations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import z3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TinyNetwork:
    """A small fully-connected ReLU neural network.

    Attributes:
        weights:     Weight matrices, one per layer. weights[i] has shape
                     (out_features, in_features).
        biases:      Bias vectors, one per layer. biases[i] has shape
                     (out_features,).
        layer_sizes: Sizes of every layer including input and output,
                     e.g. (2, 4, 1) for a 2-input, 4-hidden, 1-output network.
    """

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    layer_sizes: tuple[int, ...]


@dataclass(frozen=True)
class VerificationResult:
    """Result of a formal verification query.

    Attributes:
        property_name:       What was being verified.
        verified:            True if the property holds for ALL inputs in the range.
        counterexample:      An input violating the property, if found.
        time_seconds:        How long verification took.
        parameters_checked:  Number of network parameters.
        explanation:         Human-readable summary.
    """

    property_name: str
    verified: bool
    counterexample: dict[str, float] | None
    time_seconds: float
    parameters_checked: int
    explanation: str


@dataclass(frozen=True)
class SafetyKernelResult:
    """Result from the Constrained AI safety kernel.

    Attributes:
        model_output:       What the unverified model proposed.
        kernel_output:      What the verified kernel computed.
        kernel_verified:    Whether the kernel's properties have been formally verified.
        override_triggered: Whether the kernel overrode the model.
        final_output:       The output that was actually used.
        explanation:        Human-readable summary.
    """

    model_output: float
    kernel_output: float
    kernel_verified: bool
    override_triggered: bool
    final_output: float
    explanation: str


# ---------------------------------------------------------------------------
# Network construction and forward pass
# ---------------------------------------------------------------------------


def build_tiny_network(layer_sizes: tuple[int, ...], seed: int = 42) -> TinyNetwork:
    """Construct a network with He-initialised weights.

    Args:
        layer_sizes: Sizes of every layer including input and output.
        seed:        Random seed for reproducibility.

    Returns:
        A TinyNetwork with He-initialised weights and normally distributed biases.
    """
    rng = np.random.default_rng(seed)
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        std = np.sqrt(2.0 / fan_in)
        W = rng.normal(0.0, std, size=(fan_out, fan_in))
        b = rng.normal(0.0, std, size=(fan_out,))
        weights.append(W)
        biases.append(b)
    return TinyNetwork(weights=weights, biases=biases, layer_sizes=layer_sizes)


def forward(network: TinyNetwork, x: np.ndarray) -> np.ndarray:
    """Pure-numpy forward pass with ReLU activations.

    ReLU is applied to all layers except the last (which is linear).

    Args:
        network: The network to evaluate.
        x:       Input vector of shape (input_dim,).

    Returns:
        Output vector of shape (output_dim,).
    """
    h = x.astype(float)
    for i, (W, b) in enumerate(zip(network.weights, network.biases)):
        h = W @ h + b
        if i < len(network.weights) - 1:
            h = np.maximum(h, 0.0)
    return h


# ---------------------------------------------------------------------------
# Internal Z3 helpers
# ---------------------------------------------------------------------------


def _count_parameters(network: TinyNetwork) -> int:
    """Count the total number of parameters in a network."""
    total = 0
    for W, b in zip(network.weights, network.biases):
        total += W.size + b.size
    return total


def _z3_val_to_float(val: Any) -> float:
    """Extract a Python float from a Z3 model value."""
    if val is None:
        return 0.0
    try:
        return float(val.as_decimal(15).rstrip("?"))
    except Exception:
        try:
            frac = val.as_fraction()
            return float(frac)
        except Exception:
            return 0.0


def _encode_network_z3(
    network: TinyNetwork,
    input_vars: list[z3.ArithRef],
) -> list[z3.ArithRef]:
    """Encode a TinyNetwork as Z3 real-valued expressions.

    ReLU is encoded as: relu(x) = If(x >= 0, x, 0)

    Args:
        network:    The network to encode.
        input_vars: Z3 Real variables for each input dimension.

    Returns:
        List of Z3 expressions for the output neurons.
    """
    layer: list[z3.ArithRef] = input_vars
    for i, (W, b) in enumerate(zip(network.weights, network.biases)):
        next_layer: list[z3.ArithRef] = []
        for j in range(W.shape[0]):
            pre_activation: z3.ArithRef = z3.RealVal(float(b[j]))
            for k in range(W.shape[1]):
                pre_activation = pre_activation + z3.RealVal(float(W[j, k])) * layer[k]
            # ReLU on all layers except the last
            if i < len(network.weights) - 1:
                post: z3.ArithRef = z3.If(
                    pre_activation >= z3.RealVal(0.0),
                    pre_activation,
                    z3.RealVal(0.0),
                )
            else:
                post = pre_activation
            next_layer.append(post)
        layer = next_layer
    return layer


# ---------------------------------------------------------------------------
# Verification functions
# ---------------------------------------------------------------------------


def verify_output_bounds(
    network: TinyNetwork,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
    output_min: float | None = None,
    output_max: float | None = None,
) -> VerificationResult:
    """Formally verify output bounds hold for ALL inputs in the range.

    Uses Z3 to search for a counterexample: an input in
    [input_lower, input_upper] whose output violates [output_min, output_max].
    If Z3 finds no counterexample, the property is proven.

    Args:
        network:      The network to verify.
        input_lower:  Lower bound for each input dimension.
        input_upper:  Upper bound for each input dimension.
        output_min:   Minimum allowed output. None means no lower bound.
        output_max:   Maximum allowed output. None means no upper bound.

    Returns:
        VerificationResult with verified=True if the property holds.
    """
    t0 = time.monotonic()
    n_params = _count_parameters(network)

    solver = z3.Solver()
    input_dim = network.layer_sizes[0]
    xs = [z3.Real(f"x_{i}") for i in range(input_dim)]

    # Input bounds
    for i, (lo, hi) in enumerate(zip(input_lower, input_upper)):
        solver.add(xs[i] >= z3.RealVal(float(lo)))
        solver.add(xs[i] <= z3.RealVal(float(hi)))

    outputs = _encode_network_z3(network, xs)

    # Negate the property — search for a violation
    violation_constraints: list[z3.BoolRef] = []
    if output_min is not None:
        for out in outputs:
            violation_constraints.append(out < z3.RealVal(float(output_min)))
    if output_max is not None:
        for out in outputs:
            violation_constraints.append(out > z3.RealVal(float(output_max)))

    if not violation_constraints:
        elapsed = time.monotonic() - t0
        return VerificationResult(
            property_name="output_bounds",
            verified=True,
            counterexample=None,
            time_seconds=elapsed,
            parameters_checked=n_params,
            explanation="No bounds specified — property trivially holds.",
        )

    solver.add(z3.Or(*violation_constraints))
    result = solver.check()
    elapsed = time.monotonic() - t0

    if result == z3.unsat:
        return VerificationResult(
            property_name="output_bounds",
            verified=True,
            counterexample=None,
            time_seconds=elapsed,
            parameters_checked=n_params,
            explanation=(
                f"PROVEN: output is always within bounds for all inputs in range. "
                f"Checked {n_params} parameters."
            ),
        )
    elif result == z3.sat:
        model = solver.model()
        ce: dict[str, float] = {}
        for i, x in enumerate(xs):
            val = model[x]
            ce[f"x_{i}"] = _z3_val_to_float(val)
        return VerificationResult(
            property_name="output_bounds",
            verified=False,
            counterexample=ce,
            time_seconds=elapsed,
            parameters_checked=n_params,
            explanation=(
                f"COUNTEREXAMPLE FOUND: an input exists that violates the bounds. "
                f"Checked {n_params} parameters."
            ),
        )
    else:
        return VerificationResult(
            property_name="output_bounds",
            verified=False,
            counterexample=None,
            time_seconds=elapsed,
            parameters_checked=n_params,
            explanation="Z3 returned UNKNOWN — verification inconclusive.",
        )


def verify_monotonicity(
    network: TinyNetwork,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
    input_dimension: int,
    output_dimension: int = 0,
) -> VerificationResult:
    """Formally verify monotonicity in one input dimension.

    Encodes two input vectors x1 and x2 that agree on all dimensions except
    ``input_dimension``, where x1[dim] <= x2[dim]. Searches for a
    counterexample where f(x1)[output_dim] > f(x2)[output_dim].

    Args:
        network:          The network to verify.
        input_lower:      Lower bound for each input dimension.
        input_upper:      Upper bound for each input dimension.
        input_dimension:  Which input dimension to check monotonicity for.
        output_dimension: Which output neuron to check.

    Returns:
        VerificationResult with verified=True if monotonicity holds.
    """
    t0 = time.monotonic()
    n_params = _count_parameters(network)

    solver = z3.Solver()
    input_dim = network.layer_sizes[0]

    xs1 = [z3.Real(f"x1_{i}") for i in range(input_dim)]
    xs2 = [z3.Real(f"x2_{i}") for i in range(input_dim)]

    # Both inputs within bounds
    for i in range(input_dim):
        lo, hi = float(input_lower[i]), float(input_upper[i])
        solver.add(xs1[i] >= z3.RealVal(lo), xs1[i] <= z3.RealVal(hi))
        solver.add(xs2[i] >= z3.RealVal(lo), xs2[i] <= z3.RealVal(hi))

    # All dimensions equal except input_dimension
    for i in range(input_dim):
        if i != input_dimension:
            solver.add(xs1[i] == xs2[i])

    # x1[dim] <= x2[dim]
    solver.add(xs1[input_dimension] <= xs2[input_dimension])

    out1 = _encode_network_z3(network, xs1)
    out2 = _encode_network_z3(network, xs2)

    # Negate monotonicity: find where f(x1)[d] > f(x2)[d]
    solver.add(out1[output_dimension] > out2[output_dimension])

    result = solver.check()
    elapsed = time.monotonic() - t0

    prop_name = f"monotonicity_dim{input_dimension}"

    if result == z3.unsat:
        return VerificationResult(
            property_name=prop_name,
            verified=True,
            counterexample=None,
            time_seconds=elapsed,
            parameters_checked=n_params,
            explanation=(
                f"PROVEN: output dimension {output_dimension} is non-decreasing "
                f"with respect to input dimension {input_dimension} "
                f"for all inputs in range."
            ),
        )
    elif result == z3.sat:
        model = solver.model()
        ce: dict[str, float] = {}
        for i, (x1, x2) in enumerate(zip(xs1, xs2)):
            v1 = model[x1]
            v2 = model[x2]
            ce[f"x1_{i}"] = _z3_val_to_float(v1)
            ce[f"x2_{i}"] = _z3_val_to_float(v2)
        return VerificationResult(
            property_name=prop_name,
            verified=False,
            counterexample=ce,
            time_seconds=elapsed,
            parameters_checked=n_params,
            explanation=(
                f"COUNTEREXAMPLE FOUND: increasing input dimension {input_dimension} "
                f"can decrease output dimension {output_dimension}."
            ),
        )
    else:
        return VerificationResult(
            property_name=prop_name,
            verified=False,
            counterexample=None,
            time_seconds=elapsed,
            parameters_checked=n_params,
            explanation="Z3 returned UNKNOWN — verification inconclusive.",
        )


# ---------------------------------------------------------------------------
# Constrained AI
# ---------------------------------------------------------------------------


class ConstrainedAI:
    """The Constrained AI pattern: a verified kernel monitors an unverified model.

    The frontier model does the thinking. The mathematically proven kernel
    holds the kill switch.

    Args:
        model:          Unverified model: callable taking np.ndarray, returning float.
        safety_kernel:  A formally verified TinyNetwork.
        kernel_bounds:  (min, max) — the verified output bounds of the kernel.
    """

    def __init__(
        self,
        model: Callable[[np.ndarray], float],
        safety_kernel: TinyNetwork,
        kernel_bounds: tuple[float, float],
    ) -> None:
        self._model = model
        self._kernel = safety_kernel
        self._bounds = kernel_bounds
        self._kernel_verified = False

    def verify_kernel(
        self,
        input_lower: np.ndarray,
        input_upper: np.ndarray,
    ) -> VerificationResult:
        """Run Z3 verification on the safety kernel and record the result."""
        result = verify_output_bounds(
            self._kernel,
            input_lower,
            input_upper,
            output_min=self._bounds[0],
            output_max=self._bounds[1],
        )
        self._kernel_verified = result.verified
        return result

    @property
    def kernel_verified(self) -> bool:
        """Whether the safety kernel has been formally verified."""
        return self._kernel_verified

    def predict(self, x: np.ndarray) -> SafetyKernelResult:
        """Run the model with safety kernel oversight."""
        model_out = float(self._model(x))
        kernel_out = float(forward(self._kernel, x)[0])

        lo, hi = self._bounds
        within_bounds = lo <= model_out <= hi
        override = not within_bounds

        final = kernel_out if override else model_out

        if override:
            explanation = (
                f"OVERRIDE: model output {model_out:.3f} is outside verified bounds "
                f"[{lo}, {hi}]. Kernel output {kernel_out:.3f} used instead."
            )
        else:
            explanation = (
                f"Model output {model_out:.3f} is within verified bounds [{lo}, {hi}]. "
                f"No override required."
            )

        return SafetyKernelResult(
            model_output=model_out,
            kernel_output=kernel_out,
            kernel_verified=self._kernel_verified,
            override_triggered=override,
            final_output=final,
            explanation=explanation,
        )


# ---------------------------------------------------------------------------
# Scaling experiment
# ---------------------------------------------------------------------------


def measure_verification_scaling(
    sizes: list[tuple[int, ...]] | None = None,
) -> list[dict[str, float | int | str]]:
    """Demonstrate the state-space explosion problem.

    Verifies output bounds for networks of increasing size and records timing.

    Args:
        sizes: Architectures to test. Defaults to a progression showing scaling.

    Returns:
        List of dicts with keys: architecture, parameters, time_seconds, verified.
    """
    if sizes is None:
        sizes = [
            (2, 2, 1),
            (2, 4, 1),
            (2, 8, 1),
            (2, 16, 1),
            (2, 8, 4, 1),
            (2, 16, 8, 1),
        ]

    results: list[dict[str, float | int | str]] = []
    for size in sizes:
        net = build_tiny_network(size, seed=42)
        lo = np.zeros(size[0])
        hi = np.ones(size[0])
        r = verify_output_bounds(net, lo, hi, output_min=-100.0, output_max=100.0)
        results.append(
            {
                "architecture": str(size),
                "parameters": _count_parameters(net),
                "time_seconds": r.time_seconds,
                "verified": r.verified,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    """Run the formal verification demonstration."""
    print("Layer 09: Formal Verification")
    print("=" * 60)
    print()
    print("The gold standard nobody can fully use yet.")
    print()

    # ------------------------------------------------------------------
    # 1. Proof of output bounds
    # ------------------------------------------------------------------
    print("--- Demo 1: Output Bounds Verification ---")
    print()
    net = build_tiny_network((2, 4, 1), seed=42)
    lo = np.zeros(2)
    hi = np.ones(2)

    result = verify_output_bounds(net, lo, hi, output_min=-100.0, output_max=100.0)
    print(f"Network: (2, 4, 1) | Parameters: {result.parameters_checked}")
    print("Input range: [0,1] x [0,1]")
    print("Bounds: [-100, 100]")
    print(f"Verified: {result.verified}")
    print(f"Time: {result.time_seconds:.4f}s")
    print(f"Result: {result.explanation}")
    print()

    # Also show a failed verification
    actual_out = float(forward(net, np.array([0.5, 0.5]))[0])
    tight_min = actual_out + 1.0
    tight_max = actual_out + 2.0
    result_tight = verify_output_bounds(net, lo, hi, output_min=tight_min, output_max=tight_max)
    print(f"Tight bounds [{tight_min:.3f}, {tight_max:.3f}] (excluding actual output):")
    print(f"Verified: {result_tight.verified}")
    if result_tight.counterexample:
        print(f"Counterexample: {result_tight.counterexample}")
    print()

    # ------------------------------------------------------------------
    # 2. Monotonicity verification
    # ------------------------------------------------------------------
    print("--- Demo 2: Monotonicity Verification ---")
    print()

    # A known monotone network (all-positive weights, single layer)
    mono_net = TinyNetwork(
        weights=[np.array([[1.0, 1.0]])],
        biases=[np.array([0.0])],
        layer_sizes=(2, 1),
    )
    mono_result = verify_monotonicity(mono_net, lo, hi, input_dimension=0)
    print("Network with positive weights (2->1):")
    print(f"Verified monotone in dim 0: {mono_result.verified}")
    print(f"Result: {mono_result.explanation}")
    print()

    # A random network (likely non-monotone)
    rand_result = verify_monotonicity(net, lo, hi, input_dimension=0)
    print("Random (2,4,1) network:")
    print(f"Verified monotone in dim 0: {rand_result.verified}")
    print(f"Result: {rand_result.explanation}")
    if rand_result.counterexample:
        print(f"Counterexample keys: {list(rand_result.counterexample.keys())}")
    print()

    # ------------------------------------------------------------------
    # 3. State-space explosion
    # ------------------------------------------------------------------
    print("--- Demo 3: State-Space Explosion ---")
    print()
    scaling_results = measure_verification_scaling()
    print(f"{'Architecture':<20} {'Parameters':>12} {'Time (s)':>10} {'Verified':>10}")
    print("-" * 56)
    for r in scaling_results:
        print(
            f"{r['architecture']:<20} {r['parameters']:>12} "
            f"{r['time_seconds']:>10.4f} {str(r['verified']):>10}"
        )
    print()

    # ------------------------------------------------------------------
    # 4. Constrained AI pattern
    # ------------------------------------------------------------------
    print("--- Demo 4: Constrained AI Pattern ---")
    print()

    kernel = build_tiny_network((2, 4, 1), seed=42)

    # Unverified model that amplifies the kernel output by 3x — exceeds tight bounds for many inputs
    def unverified_model(x: np.ndarray) -> float:
        return float(forward(kernel, x)[0]) * 3.0

    # Verify the kernel first against [-1.0, 1.0] (the kernel's provable range over [0,1]^2)
    cai_wide = ConstrainedAI(unverified_model, kernel, (-1.0, 1.0))
    verify_result = cai_wide.verify_kernel(lo, hi)
    print(f"Kernel verification (bounds [-1.0, 1.0]): {verify_result.explanation}")
    print(f"Kernel verified: {cai_wide.kernel_verified}")
    print()

    # Now demonstrate override with tight bounds
    cai_tight = ConstrainedAI(unverified_model, kernel, (-0.3, 0.3))

    # Input where model is within bounds (kernel out near zero → *3 still ≤ 0.3)
    x_safe = np.array([0.35, 0.65])
    safe_result = cai_tight.predict(x_safe)
    print("Input [0.35, 0.65] - model likely within tight bounds:")
    print(f"  Model output (3x amplified): {safe_result.model_output:.3f}")
    print(f"  Kernel output: {safe_result.kernel_output:.3f}")
    print(f"  Override triggered: {safe_result.override_triggered}")
    print(f"  Final output: {safe_result.final_output:.3f}")
    print(f"  {safe_result.explanation}")
    print()

    # Input where model exceeds bounds (large kernel output → *3 exceeds 0.3)
    x_unsafe = np.array([1.0, 1.0])
    unsafe_result = cai_tight.predict(x_unsafe)
    print("Input [1.0, 1.0] - model exceeds tight bounds:")
    print(f"  Model output (3x amplified): {unsafe_result.model_output:.3f}")
    print(f"  Kernel output: {unsafe_result.kernel_output:.3f}")
    print(f"  Override triggered: {unsafe_result.override_triggered}")
    print(f"  Final output: {unsafe_result.final_output:.3f}")
    print(f"  {unsafe_result.explanation}")
    print()

    print("=" * 60)
    print("Formal verification: real proofs, real constraints, real limits.")


if __name__ == "__main__":  # pragma: no cover
    main()
