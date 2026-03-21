# Layer 09: Formal Verification — Implementation

## Quick Start

```bash
pip install -e ".[dev]"
python src/formal_verification.py
```

## Expected Output

```
Layer 09: Formal Verification
============================================================

The gold standard nobody can fully use yet.

--- Demo 1: Output Bounds Verification ---

Network: (2, 4, 1) | Parameters: 17
Input range: [0,1] x [0,1]
Bounds: [-100, 100]
Verified: True
Time: 0.0705s
Result: PROVEN: output is always within bounds for all inputs in range. Checked 17 parameters.

Tight bounds [0.845, 1.845] (excluding actual output):
Verified: False
Counterexample: {'x_0': 0.055136907710682, 'x_1': 0.0}

--- Demo 2: Monotonicity Verification ---

Network with positive weights (2->1):
Verified monotone in dim 0: True
Result: PROVEN: output dimension 0 is non-decreasing with respect to input dimension 0 for all inputs in range.

Random (2,4,1) network:
Verified monotone in dim 0: False
Result: COUNTEREXAMPLE FOUND: increasing input dimension 0 can decrease output dimension 0.
Counterexample keys: ['x1_0', 'x2_0', 'x1_1', 'x2_1']

--- Demo 3: State-Space Explosion ---

Architecture           Parameters   Time (s)   Verified
--------------------------------------------------------
(2, 2, 1)                       9     0.0287       True
(2, 4, 1)                      17     0.0497       True
(2, 8, 1)                      33     0.5653       True
(2, 16, 1)                     65     2.3719       True
(2, 8, 4, 1)                   65     5.8904       True
(2, 16, 8, 1)                 193    46.3515       True

--- Demo 4: Constrained AI Pattern ---

Kernel verification (bounds [-1.0, 1.0]): PROVEN: output is always within bounds for all inputs in range. Checked 17 parameters.
Kernel verified: True

Input [0.35, 0.65] - model likely within tight bounds:
  Model output (3x amplified): -0.292
  Kernel output: -0.097
  Override triggered: False
  Final output: -0.292
  Model output -0.292 is within verified bounds [-0.3, 0.3]. No override required.

Input [1.0, 1.0] - model exceeds tight bounds:
  Model output (3x amplified): 1.712
  Kernel output: 0.571
  Override triggered: True
  Final output: 0.571
  OVERRIDE: model output 1.712 is outside verified bounds [-0.3, 0.3]. Kernel output 0.571 used instead.

============================================================
Formal verification: real proofs, real constraints, real limits.
```

Note: Demo 3 timing varies by machine. Verification time grows super-linearly with network size — a 193-parameter network takes roughly 1000x longer than a 9-parameter network on the same hardware.

## What This Code Does

This module demonstrates three core ideas in formal verification of neural networks using the Z3 SMT solver. First, it shows how to encode a small ReLU network as a system of real-valued constraints and use Z3 to either prove that an output property holds for every input in a bounded region, or find a genuine counterexample that violates it — not an approximation, but an exact proof. Second, it measures the state-space explosion: verification time grows super-linearly with network size, which is why formal methods remain impractical for production-scale models today. Third, it implements the Constrained AI pattern: a small, formally verified safety kernel that monitors an unverified frontier model and can override its output when it steps outside the proven safe region.

## API

| Symbol | Kind | Description |
|---|---|---|
| `TinyNetwork` | dataclass | Small ReLU network stored as numpy weight/bias arrays |
| `VerificationResult` | dataclass | Outcome of a Z3 verification query |
| `SafetyKernelResult` | dataclass | Outcome of a `ConstrainedAI.predict()` call |
| `ConstrainedAI` | class | Verified kernel monitoring an unverified model |
| `build_tiny_network(layer_sizes, seed)` | function | Construct a network with He-initialised weights |
| `forward(network, x)` | function | Pure-numpy forward pass with ReLU activations |
| `verify_output_bounds(network, lo, hi, output_min, output_max)` | function | Prove output stays within bounds for all inputs in range |
| `verify_monotonicity(network, lo, hi, input_dimension, output_dimension)` | function | Prove monotonicity in one input dimension |
| `measure_verification_scaling(sizes)` | function | Demonstrate the state-space explosion with timing data |

## Testing

```bash
python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=90
```

The test suite invokes real Z3 and takes several minutes to run in full. This is expected: Z3 is doing genuine SMT solving, not simulation.

Coverage target: 90% minimum. The `main()` function is excluded with `# pragma: no cover`.

## Dependencies

| Package | Version | Why |
|---|---|---|
| `numpy` | >=1.26 | Network weight storage and forward pass arithmetic |
| `z3-solver` | >=4.12 | SMT solver for formal verification proofs |

### Note on z3-solver

`z3-solver` is a compiled package with platform-specific binaries. It is pip-installable on Windows, macOS, and Linux but takes longer to install than pure-Python packages. This is a one-off cost. See [ADR 009](../../../adr/009-z3-dependency.md) for the decision rationale.
