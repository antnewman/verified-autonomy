# Layer 09: Formal Verification

*The gold standard nobody can fully use yet*

> **Primary author:** Ant Newman
>
> **Production examples:** Reference implementation using Z3 SMT solver; field examples from aerospace, autonomous vehicles, and the Constrained AI pattern

---

## What It Is

Every other layer in this field guide tests, scores, checks, or monitors. This layer proves.

Formal verification applies mathematical proof techniques to neural networks. It does not test the network against a sample of inputs and report a pass rate. It proves that the network cannot enter a defined forbidden state for any possible input within a specified range. Not for most inputs. Not for a statistically representative sample. For all of them. The output is not a confidence score. It is a proof — or a counterexample that demonstrates exactly where the property fails.

The mechanism works through Satisfiability Modulo Theories (SMT) solvers. The network is encoded as a set of mathematical constraints: each neuron becomes a real-valued expression, each ReLU activation becomes a conditional (`If(x >= 0, x, 0)`), and the property to verify becomes an additional constraint to check. The solver exhaustively searches the constraint space. If it finds an input that violates the property, it returns a counterexample — an exact input value where the network misbehaves. If no such input exists, the property is proven for the entire input range.

The implementation demonstrates two properties that can be formally verified:

**Output bounds verification** proves that the network's output always stays within a specified range for all inputs in a bounded region. For example: "for any input where x₁ ∈ [0, 1] and x₂ ∈ [0, 1], the output is always between −5.0 and 5.0." If the proof succeeds, this holds for every one of the infinitely many inputs in that region. If it fails, the solver returns the exact input that breaks it.

**Monotonicity verification** proves that the network's output is non-decreasing with respect to a specific input dimension. For example: "increasing input dimension 0 never decreases the output." This matters in domains where monotonicity is a physical or regulatory requirement — a dosage calculator should never recommend a lower dose for a sicker patient, a risk scorer should never assign lower risk to a more dangerous input.

The catch — and it is a substantial one — is the state-space explosion. Each ReLU neuron doubles the number of possible activation patterns the solver must explore, because each neuron can be in its active region (output = input) or its inactive region (output = 0). A network with n ReLU neurons has up to 2ⁿ activation patterns. This grows exponentially. A 17-parameter network with 4 ReLU neurons verifies in under a second. A 193-parameter network with 24 ReLU neurons takes nearly a minute. A frontier model with billions of parameters and millions of ReLU activations is beyond any solver that exists or is likely to exist in the near future.

This is why the subtitle is "the gold standard nobody can fully use yet." The technique delivers the strongest guarantee in the field guide — a mathematical proof, not a test — but only for networks small enough to verify.

## Why It Matters

The realistic 2026 implementation is not "formally verify the frontier model." It is the **Constrained AI pattern**: a small, formally verified safety kernel that monitors and can override a larger unverified model.

The frontier model does the thinking. It is large, capable, and opaque. Its outputs are useful but unproven. The safety kernel is small, limited, and mathematically verified. Its output bounds have been formally proven — there exist inputs for which the proof guarantees the kernel's output stays within a specified safe range. When the frontier model's output falls outside the kernel's verified bounds, the kernel overrides it.

This is the architectural equivalent of a constitutional court. The legislature (the frontier model) proposes laws. The court (the verified kernel) checks whether they are constitutional (within proven bounds). The court cannot write laws — it is too small and limited for that. But it can veto laws that violate the constitution, and its authority to do so rests on a formal proof, not an opinion.

In aerospace, this pattern is already production reality. NASA, Boeing, and Airbus use formally verified software in safety-critical flight systems — not the entire avionics software stack, but the specific components where a proof of correctness is required by regulation (DO-178C Level A). The verified component is small and constrained. The larger system around it is tested but not proven. The verified component holds the kill switch.

In autonomous vehicles, braking systems use a similar pattern. The perception model (large, neural, unproven) proposes a braking decision. A formally verified safety module checks whether the proposed action could result in a collision given the current state. If it could, the safety module overrides with a proven-safe braking response.

The failure mode is specific: **without formal verification, even a well-tested system has unknown failure modes. Testing proves the system works for the inputs you tested. Formal verification proves it works for all inputs in the range.** The gap between "tested" and "proven" is exactly the gap that adversarial attackers and edge cases exploit — the inputs nobody thought to test.

## Production Example

The implementation is open source in the Verified Autonomy repository at [`implementations/layer_09_formal_verification/`](https://github.com/antnewman/verified-autonomy/tree/main/implementations/layer_09_formal_verification).

**Run it yourself:**

```bash
cd implementations/layer_09_formal_verification
make install
make run
```

Note: `make run` takes 30–60 seconds because Z3 is performing genuine SMT solving, not simulation. This is the computational cost of a real proof.

**The four-step demonstration from `make run`:**

**Demo 1 — Output bounds verification.** A (2, 4, 1) network with 17 parameters is verified against bounds of [−100, 100] for all inputs in [0, 1] × [0, 1]. Result: PROVEN. The output cannot exceed those bounds for any input in the range. Time: ~0.07 seconds. Then the same network is verified against deliberately tight bounds that exclude its actual output range. Result: NOT VERIFIED. Z3 returns a counterexample — an exact input (e.g. x₀ = 0.055, x₁ = 0.0) where the output falls outside the tight bounds. The counterexample is genuine: running `forward(network, counterexample)` confirms the violation.

**Demo 2 — Monotonicity verification.** A hand-crafted network with all-positive weights is verified as monotonically non-decreasing in input dimension 0. Result: PROVEN. A random (2, 4, 1) network is verified for the same property. Result: NOT VERIFIED, with a counterexample showing two specific inputs where increasing dimension 0 decreases the output.

**Demo 3 — The state-space explosion.** Networks of increasing size are verified against the same property, with timing:

```
Architecture           Parameters   Time (s)
(2, 2, 1)                       9     0.03
(2, 4, 1)                      17     0.05
(2, 8, 1)                      33     0.57
(2, 16, 1)                     65     2.37
(2, 8, 4, 1)                   65     5.89
(2, 16, 8, 1)                 193    46.35
```

A 193-parameter network takes roughly 1,500 times longer than a 9-parameter network. Frontier models have billions of parameters. The scaling wall is real and fundamental.

**Demo 4 — The Constrained AI pattern.** A verified safety kernel (17 parameters, output bounds formally proven) monitors an unverified model (the same kernel amplified by 3×, simulating a less constrained model). When the model's output is within the kernel's verified bounds (−0.3 to 0.3), the model's output is used. When the model exceeds those bounds (output 1.712 vs bound of 0.3), the kernel overrides: final output becomes the kernel's output (0.571) instead.

```python
from verified_autonomy.formal_verification import (
    ConstrainedAI, build_tiny_network, forward,
)
import numpy as np

kernel = build_tiny_network((2, 4, 1), seed=42)

def unverified_model(x: np.ndarray) -> float:
    return float(forward(kernel, x)[0]) * 3.0

cai = ConstrainedAI(unverified_model, kernel, bounds=(-0.3, 0.3))
cai.verify_kernel(
    input_lower=np.zeros(2),
    input_upper=np.ones(2),
)

result = cai.predict(np.array([1.0, 1.0]))
result.override_triggered  # True
result.explanation          # "OVERRIDE: model output 1.712 is outside verified bounds..."
```

**Code ancestry:** The Z3 encoding follows the standard approach for ReLU network verification: each neuron is a real-valued expression, each ReLU is a conditional, and the solver searches for counterexamples. The implementation is built from scratch using the `z3-solver` Python bindings. No external verification framework is used — the Z3 encoding is fully visible and auditable.

## Limitations

**State-space explosion: not applicable to frontier models.** The fundamental limitation is computational. Each ReLU neuron doubles the potential activation patterns. A network with 24 ReLU neurons (193 parameters) takes nearly a minute. A network with 1,000 ReLU neurons would take longer than the age of the universe. Formal verification of complete frontier models with billions of parameters is not feasible with current SMT-based techniques and may not become feasible without a fundamental algorithmic breakthrough.

**Can only prove explicitly specified properties.** Formal verification proves that a specific, mathematically stated property holds. It cannot verify vague properties like "the model is fair" or "the output is helpful" — these must be formalised into precise mathematical constraints before verification can even begin. The gap between what practitioners want to verify ("is this model safe?") and what can be stated formally ("does the output stay within [−5, 5] for all inputs in [0, 1]²?") is substantial. Property specification is an engineering discipline in its own right.

**Requires specialist expertise.** Encoding a neural network as SMT constraints, choosing appropriate properties to verify, interpreting counterexamples, and understanding the limits of the verification are specialist skills rare in AI engineering teams. Most ML engineers have never used an SMT solver. The gap between the technique's theoretical power and its practical accessibility is a significant adoption barrier.

**The Constrained AI pattern introduces a capability ceiling.** The verified safety kernel can only override outputs — it cannot improve them. If the frontier model produces an output outside the kernel's bounds, the kernel substitutes its own output, which is safe but limited. The system becomes as capable as the kernel in those cases, which may be insufficient for the task. The trade-off between safety and capability is explicit: tighter kernel bounds mean more overrides and less capability from the frontier model.

**Verification is static; models are updated.** A formal proof applies to a specific set of weights. If the model is retrained, fine-tuned, or updated in any way, the proof is invalidated and must be re-run. For the safety kernel in the Constrained AI pattern, this means the kernel's weights must be frozen after verification — it cannot be updated without re-verification. This creates a tension between keeping the kernel current and maintaining its verified status.

**The gap between what we can prove and what we can build.** This is the honest state of the field. We can build models with billions of parameters that exhibit remarkable capabilities. We can formally verify models with a few hundred parameters. The nine-layer architecture exists because of this gap — layers 01 through 08 provide the defence-in-depth that formal verification alone cannot yet deliver. Layer 09 is the aspiration: the day when we can prove, not just test, that our systems behave correctly. Until then, it provides the kill switch.

## Reader Takeaway

Formal verification is the only technique in this field guide that provides a mathematical proof rather than a test result. The Constrained AI pattern makes it practical today: the frontier model does the thinking, the mathematically proven safety kernel holds the kill switch. But be honest about the limits — verification scales to hundreds of parameters, not billions. The gap between what we can prove and what we can build is the defining challenge of AI trust engineering, and it is why this field guide has nine layers instead of one. Trust is built not by waiting for formal verification to scale, but by layering every available technique — confidence scoring, outlier detection, visible failures, calibration, guardrails, explainability, adversarial testing, audit trails, and formal verification — so that each layer compensates for the failure modes of the others.

---

*Contributor note: Ant Newman wrote the primary draft and implementation. Uses the Z3 SMT solver for genuine formal verification proofs. See ADR 009 for the z3-solver dependency rationale.*
