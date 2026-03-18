# Layer 01: Inverse Confidence Weighting

*Surface your weakest signal, not your average*

> **Primary author:** Ant Newman
>
> **Production examples:** Generalised from [agent-task-planning](https://github.com/antnewman/pda-platform/tree/main/packages/agent-task-planning) in the PDA Platform

---

## What It Is

When an AI system produces confidence scores across multiple output fields, those scores need to be aggregated into a single measure that a human or downstream system can act on. The standard approach is to average them. This is a mistake.

Inverse confidence weighting is an aggregation method that gives more influence to the fields the model is least confident about. Instead of treating every field equally, the weight assigned to each field is inversely proportional to its confidence: low-confidence fields pull the aggregate down, high-confidence fields have less influence.

The formula is straightforward. For each field, the weight is calculated as `weight = 2.0 - confidence`. This maps confidence values in the range [0.0, 1.0] to weights in the range [1.0, 2.0] — a field with zero confidence gets the maximum weight of 2.0, whilst a field with perfect confidence gets the minimum weight of 1.0. The aggregate is then the weighted sum of all confidence values divided by the total weight.

The result is always bounded between the minimum and maximum input confidence values. It is always a valid confidence score in [0.0, 1.0]. And critically, it is always lower than or equal to the plain arithmetic mean when confidence values are not uniform — because the weakest signals carry disproportionate influence.

This is a values choice disguised as a formula. The decision to weight inversely is a decision about what kind of system you want to build: one that presents its best face, or one that surfaces its weakest point.

## Why It Matters

Consider a system that extracts structured data from documents — dates, amounts, locations, categories, names. It processes a document and returns confidence scores for each extracted field:

| Field | Confidence |
|---|---|
| date | 0.95 |
| location | 0.92 |
| amount | 0.88 |
| category | 0.30 |

The plain arithmetic mean is **0.7625**. That looks like a reasonably confident extraction. A downstream system using that score to decide whether to auto-process or escalate to a human might let it through.

But the inverse-weighted aggregate is **0.7228**. More importantly, the gap between the two numbers tells you something: the system is hiding a problem. One field is deeply uncertain, and the average is masking it.

Now scale this to a system processing thousands of documents per hour. Every document where one critical field is uncertain but the average looks acceptable is a document that passes through without human review. In financial services, that is a compliance risk. In healthcare, it is a patient safety risk. In any domain where the cost of a confident wrong answer exceeds the cost of a flagged uncertain one, averaging is the wrong aggregation method.

The failure mode is specific and predictable: **a single uncertain field in an otherwise confident extraction produces an overall score that looks safe when it is not.** Inverse confidence weighting exists to make that failure mode visible.

## Production Example

The implementation is open source in the Verified Autonomy repository at [`implementations/layer_01_confidence_weighting/`](https://github.com/antnewman/verified-autonomy/tree/main/implementations/layer_01_confidence_weighting).

**Run it yourself in under two minutes:**

```bash
cd implementations/layer_01_confidence_weighting
make install
make run
```

**The core function:**

```python
from verified_autonomy.confidence_weighting import inverse_confidence_weight

scores = {
    "date":     0.95,
    "location": 0.92,
    "amount":   0.88,
    "category": 0.30,
}

result = inverse_confidence_weight(scores)
# Returns 0.7228 — below the plain mean of 0.7625
```

The function takes a dictionary mapping field names to confidence scores in [0.0, 1.0] and returns a single weighted aggregate. It validates all inputs, raising a `ValueError` with the offending field name if any confidence score is outside the valid range. It returns 0.0 for an empty dictionary — no fields means no confidence.

**Key behaviours demonstrated by the test suite (31 tests, 100% coverage):**

When all fields have equal confidence, the aggregate equals that common value — the weighting has no distorting effect. When one field is weak, the aggregate is pulled below the plain mean. When a field has near-zero confidence (0.01) among otherwise strong fields (0.92–0.95), the aggregate drops well below the mean. The result is always bounded between the minimum and maximum input values — it never produces a score outside the range of its inputs.

The boundary case is revealing: when one field has confidence 0.0 and another has confidence 1.0, the plain mean would be 0.5. The inverse-weighted aggregate is 0.333 — pulled toward the uncertain field because that field's weight (2.0) is double the certain field's weight (1.0). The system reports that it is more uncertain than certain. This is the correct behaviour.

**Code ancestry:** This implementation is generalised from the `agent-task-planning` package in the [PDA Platform](https://github.com/antnewman/pda-platform), where it was originally used to aggregate confidence scores across project management data fields. The algorithm is identical; the domain-specific terminology has been removed to make it applicable to any AI system that produces per-field confidence scores.

## Limitations

**Increases escalation rate.** By design, inverse confidence weighting produces lower aggregate scores than plain averaging. In a system where the aggregate score determines whether an output is auto-processed or escalated to a human reviewer, this means more outputs will be escalated. If reviewer capacity is constrained, this can create a bottleneck. The technique trades throughput for honesty — that is a resource allocation decision, not just a technical one.

**Requires per-field confidence scores.** The technique only works when the model produces separate confidence scores for each output field. A model that produces a single overall confidence score, or no confidence score at all, cannot use this approach. Extracting per-field confidence from models that do not natively provide it (e.g. by running multiple samples and measuring agreement) is a separate engineering problem addressed in Layer 02.

**Assumes fields carry roughly equal importance.** The weighting formula treats all fields symmetrically — a low-confidence score in a trivial field has the same pulling effect as a low-confidence score in a critical field. In domains where some fields matter more than others (e.g. a patient identifier matters more than a formatting field), an additional importance weighting layer is needed on top of the inverse confidence weighting. The current implementation does not include this.

**The weight function is linear, not calibrated.** The formula `weight = 2.0 - confidence` is a linear mapping. It does not account for the possibility that confidence scores from the underlying model may not be well-calibrated — a model that reports 0.9 confidence may only be correct 70% of the time (see Layer 04: Calibration and Conformal Prediction). Inverse confidence weighting amplifies the signal from low-confidence fields, but if the confidence scores themselves are unreliable, the amplified signal is also unreliable. This technique works best when combined with a calibration layer.

**Does not distinguish between "uncertain" and "wrong".** A field with low confidence might be uncertain because the input data is ambiguous, or it might be uncertain because the model is about to hallucinate. Inverse confidence weighting treats both cases the same — it flags them for review. The distinction between ambiguity and error requires additional techniques (see Layer 03: Making Failures Visible).

## Reader Takeaway

When you aggregate confidence scores, weight toward the weakest signal, not the average. The formula is simple (`weight = 2.0 - confidence`), the implementation is a single pure-Python function with no dependencies, and the effect is immediate: your system starts telling you where it is least confident instead of hiding it behind a reassuring number. This is the foundational design principle of the entire nine-layer architecture — trust is built by making uncertainty visible, not by smoothing it away.

---

*Contributor note: Ant Newman wrote the primary draft. Implementation generalised from the PDA Platform agent-task-planning package by Ant Newman. Lawrence Rowland contributed the original requirements and conceptual design for the confidence extraction capability.*
