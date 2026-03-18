# Layer 02: Outlier Detection as Hard Escalation

*Statistical disagreement is a signal, not noise*

> **Primary author:** Ant Newman
>
> **Production examples:** Generalised from [agent-task-planning](https://github.com/antnewman/pda-platform/tree/main/packages/agent-task-planning) in the PDA Platform

---

## What It Is

When you run multiple model samples over the same input and they disagree about a specific value, the disagreement itself contains information. Most systems average that disagreement away. This layer treats it as a routing signal.

Outlier detection as hard escalation uses Tukey's Inter-Quartile Range (IQR) fence method to identify when one or more model outputs diverge significantly from the group consensus. Values outside the fence `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` are flagged as statistical outliers. Each flagged value carries a human-readable label, its divergence score, and a plain-English reason explaining why it was flagged.

The key architectural decision is what happens next. The system does not log the outlier and move on. It does not downweight the outlier and recalculate. It treats the outlier as a hard escalation trigger — the output is routed to a human reviewer before it is acted upon.

The escalation routing uses a four-tier system:

| Tier | Trigger | What Happens |
|---|---|---|
| **EXPERT_REQUIRED** | Any outlier detected OR confidence < 0.4 | Route to a domain expert. Output is not acted upon until reviewed. |
| **DETAILED_REVIEW** | Confidence between 0.4 and 0.6 | A human reads every output in this tier. |
| **SPOT_CHECK** | Confidence between 0.6 and 0.8 | Periodic sampling — a proportion of outputs are reviewed. |
| **NONE** | Confidence ≥ 0.8 and no outliers | No human review required. Auto-process. |

The critical design choice is the **OR condition**: outliers OR low confidence, each independently, trigger escalation. This means the system fails safe. A high-confidence output with an outlier still goes to an expert. A low-confidence output with no outliers still goes to an expert. Both paths lead to human review. The only way an output avoids review is if confidence is high AND no statistical disagreement exists.

## Why It Matters

Consider a system that runs five model samples over the same input to extract a numerical value — a cost estimate, a measurement, a dosage. Four samples return values tightly clustered around 10.1. The fifth returns 80.0.

The average of those five values is 24.2 — a number that bears no relationship to reality. Even the median (10.2) looks clean, but it hides the fact that one sample produced a wildly different answer. That disagreement is not noise. The model is disagreeing with itself, and there is a reason.

If the system averages away the disagreement, the output looks confident and is auto-processed. If the system uses the median, the output looks clean and the disagreement is invisible. In both cases, the signal that something is wrong is destroyed.

In financial services, that destroyed signal might be a misread figure in a regulatory filing. In healthcare, it might be a dosage calculation where one model path produced a dangerously different result. In any domain where the cost of acting on a wrong answer exceeds the cost of pausing for human review, silently smoothing disagreement is the wrong architectural choice.

The failure mode is specific: **averaging or medianing away model disagreement destroys the signal that the model is uncertain about this specific output, producing a clean-looking result that hides a genuine problem.**

## Production Example

The implementation is open source in the Verified Autonomy repository at [`implementations/layer_02_outlier_escalation/`](https://github.com/antnewman/verified-autonomy/tree/main/implementations/layer_02_outlier_escalation).

**Run it yourself in under two minutes:**

```bash
cd implementations/layer_02_outlier_escalation
make install
make run
```

**The core function:**

```python
from verified_autonomy.outlier_escalation import detect_outliers, EscalationLevel

result = detect_outliers(
    values=[10.0, 10.2, 10.1, 10.0, 10.1, 10.2, 100.0],
    labels=["agent_A", "agent_B", "agent_C", "agent_D", "agent_E", "agent_F", "agent_G"],
    confidence=0.95,
)

result.escalation_level   # EscalationLevel.EXPERT_REQUIRED
result.consensus           # 10.1 (median — the agreed reference point)
result.has_outliers        # True
result.outliers[0].label   # "agent_G"
result.outliers[0].reason  # "agent_G (100) is above the upper fence (10.5); consensus is 10.1"
```

The function takes a list of numeric values (model sample outputs), optional human-readable labels, and an overall confidence score. It returns a `DetectionResult` containing the escalation level, the consensus value (median), and a list of `OutlierFlag` objects for every value that fell outside the IQR fence. Each flag includes the value, the consensus it was compared against, a normalised divergence score between 0.0 and 1.0, and a plain-English reason string that can be shown directly to a human reviewer.

**The OR condition in action (from the test suite, 44 tests, 98% coverage):**

The test suite demonstrates four distinct escalation paths. Clean data with high confidence (≥ 0.8) produces `NONE` — no review needed. Clean data with moderate confidence (0.6–0.8) produces `SPOT_CHECK`. Clean data with low confidence (0.4–0.6) produces `DETAILED_REVIEW`. And here is where the OR condition matters: the presence of any outlier immediately escalates to `EXPERT_REQUIRED`, regardless of confidence. Even with perfect confidence of 1.0, a single statistical outlier overrides it. The test `test_outlier_overrides_perfect_confidence` verifies this explicitly — six identical values at 1.0 plus one at 100.0 triggers `EXPERT_REQUIRED` despite confidence being at its maximum.

The boundary tests are precise: confidence of exactly 0.4 routes to `DETAILED_REVIEW` (not `EXPERT_REQUIRED`), confidence of exactly 0.6 routes to `SPOT_CHECK`, and confidence of exactly 0.8 routes to `NONE`. These boundary behaviours are tested explicitly because they determine the routing decision for every output the system processes.

**The reason strings:** Every `OutlierFlag` carries a `reason` field with a complete English sentence: which value was flagged, whether it was above the upper fence or below the lower fence, what the fence value was, and what the consensus was. This is designed to be shown directly to a human reviewer without requiring them to understand the underlying statistics.

**Code ancestry:** This implementation is generalised from the `agent-task-planning` package in the [PDA Platform](https://github.com/antnewman/pda-platform), where it was originally used to detect disagreement across multiple AI samples extracting project schedule data. The IQR algorithm and four-tier escalation architecture are identical; the domain-specific terminology has been removed.

## Limitations

**IQR is a blunt instrument for heavy-tailed distributions.** Tukey's fence method assumes the underlying data is roughly symmetrically distributed. For heavy-tailed distributions (e.g. cost estimates, response times, financial returns), the 1.5×IQR multiplier may be too tight (flagging legitimate variation as outliers) or too loose (missing genuine anomalies in the tails). Practitioners in these domains may need to adjust the multiplier or use a different detection method entirely — the architecture (detect → flag → escalate) remains the same, but the detection algorithm may need to change.

**Small sample sizes limit detection reliability.** The implementation requires at least two values for IQR computation, but in practice Tukey's method needs at least six values with a tight cluster before it reliably detects outliers. With fewer values, a single outlier can inflate Q3 and widen the fence past itself, meaning the very thing you are trying to detect is absorbed into the fence calculation. This is documented in the test suite and is a fundamental limitation of IQR-based methods, not an implementation bug. If your system only runs three or four model samples, consider increasing the sample count or using a different detection method.

**Does not distinguish between ambiguous data and genuine distribution shift.** An outlier might occur because the input data is genuinely ambiguous (the model is uncertain and one sample happened to go a different direction) or because the underlying distribution has shifted (the model is seeing something it was not trained on). The escalation system treats both cases identically — it routes to a human. Distinguishing between the two requires additional context that the outlier detection layer does not have.

**Escalation tiers are fixed.** The confidence thresholds (0.4, 0.6, 0.8) and the four-tier structure are hardcoded. In practice, different domains have different risk tolerances — a financial trading system might want a tighter threshold than a content recommendation system. The current implementation does not support configurable thresholds. This is a deliberate simplicity choice for the reference implementation; production deployments should parameterise these values.

**The OR condition increases escalation volume.** Because outliers and low confidence independently trigger escalation, the system will escalate more outputs than a system using an AND condition (requiring both to be present). This is by design — it fails safe — but it means the human review queue will be larger. If reviewer capacity is constrained, this can create a bottleneck. The trade-off is explicit: missed problems are more expensive than unnecessary reviews.

**Requires multiple model samples.** The entire approach depends on running the same input through the model multiple times to generate values that can be compared. This multiplies inference cost linearly with sample count. A system running five samples per input is five times more expensive to operate than a single-pass system. This cost must be weighed against the value of detecting disagreement.

## Reader Takeaway

When multiple model outputs disagree, the disagreement is information — do not average it away. Use an OR condition for escalation routing: outliers OR low confidence, each independently, trigger human review. This means the system fails safe. The four-tier architecture (EXPERT_REQUIRED, DETAILED_REVIEW, SPOT_CHECK, NONE) gives you graduated response rather than a binary pass/fail, and the plain-English reason strings on each outlier flag mean a human reviewer can act on the escalation without understanding the underlying statistics. The model is disagreeing with itself for a reason. Route accordingly.

---

*Contributor note: Ant Newman wrote the primary draft. Implementation generalised from the PDA Platform agent-task-planning package by Ant Newman. Malia Hosseini implemented the original outlier mining module. Lawrence Rowland contributed the original requirements and conceptual design.*
