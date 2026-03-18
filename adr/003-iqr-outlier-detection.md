# ADR 003: IQR-Based Outlier Detection

## Status

Accepted

## Context

When multiple model samples disagree about a specific value, the disagreement contains information. A model that consistently produces different answers when asked the same question five times is signalling something — uncertainty, instability, or sensitivity to prompt variation. The question is how to detect this disagreement and what to do with it.

## Decision

Use Inter-Quartile Range (IQR) detection across samples. Treat detected outliers as hard escalation triggers using an OR condition: outlier OR low confidence triggers expert review. This means the system fails safe — either signal alone is sufficient to escalate.

Four-tier escalation routing:
- `EXPERT_REQUIRED` — outlier detected or confidence below minimum threshold
- `DETAILED_REVIEW` — confidence in caution band
- `SPOT_CHECK` — confidence acceptable, no outliers
- `NONE` — high confidence, no outliers

## Consequences

**Benefits:**
- Disagreement between samples becomes a routing signal rather than noise to be averaged away.
- The OR condition means the system fails safe: a case that is confident but inconsistent still escalates.
- Four-tier routing gives downstream systems actionable distinctions rather than a binary pass/fail.

**Trade-offs:**
- IQR is a blunt instrument for heavy-tailed distributions, where extreme values are expected rather than anomalous.
- Small sample size (five samples in the reference implementation) limits detection reliability. More samples improve detection but increase inference cost.

**Alternatives considered:**
- Z-score detection — sensitive to outliers in the reference distribution itself; less robust for small samples.
- Ignoring disagreement and averaging — rejected; disagreement is a signal, not noise. Averaging it away discards exactly the information that matters.
