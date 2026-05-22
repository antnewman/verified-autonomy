# ADR 004: Conformal Prediction Over Pure Bayesian Approaches

## Status

Accepted

## Context

Standard softmax probabilities are systematically overconfident. The NREL case study is the canonical example: a model designed to produce 90% confidence intervals only covered 48–83% of actual outcomes in practice. The model said it was 90% confident; it was right far less often than that.

Bayesian approaches improve on raw softmax but introduce their own failure modes: they require prior specification, are computationally expensive at scale, and their theoretical guarantees depend on the prior being correct — which it often is not.

## Decision

Recommend conformal prediction as the primary uncertainty quantification framework, with Bayesian approaches as complements where appropriate. Conformal prediction is distribution-free and provides a formal coverage guarantee: if you ask for 90% coverage, you get at least 90% coverage on exchangeable test data. This guarantee holds regardless of the underlying model architecture.

## Consequences

**Benefits:**
- Mathematical guarantee on coverage — the only framework that provides one without strong distributional assumptions.
- Distribution-free: no assumptions about the underlying model or data distribution.
- Model-agnostic: works as a post-hoc wrapper around any existing model.

**Trade-offs:**
- Requires a calibration dataset that is exchangeable with the test data. Distribution shift between calibration and deployment breaks the coverage guarantee.
- Can produce wide, uninformative prediction intervals if the underlying model is poor. Conformal prediction makes an honest model more trustworthy; it cannot rescue a dishonest one.
- Computationally more expensive than reading off softmax probabilities directly.

**Alternatives considered:**
- Pure Bayesian approaches — valid complement but lack the formal frequentist coverage guarantee. Appropriate where prior knowledge is strong and the prior can be specified credibly.
- Softmax probabilities — rejected as primary uncertainty measure; systematically overconfident as demonstrated by NREL and replicated across domains.
