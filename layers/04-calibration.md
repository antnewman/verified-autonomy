# Layer 04: Calibration and Conformal Prediction

*The only confidence score with a mathematical guarantee*

> **Primary author:** Ant Newman
>
> **Production examples:** Reference implementation with verified citations

---

## What It Is

When an AI system reports a confidence score, that score should mean something specific. If a model says "90% confident," it should be correct 90% of the time when it makes that claim. This property is called calibration. Most AI systems are not calibrated. Their confidence scores are systematically too high — a phenomenon called overconfidence.

This layer covers three techniques, in order of increasing rigour.

**Expected Calibration Error (ECE)** is the standard metric for measuring how miscalibrated a model is. It divides predictions into bins by confidence level and measures the gap between stated confidence and actual accuracy in each bin. An ECE of zero means the model is perfectly calibrated. In practice, modern neural networks and gradient-boosted models produce ECE values well above zero — they claim to be more confident than their accuracy justifies.

**Temperature scaling** is the simplest fix. It takes the model's raw outputs (logits) and divides them by a single learned parameter T before applying the softmax function. When T > 1, the output probabilities are softened — made less extreme. A model with T = 2.27 was roughly twice as overconfident as it should have been. Temperature scaling typically reduces ECE significantly, but it offers no formal guarantee. It is a point estimate that can degrade under distribution shift — when the test data differs from the training data.

**Conformal prediction** is fundamentally different. Instead of adjusting the model's confidence scores, it produces prediction sets — sets of possible answers — with a mathematical guarantee: the true answer is contained in the set with probability at least 1 − α, where α is a user-chosen miscoverage rate. If you set α = 0.1, the prediction sets will contain the true answer at least 90% of the time. This is not an estimate. It is a finite-sample guarantee that holds regardless of the model's architecture, how it was trained, or how complex it is. The only requirement is that the calibration data and test data are exchangeable — roughly, that they come from the same distribution with no temporal ordering effects.

The guarantee comes from a simple mechanism. Split conformal prediction holds out a calibration set, computes a conformity score for each calibration example (how unlikely was the true class under the model's predictions?), takes an appropriate quantile of those scores as a threshold, and then at test time includes any class whose score falls below the threshold. The mathematics guarantees that this procedure achieves the desired coverage in finite samples.

## Why It Matters

In 2025, researchers at the Georgia Institute of Technology evaluated NREL's publicly released PERFORM dataset of probabilistic solar energy forecasts across 1,149 solar sites in three US Independent System Operators. The forecasting models were designed to produce 90% prediction intervals — the actual energy output should fall within the interval 90% of the time.

The evaluation found that the intervals only captured the actual outcome between 47.9% and 83% of the time, depending on whether the forecast was for individual sites (worst case: 47.9% coverage in ERCOT) or fleet-aggregated system-level predictions (best case: 83% coverage in MISO). A companion study corroborated these findings, reporting only 75% empirical coverage for ERCOT system-level forecasts at the 90% target.

These are not obscure academic findings. Energy grid management, storage allocation, and pricing decisions were being made on the basis of confidence intervals that were missing reality between 17% and 52% of the time. The models were stating 90% confidence. The reality was as low as 48%.

This miscalibration pattern is not unique to energy forecasting. Guo, Pleiss, Sun, and Weinberger demonstrated in their 2017 ICML paper "On Calibration of Modern Neural Networks" that modern deep neural networks are severely overconfident. A 110-layer ResNet on CIFAR-100 produced ECE of 12.75% — its stated confidences bore little relationship to its actual accuracy. The problem worsened with increased depth, increased width, batch normalisation, and reduced weight decay. The very architectural choices that improve accuracy make calibration worse.

The failure mode is not that the model is inaccurate. It is that **the model's stated confidence has no reliable relationship to its actual probability of being correct.** Every downstream decision that depends on a confidence score — whether to auto-process or escalate, whether to discharge a patient, how to size a trading position — is only as good as the calibration of that score.

## Production Example

The implementation is open source in the Verified Autonomy repository at [`implementations/layer_04_calibration/`](https://github.com/antnewman/verified-autonomy/tree/main/implementations/layer_04_calibration).

**Run it yourself in under two minutes:**

```bash
cd implementations/layer_04_calibration
make install
make run
```

**The three-step demonstration:**

The implementation trains a `GradientBoostingClassifier` with deep trees on a synthetic five-class dataset (3,000 samples split into training, calibration, and test sets of 1,000 each). It then walks through the full narrative arc:

**Step 1 — The Problem.** The uncalibrated model achieves 88% test accuracy but produces an ECE of 0.077. Its reliability diagram reveals the miscalibration pattern: in the highest-confidence bin (97%), the model predicts with near-certainty (confidence 1.00) but is only correct 94% of the time. Across the lower bins, the gap between stated confidence and actual accuracy is inconsistent and often large.

**Step 2 — The Partial Fix.** Temperature scaling finds an optimal T = 2.27 — the model was substantially overconfident. After scaling, ECE drops from 0.077 to 0.019. This is a meaningful improvement, but it is a point estimate with no formal guarantee. Under distribution shift, it can degrade.

**Step 3 — The Guaranteed Fix.** Split conformal prediction computes a threshold from the calibration set with α = 0.1 (targeting 90% coverage). On the held-out test set, empirical coverage is 91.6% — exceeding the 90% target. This is not a lucky result on this particular test set. It is a consequence of the mathematical guarantee that holds for any exchangeable split.

```python
from verified_autonomy.calibration import (
    compute_ece,
    find_temperature,
    apply_temperature_scaling,
    calibrate_conformal,
    conformal_predict,
    evaluate_coverage,
)

# Measure miscalibration
ece, bins = compute_ece(y_test, probs_test)               # 0.077

# Temperature scaling (partial fix)
T = find_temperature(logits_cal, y_cal)                    # 2.27
probs_scaled = apply_temperature_scaling(logits_test, T)
ece_after, _ = compute_ece(y_test, probs_scaled)           # 0.019

# Conformal prediction (guaranteed fix)
q_hat = calibrate_conformal(y_cal, probs_cal, alpha=0.1)   # threshold
sets = conformal_predict(probs_test, q_hat)                 # prediction sets
coverage = evaluate_coverage(y_test, sets)                  # 0.916 (≥ 90%)
```

**The comparison table from `make run`:**

```
  Method                        ECE  Coverage  Guarantee?
  ------------------------  -------  --------  ----------
  Uncalibrated softmax       0.0773       N/A          No
  Temperature scaling        0.0188       N/A          No
  Conformal prediction          N/A     91.6%  Yes (>=90%)
```

**The integration test (65 tests, 100% coverage):** The most important test in the suite — `test_empirical_coverage_meets_guarantee_alpha_01` — trains a fresh model, computes the conformal threshold on a held-out calibration set, produces prediction sets on a separate test set, and verifies that empirical coverage meets the 90% target within a statistically rigorous 2-sigma tolerance. A second integration test verifies the guarantee at α = 0.5 (50% coverage). A third verifies that temperature scaling reduces ECE on held-out data. These are not unit tests of individual functions — they verify the end-to-end guarantees that the techniques claim to provide.

**Production deployments of conformal prediction:** AstraZeneca uses conformal prediction in their Predictive Insight Platform for compound prioritisation in drug discovery. ING Bank built a conformal intent classification system for customer service chatbots, guaranteeing that clarification questions contain the true user intent at a pre-defined confidence level. NatWest Group developed Longitudinal Predictive Conformal Inference for financial time series forecasting. EDF (Électricité de France) applied conformal prediction to gas demand forecasting. In healthcare, the Shahbazi et al. 2026 study in Nature Scientific Reports demonstrated a Bayesian-conformal hybrid for hospital length-of-stay prediction across 61,538 admissions in 3,793 US hospitals, achieving 94.3% coverage against a 95% target with 21% narrower intervals for low-uncertainty cases.

**The leading Python library for production conformal prediction is MAPIE** (scikit-learn-contrib), backed by Capgemini Invent and Quantmetry, with broad method coverage across regression, classification, and time series. The Verified Autonomy implementation is built from scratch for transparency — in production, practitioners should use MAPIE or a similar maintained library.

## Limitations

**The coverage guarantee is marginal, not conditional.** Conformal prediction guarantees that the true label is in the prediction set with probability ≥ 1 − α averaged over all possible test points. It does not guarantee coverage for any specific input. A system might have 95% coverage overall while systematically undercovering a particular subgroup. Barber et al. (2021) and Lei & Wasserman (2014) proved that exact conditional coverage is impossible without additional assumptions in a distribution-free setting. For high-stakes applications where specific-subgroup coverage matters, additional monitoring is needed.

**Exchangeability is required and can be violated.** The guarantee assumes the calibration and test data are exchangeable — essentially, that the order carries no information and they come from the same distribution. This breaks with time series data (temporal autocorrelation), distribution shift (the deployment environment differs from the calibration environment), concept drift, and feedback loops. When exchangeability is violated, conformal prediction can undercover (prediction sets miss the true value more often than promised) or overcover with uselessly wide intervals. Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021) addresses this for online settings by maintaining a time-varying miscoverage level that achieves target long-run coverage without any distributional assumptions — including exchangeability — but is not implemented in this reference module.

**Temperature scaling fails under distribution shift.** Ovadia et al. (NeurIPS 2019) ran a large-scale benchmark showing that post-hoc calibration methods including temperature scaling degrade significantly when test data differs from training data. Deep ensembles proved most robust under shift. Temperature scaling is a useful first step when the deployment environment matches the calibration environment. It is not sufficient when that assumption does not hold.

**ECE is a flawed metric.** The Expected Calibration Error is the standard calibration metric, but it has well-documented problems: results change significantly with different bin counts, an ECE estimate of 12% at small sample sizes could correspond to true error of 5% or 8% due to estimation bias, and it only considers the top predicted probability, ignoring full multiclass miscalibration. Roelofs et al. (AISTATS 2022) showed that ECE with 15 equal-width bins selected the correct recalibration method only 3 out of 10 times. The implementation uses ECE because it is the established standard (and readers will encounter it everywhere), but practitioners should supplement it with reliability diagrams and proper scoring rules (Brier score, log-loss).

**Prediction sets can be uninformatively wide.** If the underlying model is poor, conformal prediction will achieve the coverage guarantee by producing large prediction sets — sets that contain many classes. The guarantee is met, but the output is not useful. Average prediction set size is the diagnostic metric: a set size of 1.0 means the model is precise; a set size approaching the total number of classes means the model is uncertain about everything and conformal prediction is merely reflecting that honestly. The technique cannot create precision that the model does not have.

**Requires a held-out calibration set.** Split conformal prediction consumes data — the calibration set cannot be used for training. In data-scarce settings, this is a real cost. Full conformal prediction avoids this by retraining the model for every candidate value, but this is computationally prohibitive for complex models. The calibration set must be large enough for the quantile estimate to be reliable — in practice, at least several hundred samples.

## Reader Takeaway

Do not trust raw confidence scores from any model. Measure the gap between stated confidence and actual accuracy using ECE and reliability diagrams. Apply temperature scaling as a quick improvement, but understand that it offers no formal guarantee and degrades under distribution shift. For applications where confidence must be reliable, use conformal prediction: it provides a finite-sample coverage guarantee that holds regardless of model architecture, requires only a held-out calibration set and the exchangeability assumption, and is simple to implement. The distinction between "this model says 90% confident" and "this prediction set contains the true answer at least 90% of the time, guaranteed" is the difference between a number that sounds precise and a number that is.

---

*Contributor note: Ant Newman wrote the primary draft and implementation. The conformal prediction algorithm is implemented from scratch for transparency. Citations verified against Moradi et al. (2025) for NREL coverage figures, Guo et al. (2017) for temperature scaling, Shahbazi et al. (2026) for the Bayesian-conformal hybrid, and Angelopoulos & Bates (2023) for conformal prediction foundations.*
