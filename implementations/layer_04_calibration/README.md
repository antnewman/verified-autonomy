# Layer 04: Calibration and Conformal Prediction — Implementation

## Quick Start

```bash
make install
make run
```

## Expected Output

```
Layer 04: Calibration and Conformal Prediction
==================================================

Step 1 -- The Problem: Softmax Overconfidence
----------------------------------------------
Training a GradientBoostingClassifier on synthetic 5-class data...
  Test accuracy: 88.0%

Uncalibrated ECE: 0.0773
  The model's stated confidence does not match its actual accuracy.
  Reliability diagram (perfect calibration = diagonal):
    Bin 0.37: accuracy=0.33, confidence=0.38, n=3
    Bin 0.43: accuracy=0.50, confidence=0.42, n=2
    Bin 0.50: accuracy=0.20, confidence=0.51, n=10
    Bin 0.57: accuracy=0.59, confidence=0.56, n=17
    Bin 0.63: accuracy=0.50, confidence=0.64, n=26
    Bin 0.70: accuracy=0.33, confidence=0.71, n=18
    Bin 0.77: accuracy=0.71, confidence=0.77, n=17
    Bin 0.83: accuracy=0.63, confidence=0.84, n=19
    Bin 0.90: accuracy=0.71, confidence=0.91, n=34
    Bin 0.97: accuracy=0.94, confidence=1.00, n=854

Step 2 -- Partial Fix: Temperature Scaling
-------------------------------------------
Optimal temperature: T = 2.27
  (T > 1 means the model was overconfident)

Calibrated ECE: 0.0188
  ECE reduced from 0.0773 to 0.0188 -- but this is a point estimate
  with no formal guarantee. Under distribution shift, it can degrade.

Step 3 -- The Guaranteed Fix: Split Conformal Prediction
--------------------------------------------------------
Target coverage: 90% (alpha = 0.10)
Conformal threshold (q_hat): 0.8298

Empirical coverage on test set: 91.6%
  Coverage >= 90% target: YES
  This is a finite-sample guarantee, not an estimate.

Average prediction set size: 1.1 classes
  (Smaller sets = more precise; 1.0 = single class predicted)

Comparison:
  Method                        ECE  Coverage  Guarantee?
  ------------------------  -------  --------  ----------
  Uncalibrated softmax       0.0773       N/A          No
  Temperature scaling        0.0188       N/A          No
  Conformal prediction          N/A     91.6%  Yes (>=90%)

See: layers/04-calibration.md
```

## What This Code Does

Demonstrates a three-step narrative arc in calibration.  First, a
`GradientBoostingClassifier` with deep trees is trained on a five-class
synthetic dataset: its softmax probabilities reliably exceed its actual
accuracy (ECE = 0.077).  Second, temperature scaling (Guo et al. 2017) finds
the optimal temperature T = 2.27, which meaningfully reduces ECE to 0.019 —
but this is a point estimate with no formal guarantee.  Third, split conformal
prediction (Venn & Gammerman; Angelopoulos & Bates 2022) computes a threshold
from a held-out calibration set and produces prediction sets that are
guaranteed, by construction, to contain the true class with probability ≥ 90%.
Empirical coverage is 91.6%.  The conformal prediction algorithm is implemented
from scratch — no MAPIE or other CP library — so the mechanism is fully
transparent.

## API

| Function | Signature | Description |
|---|---|---|
| `compute_ece` | `(y_true, y_prob, n_bins=15) → tuple[float, list[BinData]]` | Expected Calibration Error (Guo et al. 2017 L1 formulation) |
| `reliability_diagram_data` | `(y_true, y_prob, n_bins=15) → list[BinData]` | Per-bin accuracy vs confidence for a reliability diagram |
| `find_temperature` | `(logits, y_true) → float` | Optimal temperature T via NLL minimisation over a calibration set |
| `apply_temperature_scaling` | `(logits, temperature) → np.ndarray` | Apply `softmax(logits / T)` |
| `calibrate_conformal` | `(y_true, y_prob, alpha=0.1) → float` | Conformal threshold q̂ with formal coverage guarantee ≥ 1 − α |
| `conformal_predict` | `(y_prob, threshold) → list[np.ndarray]` | Prediction sets: include class c if 1 − prob[c] ≤ threshold |
| `evaluate_coverage` | `(y_true, prediction_sets) → float` | Fraction of test samples where true label is in prediction set |

## Testing

```bash
make test
```

Tests cover: ECE correctness (L1 formula, bin edges, empty bins), temperature
scaling (NLL minimisation, T > 1 for overconfident models, ValueError on T ≤ 0),
conformal prediction (threshold direction, guarantee tolerance, edge thresholds),
coverage evaluation, input validation, and a full integration test that verifies
empirical coverage ≥ 1 − α − 2σ on held-out data.

## Dependencies

| Package | Version | Why |
|---|---|---|
| `numpy` | ≥ 1.26 | Array arithmetic throughout; `np.quantile` for conformal threshold |
| `scikit-learn` | ≥ 1.4 | `GradientBoostingClassifier`, `make_classification` in demo and integration tests |
| `scipy` | ≥ 1.12 | `minimize_scalar` for temperature optimisation (bounded 1-D search) |

Layers 01–03 carry zero runtime dependencies.  Layer 04 is the first to
introduce external libraries, justified because: (a) numpy is a transitive
dependency of scikit-learn anyway; (b) scipy is the standard tool for
numerical optimisation with no lightweight alternative; (c) all three packages
are stable, widely audited, and available on all target platforms.  The
conformal prediction core is implemented from scratch so no additional CP
library is required.
