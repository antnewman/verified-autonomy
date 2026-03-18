"""
Layer 04: Calibration and Conformal Prediction

The only confidence score with a mathematical guarantee.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

This module demonstrates three things in sequence:

1. The problem  -- softmax overconfidence (ECE >> 0)
2. The partial fix -- temperature scaling (improves ECE, no formal guarantee)
3. The guaranteed fix -- split conformal prediction (finite-sample coverage guarantee)

The conformal prediction algorithm is implemented from scratch (no MAPIE or
other CP library) so the mechanism is fully transparent.  The classifier
and synthetic data use scikit-learn; temperature optimisation uses scipy.

See layers/04-calibration.md for the full explanation, production examples,
and limitations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinData:
    """Data for a single bin in a reliability diagram.

    Attributes:
        bin_midpoint: Centre of the confidence bin.
        accuracy: Fraction of predictions in this bin that were correct.
        confidence: Mean predicted confidence for samples in this bin.
        count: Number of samples in this bin.
    """

    bin_midpoint: float
    accuracy: float
    confidence: float
    count: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def _validate_lengths(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    if len(y_true) != len(y_prob):
        raise ValueError(
            f"y_true length ({len(y_true)}) must match y_prob length ({len(y_prob)})."
        )


# ---------------------------------------------------------------------------
# ECE and reliability diagram
# ---------------------------------------------------------------------------


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, list[BinData]]:
    """Compute Expected Calibration Error using equal-width binning.

    Returns the scalar ECE value and per-bin data for reliability diagrams.
    Uses the L1 formulation from Guo et al. 2017 with M equal-width bins.

    ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|

    where acc(B_m) is the fraction correct in the bin and conf(B_m) is the
    mean maximum predicted probability in the bin.

    Args:
        y_true: True integer class labels, shape (n,).
        y_prob: Predicted probabilities, shape (n, n_classes).
        n_bins: Number of equal-width confidence bins.  Default 15.

    Returns:
        Tuple of (ece_scalar, list_of_BinData).  Empty bins are omitted from
        the list and contribute 0 to ECE.

    Raises:
        ValueError: If y_true and y_prob have different lengths.
    """
    _validate_lengths(y_true, y_prob)

    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    n = len(y_true)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    bins: list[BinData] = []

    for i in range(n_bins):
        lower, upper = float(bin_edges[i]), float(bin_edges[i + 1])
        # Last bin includes the right boundary
        if i < n_bins - 1:
            mask = (confidences >= lower) & (confidences < upper)
        else:
            mask = (confidences >= lower) & (confidences <= upper)

        count = int(mask.sum())
        if count == 0:
            continue

        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        bin_midpoint = (lower + upper) / 2.0

        ece += (count / n) * abs(bin_acc - bin_conf)
        bins.append(
            BinData(
                bin_midpoint=bin_midpoint,
                accuracy=bin_acc,
                confidence=bin_conf,
                count=count,
            )
        )

    return ece, bins


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> list[BinData]:
    """Compute per-bin accuracy vs confidence data for a reliability diagram.

    Each BinData contains: bin_midpoint, accuracy, confidence, count.
    Perfect calibration means accuracy == confidence for every bin.

    Args:
        y_true: True integer class labels, shape (n,).
        y_prob: Predicted probabilities, shape (n, n_classes).
        n_bins: Number of equal-width confidence bins.  Default 15.

    Returns:
        List of BinData for non-empty bins only.
    """
    _, bins = compute_ece(y_true, y_prob, n_bins=n_bins)
    return bins


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def find_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Find the optimal temperature T for temperature scaling.

    Minimises negative log-likelihood on a validation set via
    ``scipy.optimize.minimize_scalar`` (bounded search in [0.1, 10.0]).
    Returns the optimal T > 0.  When T > 1 the softmax is softened
    (the model was overconfident).  When T < 1 it is sharpened.

    Based on Guo et al. 2017 "On Calibration of Modern Neural Networks."

    Note: when the input model produces probabilities rather than raw logits
    (e.g. RandomForestClassifier, GradientBoostingClassifier), convert to
    pseudo-logits via ``np.log(probs + epsilon)`` before calling this function.
    This preserves the ordinal structure needed for temperature scaling and
    is the standard approach when raw logits are unavailable.

    Args:
        logits: Pre-softmax activations (or pseudo-logits), shape (n, n_classes).
        y_true: True integer class labels, shape (n,).

    Returns:
        Optimal temperature T in (0.1, 10.0).
    """

    def nll(T: float) -> float:
        scaled = logits / T
        # Numerically stable log-softmax
        max_vals = scaled.max(axis=1, keepdims=True)
        shifted = scaled - max_vals
        log_sum_exp = np.log(np.exp(shifted).sum(axis=1))
        # Log-prob of the true class for each sample
        log_true = shifted[np.arange(len(y_true)), y_true] - log_sum_exp
        return float(-log_true.mean())

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def apply_temperature_scaling(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply temperature scaling to logits: softmax(logits / T).

    Returns calibrated probability array with the same shape as input.

    Args:
        logits: Pre-softmax activations, shape (n, n_classes) or (n_classes,).
        temperature: Scaling temperature T.  Must be > 0.

    Returns:
        Calibrated probabilities, same shape as logits.

    Raises:
        ValueError: If temperature <= 0.
    """
    if temperature <= 0.0:
        raise ValueError(
            f"temperature must be positive, got {temperature!r}."
        )
    return _softmax(logits / temperature)


# ---------------------------------------------------------------------------
# Split conformal prediction
# ---------------------------------------------------------------------------


def calibrate_conformal(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Compute the conformal threshold (q_hat) from a calibration set.

    Uses split conformal prediction.  Computes conformity scores as
    ``1 - f(x)_y`` (the probability the model assigns to the true class),
    then takes the ``ceil((n+1)(1-alpha))/n`` quantile of the scores.

    The resulting threshold carries a formal guarantee: if test data is
    exchangeable with the calibration set, prediction sets built using
    this threshold will contain the true class with probability >= 1 - alpha.

    Higher alpha => lower threshold => smaller prediction sets => lower coverage.
    Lower alpha  => higher threshold => larger prediction sets => higher coverage.

    Args:
        y_true: True labels for the calibration set, shape (n,).
        y_prob: Predicted probability matrix, shape (n, n_classes).
        alpha: Target miscoverage rate.  Default 0.1 (90% coverage).

    Returns:
        The conformal threshold q_hat in [0.0, 1.0].

    Raises:
        ValueError: If alpha is not in (0.0, 1.0).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f"alpha must be in (0.0, 1.0), got {alpha!r}."
        )

    n = len(y_true)
    # Conformity score for sample i: how unlikely the true class is
    scores = 1.0 - y_prob[np.arange(n), y_true]

    # Finite-sample quantile level (Angelopoulos & Bates 2022)
    q_level = math.ceil((n + 1) * (1.0 - alpha)) / n
    q_level = min(q_level, 1.0)  # Clamp for very small n

    return float(np.quantile(scores, q_level))


def conformal_predict(
    y_prob: np.ndarray,
    threshold: float,
) -> list[np.ndarray]:
    """Produce prediction sets using a conformal threshold.

    For each sample, includes all class indices whose conformity score
    (1 - predicted probability) is at most the threshold.  Equivalently,
    includes all classes the model assigns probability >= 1 - threshold.

    The coverage guarantee: if the threshold was computed by
    ``calibrate_conformal`` with alpha=0.1 on exchangeable data, these
    prediction sets contain the true class with probability >= 90%.

    Note on threshold direction (important):
        - Higher threshold => more classes included => larger sets
        - Lower threshold => fewer classes included => smaller sets
        - threshold=0.0 => only classes with predicted probability >= 1.0
          (near-empty in practice)
        - threshold=1.0 => all classes included (score <= 1 is always true)

    Args:
        y_prob: Predicted probability matrix, shape (n, n_classes).
        threshold: The conformal threshold from ``calibrate_conformal``.

    Returns:
        List of arrays, one per sample, each containing the class
        indices in the prediction set.
    """
    prediction_sets: list[np.ndarray] = []
    for probs in y_prob:
        scores = 1.0 - probs  # Conformity scores
        included = np.where(scores <= threshold)[0]
        prediction_sets.append(included)
    return prediction_sets


def evaluate_coverage(
    y_true: np.ndarray,
    prediction_sets: list[np.ndarray],
) -> float:
    """Compute empirical coverage: fraction of samples where the true label
    is contained in the prediction set.

    This should be >= 1 - alpha when the conformal threshold was correctly
    computed on exchangeable data.

    Args:
        y_true: True class labels, shape (n,).
        prediction_sets: List of prediction-set arrays, one per sample.

    Returns:
        Empirical coverage in [0.0, 1.0].  Returns 0.0 for empty input.
    """
    if len(y_true) == 0:
        return 0.0
    covered = sum(
        int(y_true[i]) in prediction_sets[i] for i in range(len(y_true))
    )
    return covered / len(y_true)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Walk through the full three-step calibration narrative.

    Step 1: The Problem    -- softmax overconfidence (ECE >> 0)
    Step 2: Partial Fix    -- temperature scaling (ECE reduced, no guarantee)
    Step 3: Guaranteed Fix -- split conformal prediction (formal coverage)
    """
    # Imports kept local so the module is importable without sklearn/scipy
    # installed (useful when only the algorithm code is needed).
    from sklearn.datasets import make_classification  # type: ignore[import-untyped]
    from sklearn.ensemble import GradientBoostingClassifier  # type: ignore[import-untyped]

    print("Layer 04: Calibration and Conformal Prediction")
    print("=" * 50)
    print()

    # --- Data: 3000 samples, 5 classes, 20 features ---
    # n_redundant=10 ensures the task is hard enough that the model's softmax
    # outputs reliably exceed its actual accuracy (ECE >> 0).
    X, y = make_classification(
        n_samples=3000,
        n_features=20,
        n_classes=5,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_train, y_train = X[:1000], y[:1000]
    X_cal, y_cal = X[1000:2000], y[1000:2000]
    X_test, y_test = X[2000:], y[2000:]

    # GradientBoostingClassifier with deep trees is reliably overconfident:
    # the ensemble's probability estimates consistently exceed actual accuracy,
    # yielding ECE >> 0 and an optimal temperature T > 1.
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    test_acc = float(clf.score(X_test, y_test))

    # -----------------------------------------------------------------------
    # Step 1 -- The Problem: Softmax Overconfidence
    # -----------------------------------------------------------------------
    print("Step 1 -- The Problem: Softmax Overconfidence")
    print("-" * 46)
    print("Training a GradientBoostingClassifier on synthetic 5-class data...")
    print(f"  Test accuracy: {test_acc:.1%}")
    print()

    probs_test = clf.predict_proba(X_test)
    ece_before, bins_before = compute_ece(y_test, probs_test)

    print(f"Uncalibrated ECE: {ece_before:.4f}")
    print("  The model's stated confidence does not match its actual accuracy.")
    print("  Reliability diagram (perfect calibration = diagonal):")
    for b in bins_before:
        print(
            f"    Bin {b.bin_midpoint:.2f}: accuracy={b.accuracy:.2f}, "
            f"confidence={b.confidence:.2f}, n={b.count}"
        )
    print()

    # -----------------------------------------------------------------------
    # Step 2 -- Partial Fix: Temperature Scaling
    # -----------------------------------------------------------------------
    print("Step 2 -- Partial Fix: Temperature Scaling")
    print("-" * 43)

    # GradientBoostingClassifier produces predict_proba, not raw logits.
    # Convert to pseudo-logits via log(p + ε) — the standard approach when
    # raw logits are unavailable.
    epsilon = 1e-7
    probs_cal = clf.predict_proba(X_cal)
    logits_cal = np.log(probs_cal + epsilon)
    logits_test = np.log(probs_test + epsilon)

    T = find_temperature(logits_cal, y_cal)
    probs_test_scaled = apply_temperature_scaling(logits_test, T)
    ece_after, _ = compute_ece(y_test, probs_test_scaled)

    print(f"Optimal temperature: T = {T:.2f}")
    print("  (T > 1 means the model was overconfident)")
    print()
    print(f"Calibrated ECE: {ece_after:.4f}")
    print(
        f"  ECE reduced from {ece_before:.4f} to {ece_after:.4f} -- but this is a point estimate"
    )
    print(
        "  with no formal guarantee. Under distribution shift, it can degrade."
    )
    print()

    # -----------------------------------------------------------------------
    # Step 3 -- The Guaranteed Fix: Split Conformal Prediction
    # -----------------------------------------------------------------------
    print("Step 3 -- The Guaranteed Fix: Split Conformal Prediction")
    print("-" * 56)

    alpha = 0.1
    q_hat = calibrate_conformal(y_cal, probs_cal, alpha=alpha)
    prediction_sets = conformal_predict(probs_test, q_hat)
    coverage = evaluate_coverage(y_test, prediction_sets)
    avg_set_size = float(np.mean([len(ps) for ps in prediction_sets]))
    guarantee_met = coverage >= (1.0 - alpha)

    print(f"Target coverage: {(1-alpha):.0%} (alpha = {alpha:.2f})")
    print(f"Conformal threshold (q_hat): {q_hat:.4f}")
    print()
    print(f"Empirical coverage on test set: {coverage:.1%}")
    print(f"  Coverage >= {(1-alpha):.0%} target: {'YES' if guarantee_met else 'NO'}")
    print("  This is a finite-sample guarantee, not an estimate.")
    print()
    print(f"Average prediction set size: {avg_set_size:.1f} classes")
    print("  (Smaller sets = more precise; 1.0 = single class predicted)")
    print()

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    print("Comparison:")
    print(f"  {'Method':<24}  {'ECE':>7}  {'Coverage':>8}  {'Guarantee?':>10}")
    print(f"  {'-'*24}  {'-'*7}  {'-'*8}  {'-'*10}")
    print(f"  {'Uncalibrated softmax':<24}  {ece_before:>7.4f}  {'N/A':>8}  {'No':>10}")
    print(f"  {'Temperature scaling':<24}  {ece_after:>7.4f}  {'N/A':>8}  {'No':>10}")
    print(
        f"  {'Conformal prediction':<24}  {'N/A':>7}  {coverage:>8.1%}  {'Yes (>=90%)':>10}"
    )
    print()
    print("See: layers/04-calibration.md")


if __name__ == "__main__":  # pragma: no cover
    main()
