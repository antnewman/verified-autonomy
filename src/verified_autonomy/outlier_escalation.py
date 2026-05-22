"""
Layer 02: Outlier Detection as Hard Escalation

Statistical disagreement is a signal, not noise.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

See layers/02-outlier-escalation.md for the full explanation,
production examples, and limitations.
See implementations/layer_02_outlier_escalation/ for the runnable implementation.
"""

# Re-export from the standalone implementation package.
# To use this function, install the layer package:
#   cd implementations/layer_02_outlier_escalation && make install
# Or copy outlier_escalation.py into your project directly.

from __future__ import annotations

from typing import Any


def detect_outliers(
    values: list[float],
    labels: list[str] | None = None,
    confidence: float = 1.0,
) -> Any:
    """Detect statistical outliers and route to an appropriate escalation tier.

    Uses Tukey's IQR fence: values outside ``[Q1 - 1.5×IQR, Q3 + 1.5×IQR]``
    are flagged.  The escalation tier is determined by an OR condition —
    outliers *or* low confidence each independently trigger escalation.

    Escalation tiers:
    - ``EXPERT_REQUIRED``: outliers detected OR confidence < 0.4
    - ``DETAILED_REVIEW``: confidence in [0.4, 0.6)
    - ``SPOT_CHECK``: confidence in [0.6, 0.8)
    - ``NONE``: confidence >= 0.8 and no outliers

    Args:
        values: Numeric values to analyse.
        labels: Optional human-readable labels, one per value.
        confidence: Overall confidence score in [0.0, 1.0].

    Returns:
        ``DetectionResult`` with escalation level, consensus, and outliers.
        See implementations/layer_02_outlier_escalation/ for type definitions.

    Raises:
        NotImplementedError: Install the standalone layer package.
            See implementations/layer_02_outlier_escalation/.
        ValueError: If ``confidence`` is outside [0.0, 1.0] or ``labels``
            length does not match ``values`` length.
    """
    raise NotImplementedError(
        "Install the standalone layer package to use this function:\n"
        "  cd implementations/layer_02_outlier_escalation && make install\n"
        "See implementations/layer_02_outlier_escalation/ for the full implementation."
    )
