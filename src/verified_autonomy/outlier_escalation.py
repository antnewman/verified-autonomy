"""
Layer 02: Outlier Detection as Hard Escalation

Statistical disagreement is a signal, not noise.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

See layers/02-outlier-escalation.md for the full explanation,
production examples, and limitations.
See implementations/layer_02_outlier_escalation/ for the runnable implementation.
"""

# TODO: Implementation — see implementations/layer_02_outlier_escalation/

from typing import Any


def detect_outliers(*args: Any, **kwargs: Any) -> Any:
    """Detect statistical outliers across model samples using IQR-based detection.

    Args:
        *args: Positional arguments — to be defined when implementation is added.
        **kwargs: Keyword arguments — to be defined when implementation is added.

    Returns:
        To be defined when implementation is added.

    Raises:
        NotImplementedError: This function is a placeholder pending implementation.
    """
    raise NotImplementedError(
        "Layer 02 is not yet implemented. "
        "See implementations/layer_02_outlier_escalation/ for the planned implementation."
    )
