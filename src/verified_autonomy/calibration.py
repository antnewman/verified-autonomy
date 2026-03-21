"""
Layer 04: Calibration and Conformal Prediction

The only confidence score with a mathematical guarantee.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

See layers/04-calibration.md for the full explanation,
production examples, and limitations.
See implementations/layer_04_calibration/ for the runnable implementation.
"""

# TODO: Implementation — see implementations/layer_04_calibration/

from typing import Any


def compute_ece(*args: Any, **kwargs: Any) -> Any:
    """Compute Expected Calibration Error using equal-width binning.

    Args:
        *args: Positional arguments — to be defined when implementation is added.
        **kwargs: Keyword arguments — to be defined when implementation is added.

    Returns:
        To be defined when implementation is added.

    Raises:
        NotImplementedError: This function is a placeholder pending implementation.
    """
    raise NotImplementedError(
        "Layer 04 is not yet implemented. "
        "See implementations/layer_04_calibration/ for the planned implementation."
    )


def calibrate_conformal(*args: Any, **kwargs: Any) -> Any:
    """Compute the conformal threshold from a calibration set.

    Args:
        *args: Positional arguments — to be defined when implementation is added.
        **kwargs: Keyword arguments — to be defined when implementation is added.

    Returns:
        To be defined when implementation is added.

    Raises:
        NotImplementedError: This function is a placeholder pending implementation.
    """
    raise NotImplementedError(
        "Layer 04 is not yet implemented. "
        "See implementations/layer_04_calibration/ for the planned implementation."
    )


def conformal_predict(*args: Any, **kwargs: Any) -> Any:
    """Produce prediction sets using a conformal threshold.

    Args:
        *args: Positional arguments — to be defined when implementation is added.
        **kwargs: Keyword arguments — to be defined when implementation is added.

    Returns:
        To be defined when implementation is added.

    Raises:
        NotImplementedError: This function is a placeholder pending implementation.
    """
    raise NotImplementedError(
        "Layer 04 is not yet implemented. "
        "See implementations/layer_04_calibration/ for the planned implementation."
    )
