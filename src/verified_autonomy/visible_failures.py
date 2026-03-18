"""
Layer 03: Making Failures Visible

The potential hallucinations list.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

See layers/03-visible-failures.md for the full explanation,
production examples, and limitations.
See implementations/layer_03_visible_failures/ for the runnable implementation.
"""

# TODO: Implementation — see implementations/layer_03_visible_failures/

from typing import Any


def flag_potential_hallucinations(*args: Any, **kwargs: Any) -> Any:
    """Flag model outputs that fall below the quality threshold as potential hallucinations.

    Args:
        *args: Positional arguments — to be defined when implementation is added.
        **kwargs: Keyword arguments — to be defined when implementation is added.

    Returns:
        To be defined when implementation is added.

    Raises:
        NotImplementedError: This function is a placeholder pending implementation.
    """
    raise NotImplementedError(
        "Layer 03 is not yet implemented. "
        "See implementations/layer_03_visible_failures/ for the planned implementation."
    )
