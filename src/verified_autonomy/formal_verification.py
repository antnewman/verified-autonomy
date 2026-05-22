"""
Layer 09: Formal Verification

The gold standard nobody can fully use yet.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

See layers/09-formal-verification.md for the full explanation,
production examples, and limitations.
See implementations/layer_09_formal_verification/ for the runnable implementation.
"""

from typing import Any


def verify_output_bounds(*args: Any, **kwargs: Any) -> Any:
    """Formally verify that network outputs stay within bounds for all inputs.

    Uses Z3 SMT solver. Full implementation in
    implementations/layer_09_formal_verification/src/formal_verification.py
    """
    raise NotImplementedError(
        "Layer 09 stub only. "
        "See implementations/layer_09_formal_verification/ for the full implementation."
    )


def verify_monotonicity(*args: Any, **kwargs: Any) -> Any:
    """Formally verify monotonicity in one input dimension.

    Full implementation in
    implementations/layer_09_formal_verification/src/formal_verification.py
    """
    raise NotImplementedError(
        "Layer 09 stub only. "
        "See implementations/layer_09_formal_verification/ for the full implementation."
    )
