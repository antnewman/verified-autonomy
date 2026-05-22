"""
Layer 08: Cryptographic Audit Trails

Proving what the system did.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

See layers/08-cryptographic-audit.md for the full explanation,
production examples, and limitations.
See implementations/layer_08_cryptographic_audit/ for the runnable implementation.
"""

# TODO: Implementation — see implementations/layer_08_cryptographic_audit/

from typing import Any


def record_decision(*args: Any, **kwargs: Any) -> Any:
    """Record an AI system decision as a tamper-evident audit entry.

    Args:
        *args: Positional arguments — to be defined when implementation is added.
        **kwargs: Keyword arguments — to be defined when implementation is added.

    Returns:
        To be defined when implementation is added.

    Raises:
        NotImplementedError: This function is a placeholder pending implementation.
    """
    raise NotImplementedError(
        "Layer 08 is not yet implemented. "
        "See implementations/layer_08_cryptographic_audit/ for the planned implementation."
    )


def verify_chain(*args: Any, **kwargs: Any) -> Any:
    """Verify the integrity of a cryptographic audit chain.

    Args:
        *args: Positional arguments — to be defined when implementation is added.
        **kwargs: Keyword arguments — to be defined when implementation is added.

    Returns:
        To be defined when implementation is added.

    Raises:
        NotImplementedError: This function is a placeholder pending implementation.
    """
    raise NotImplementedError(
        "Layer 08 is not yet implemented. "
        "See implementations/layer_08_cryptographic_audit/ for the planned implementation."
    )
