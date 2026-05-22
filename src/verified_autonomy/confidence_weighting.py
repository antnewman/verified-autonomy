"""
Layer 01: Inverse Confidence Weighting

Surface your weakest signal, not your average.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

See layers/01-confidence-weighting.md for the full explanation,
production examples, and limitations.
See implementations/layer_01_confidence_weighting/ for the runnable implementation.
"""

# Re-export from the standalone implementation package.
# To use this function, install the layer package:
#   cd implementations/layer_01_confidence_weighting && make install
# Or copy confidence_weighting.py into your project directly.

from __future__ import annotations


def inverse_confidence_weight(field_confidences: dict[str, float]) -> float:
    """Compute an aggregate confidence score using inverse weighting.

    Fields with lower confidence are weighted more heavily, forcing the
    aggregate to reflect the weakest signal rather than masking it in a
    plain average.

    The weight for each field is: ``weight = 2.0 - confidence``.

    Args:
        field_confidences: Mapping of field name to confidence score.
            Each confidence value must be in the range [0.0, 1.0].

    Returns:
        Weighted aggregate confidence in [0.0, 1.0].
        Returns 0.0 for an empty mapping.

    Raises:
        NotImplementedError: Install the standalone layer package.
            See implementations/layer_01_confidence_weighting/.
        ValueError: If any confidence value is outside [0.0, 1.0].
    """
    raise NotImplementedError(
        "Install the standalone layer package to use this function:\n"
        "  cd implementations/layer_01_confidence_weighting && make install\n"
        "See implementations/layer_01_confidence_weighting/ for the full implementation."
    )
