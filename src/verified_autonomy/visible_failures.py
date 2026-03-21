"""
Layer 03: Making Failures Visible

The potential hallucinations list.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

See layers/03-visible-failures.md for the full explanation,
production examples, and limitations.
See implementations/layer_03_visible_failures/ for the runnable implementation.
"""

# Re-export from the standalone implementation package.
# To use these functions, install the layer package:
#   cd implementations/layer_03_visible_failures && make install
# Or copy visible_failures.py into your project directly.

from __future__ import annotations

from typing import Any


def compute_quality_score(
    coherence: float,
    relevance: float,
    completeness: float,
    semantic_entropy: float,
    threshold: float = 0.6,
) -> Any:
    """Compute a quality score for a single model-generated sample.

    Formula::

        entropy_penalty = max(0, semantic_entropy - 0.5) * 0.5
        overall = (coherence×0.4 + relevance×0.4 + completeness×0.2)
                  - entropy_penalty

    Args:
        coherence: Internal consistency of the output (0.0–1.0).
        relevance: On-topic quality of the output (0.0–1.0).
        completeness: How fully the request was addressed (0.0–1.0).
        semantic_entropy: Cross-sample variance proxy.  Values above 0.5
            incur a quality penalty.
        threshold: Minimum overall score to pass.  Defaults to 0.6.

    Returns:
        ``QualityScore`` dataclass.
        See implementations/layer_03_visible_failures/ for type definitions.

    Raises:
        NotImplementedError: Install the standalone layer package.
            See implementations/layer_03_visible_failures/.
    """
    raise NotImplementedError(
        "Install the standalone layer package to use this function:\n"
        "  cd implementations/layer_03_visible_failures && make install\n"
        "See implementations/layer_03_visible_failures/ for the full implementation."
    )


def flag_potential_hallucinations(
    samples: Any,
    entropy_threshold: float = 0.7,
) -> list[str]:
    """Return the IDs of samples at elevated hallucination risk.

    A sample is flagged when its ``semantic_entropy`` exceeds
    ``entropy_threshold``.  High entropy indicates the model produced
    markedly different outputs across repeated runs — a proxy for
    confabulation.

    This list is intended to be published alongside model outputs, not
    to silently suppress them.

    Args:
        samples: List of ``Sample`` objects.
            See implementations/layer_03_visible_failures/ for type definitions.
        entropy_threshold: Entropy above this value triggers a flag.
            Defaults to 0.7.

    Returns:
        List of ``Sample.id`` values for samples at hallucination risk.

    Raises:
        NotImplementedError: Install the standalone layer package.
            See implementations/layer_03_visible_failures/.
    """
    raise NotImplementedError(
        "Install the standalone layer package to use this function:\n"
        "  cd implementations/layer_03_visible_failures && make install\n"
        "See implementations/layer_03_visible_failures/ for the full implementation."
    )


def compute_pass_rate(samples: Any) -> float:
    """Compute the fraction of samples that passed the quality threshold.

    Args:
        samples: List of ``Sample`` objects.
            See implementations/layer_03_visible_failures/ for type definitions.

    Returns:
        Pass rate in [0.0, 1.0].  Returns 0.0 for an empty list.

    Raises:
        NotImplementedError: Install the standalone layer package.
            See implementations/layer_03_visible_failures/.
    """
    raise NotImplementedError(
        "Install the standalone layer package to use this function:\n"
        "  cd implementations/layer_03_visible_failures && make install\n"
        "See implementations/layer_03_visible_failures/ for the full implementation."
    )
