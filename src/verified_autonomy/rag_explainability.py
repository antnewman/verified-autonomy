"""
Layer 06: Retrieval-Augmented Generation as Explainability

Shift trust from model intent to source facticity.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

See layers/06-rag-explainability.md for the full explanation,
production examples, and limitations.
See implementations/layer_06_rag_explainability/ for the runnable implementation.
"""

from typing import Any


def check_groundedness(*args: Any, **kwargs: Any) -> Any:
    """Evaluate how well an answer is grounded in retrieved source documents.

    Uses token overlap between the answer and source document contents,
    with stop-word removal for signal quality.

    Full implementation: implementations/layer_06_rag_explainability/src/rag_explainability.py
    """
    raise NotImplementedError(
        "Layer 06 stub only. "
        "See implementations/layer_06_rag_explainability/ for the full implementation."
    )


def rag_query(*args: Any, **kwargs: Any) -> Any:
    """Run the full RAG explainability pipeline: retrieve, assess, report.

    Retrieves the most relevant documents for a query, evaluates how well
    the supplied answer is grounded in those documents, and produces a
    human-readable provenance trail.

    Full implementation: implementations/layer_06_rag_explainability/src/rag_explainability.py
    """
    raise NotImplementedError(
        "Layer 06 stub only. "
        "See implementations/layer_06_rag_explainability/ for the full implementation."
    )
