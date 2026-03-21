"""
Layer 06: Retrieval-Augmented Generation as Explainability

Shift trust from model intent to source facticity.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

This module implements a structured RAG-based explainability harness for AI
systems.  It demonstrates how to ground model outputs in verifiable source
documents, score each response for groundedness, surface per-document
citations, and produce a human-readable provenance trail.

The retrieval engine uses TF-IDF (scikit-learn) with cosine similarity.
Groundedness scoring uses token overlap between the answer and retrieved
source documents, with stop-word removal for signal quality.

Architecture
------------
- ``Document``:          A single source document in the knowledge base.
- ``RetrievalResult``:   A document paired with its relevance score.
- ``Citation``:          One retrieved document's contribution to an answer.
- ``GroundednessResult``: Token-overlap groundedness assessment for an answer.
- ``RAGResult``:         Query, answer, retrieved docs, groundedness, provenance.
- ``KnowledgeBase``:     Corpus of documents with TF-IDF retrieval.
- ``check_groundedness``: Evaluate an answer against a set of source documents.
- ``rag_query``:         Full RAG pipeline — retrieve, check, report.

See layers/06-rag-explainability.md for the full explanation, production
examples, and limitations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
        "so", "yet", "both", "either", "neither", "each", "every", "all",
        "any", "few", "more", "most", "other", "some", "such", "than", "too",
        "very", "just", "about", "above", "after", "also", "as", "because",
        "before", "between", "during", "if", "into", "it", "its", "itself",
        "that", "their", "them", "then", "there", "these", "they", "this",
        "those", "through", "under", "up", "us", "we", "what", "when",
        "where", "which", "while", "who", "whom", "whose", "why", "how",
    }
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Document:
    """A single source document in the knowledge base.

    Attributes:
        id:      Unique identifier (e.g. ``"doc-001"``).
        title:   Human-readable title.
        content: Full text of the document.
        source:  Origin reference (e.g. URL, filename, regulation number).
    """

    id: str
    title: str
    content: str
    source: str


@dataclass(frozen=True)
class RetrievalResult:
    """A document retrieved from the knowledge base with its relevance score.

    Attributes:
        document:        The retrieved document.
        relevance_score: Cosine similarity score (0.0–1.0).
    """

    document: Document
    relevance_score: float


@dataclass(frozen=True)
class Citation:
    """A single source document's contribution to an answer.

    Attributes:
        document_id:       ID of the cited document.
        document_title:    Title of the cited document.
        source:            Origin reference for the cited document.
        relevance_score:   Cosine similarity of the document to the query.
        groundedness_score: Token-overlap fraction of answer terms in this doc.
    """

    document_id: str
    document_title: str
    source: str
    relevance_score: float
    groundedness_score: float


@dataclass(frozen=True)
class GroundednessResult:
    """Token-overlap groundedness assessment for a model answer.

    Attributes:
        overall_score:    Fraction of answer tokens found in any source (0–1).
        grounded:         True if ``overall_score >= threshold``.
        citations:        Per-document citation with individual groundedness.
        ungrounded_terms: Answer tokens absent from all source documents.
        explanation:      Human-readable summary of the assessment.
    """

    overall_score: float
    grounded: bool
    citations: list[Citation]
    ungrounded_terms: list[str]
    explanation: str


@dataclass
class RAGResult:
    """Full result of a RAG query.

    Attributes:
        query:               The original query string.
        answer:              The answer being assessed.
        retrieved_documents: Documents returned from the knowledge base.
        groundedness:        Token-overlap groundedness assessment.
        provenance_trail:    Human-readable audit chain.
    """

    query: str
    answer: str
    retrieved_documents: list[RetrievalResult]
    groundedness: GroundednessResult
    provenance_trail: str


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """A corpus of documents with TF-IDF retrieval.

    Documents are added incrementally.  The TF-IDF index is rebuilt lazily
    on the first ``retrieve`` call after any document is added.

    Args:
        documents: Optional initial list of documents.
    """

    def __init__(self, documents: list[Document] | None = None) -> None:
        self._documents: list[Document] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: Any = None
        self._dirty: bool = True

        if documents:
            self.add_documents(documents)

    def add_document(self, document: Document) -> None:
        """Add a single document to the corpus."""
        self._documents.append(document)
        self._dirty = True

    def add_documents(self, documents: list[Document]) -> None:
        """Add multiple documents to the corpus."""
        self._documents.extend(documents)
        self._dirty = True

    @property
    def size(self) -> int:
        """Number of documents in the corpus."""
        return len(self._documents)

    def _rebuild_index(self) -> None:
        """Rebuild the TF-IDF index from all current documents."""
        if not self._documents:
            self._vectorizer = None
            self._matrix = None
            self._dirty = False
            return
        self._vectorizer = TfidfVectorizer()
        self._matrix = self._vectorizer.fit_transform(
            [doc.content for doc in self._documents]
        )
        self._dirty = False

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_relevance: float = 0.0,
    ) -> list[RetrievalResult]:
        """Retrieve the most relevant documents for a query.

        Args:
            query:         The search query.
            top_k:         Maximum number of results to return.
            min_relevance: Minimum cosine similarity threshold (inclusive).

        Returns:
            List of ``RetrievalResult`` objects sorted by relevance descending.
            Empty if the corpus is empty or no document meets ``min_relevance``.
        """
        if self._dirty:
            self._rebuild_index()

        if self._vectorizer is None or self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        results: list[RetrievalResult] = []
        for idx, score in enumerate(scores):
            if float(score) >= min_relevance:
                results.append(
                    RetrievalResult(
                        document=self._documents[idx],
                        relevance_score=float(score),
                    )
                )

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> list[str]:
    """Tokenise text to lowercase alphabetic tokens, removing stop words."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def check_groundedness(
    answer: str,
    sources: list[RetrievalResult],
    threshold: float = 0.5,
) -> GroundednessResult:
    """Evaluate how well an answer is grounded in retrieved source documents.

    Uses token overlap: what fraction of meaningful answer tokens appear in
    the source documents?  Matching is case-insensitive and stop words are
    excluded.

    Args:
        answer:    The model answer to evaluate.
        sources:   Retrieved source documents to check against.
        threshold: Minimum overall score to be considered grounded (0–1).

    Returns:
        A ``GroundednessResult`` with score, grounded flag, per-citation
        breakdown, ungrounded terms, and a human-readable explanation.
    """
    answer_tokens = set(_tokenise(answer))

    if not answer_tokens or not sources:
        overall_score = 0.0
        citations: list[Citation] = [
            Citation(
                document_id=r.document.id,
                document_title=r.document.title,
                source=r.document.source,
                relevance_score=r.relevance_score,
                groundedness_score=0.0,
            )
            for r in sources
        ]
        ungrounded_terms = sorted(answer_tokens)
        grounded = overall_score >= threshold
        explanation = _build_groundedness_explanation(
            overall_score, grounded, threshold, ungrounded_terms
        )
        return GroundednessResult(
            overall_score=overall_score,
            grounded=grounded,
            citations=citations,
            ungrounded_terms=ungrounded_terms,
            explanation=explanation,
        )

    all_source_tokens: set[str] = set()
    for result in sources:
        all_source_tokens.update(_tokenise(result.document.content))

    overlap = answer_tokens & all_source_tokens
    overall_score = len(overlap) / len(answer_tokens)
    ungrounded_terms = sorted(answer_tokens - all_source_tokens)

    citations = []
    for result in sources:
        doc_tokens = set(_tokenise(result.document.content))
        doc_overlap = answer_tokens & doc_tokens
        groundedness_score = len(doc_overlap) / len(answer_tokens)
        citations.append(
            Citation(
                document_id=result.document.id,
                document_title=result.document.title,
                source=result.document.source,
                relevance_score=result.relevance_score,
                groundedness_score=groundedness_score,
            )
        )

    grounded = overall_score >= threshold
    explanation = _build_groundedness_explanation(
        overall_score, grounded, threshold, ungrounded_terms
    )

    return GroundednessResult(
        overall_score=overall_score,
        grounded=grounded,
        citations=citations,
        ungrounded_terms=ungrounded_terms,
        explanation=explanation,
    )


def _build_groundedness_explanation(
    score: float,
    grounded: bool,
    threshold: float,
    ungrounded_terms: list[str],
) -> str:
    """Build a human-readable groundedness explanation."""
    status = "GROUNDED" if grounded else "UNGROUNDED"
    lines = [
        f"Groundedness: {status} ({score:.1%} of answer tokens found in sources; "
        f"threshold {threshold:.0%})"
    ]
    if ungrounded_terms:
        sample = ungrounded_terms[:5]
        more = len(ungrounded_terms) - len(sample)
        term_str = ", ".join(sample)
        if more > 0:
            term_str += f" ... (+{more} more)"
        lines.append(f"Ungrounded terms: {term_str}")
    return "  ".join(lines)


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------


def rag_query(
    query: str,
    knowledge_base: KnowledgeBase,
    answer: str,
    top_k: int = 3,
    groundedness_threshold: float = 0.5,
) -> RAGResult:
    """Run the full RAG explainability pipeline.

    Retrieves the most relevant documents for ``query``, evaluates how well
    ``answer`` is grounded in those documents, and produces a provenance trail.

    Note: ``answer`` is a parameter rather than being generated here — this
    harness assesses an existing answer, it does not call an LLM.

    Args:
        query:                  The query string.
        knowledge_base:         The document corpus to retrieve from.
        answer:                 The answer to assess for groundedness.
        top_k:                  Maximum documents to retrieve.
        groundedness_threshold: Minimum score to be considered grounded.

    Returns:
        A ``RAGResult`` with retrieved docs, groundedness assessment, and
        provenance trail.
    """
    retrieved = knowledge_base.retrieve(query, top_k=top_k)
    groundedness = check_groundedness(
        answer, retrieved, threshold=groundedness_threshold
    )
    provenance_trail = _build_provenance_trail(query, retrieved, answer, groundedness)

    return RAGResult(
        query=query,
        answer=answer,
        retrieved_documents=retrieved,
        groundedness=groundedness,
        provenance_trail=provenance_trail,
    )


def _build_provenance_trail(
    query: str,
    retrieved: list[RetrievalResult],
    answer: str,
    groundedness: GroundednessResult,
) -> str:
    """Build a human-readable provenance audit trail."""
    lines: list[str] = []
    lines.append("Provenance Trail")
    lines.append("-" * 40)
    lines.append(f"Query: {query}")
    lines.append("")
    lines.append("Retrieved Sources:")
    for i, result in enumerate(retrieved, 1):
        lines.append(
            f"  [{i}] {result.document.title} "
            f"(relevance: {result.relevance_score:.3f})"
        )
        lines.append(f"       Source: {result.document.source}")
    lines.append("")
    lines.append(f"Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")
    lines.append("")
    lines.append(groundedness.explanation)
    if groundedness.citations:
        lines.append("")
        lines.append("Per-source groundedness:")
        for citation in groundedness.citations:
            lines.append(
                f"  {citation.document_title}: "
                f"{citation.groundedness_score:.1%}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    """Demonstrate the RAG explainability pattern.

    Uses a small knowledge base of fictional AI regulation documents and three
    scenarios: a grounded answer, a hallucinating answer, and a source-mismatch
    answer.  No LLM calls are made.
    """
    print("Layer 06: Retrieval-Augmented Generation as Explainability")
    print("=" * 60)
    print()
    print("Pattern: retrieve sources, assess groundedness, surface citations.")
    print("No LLM calls — the answer is supplied; the harness evaluates it.")
    print()

    # -----------------------------------------------------------------------
    # Knowledge base — fictional AI regulatory documents
    # -----------------------------------------------------------------------
    kb = KnowledgeBase()
    kb.add_documents(
        [
            Document(
                id="doc-001",
                title="AI Safety Framework 2024",
                content=(
                    "All AI systems deployed in critical infrastructure must undergo "
                    "independent safety assessments before deployment. Systems must "
                    "include human oversight mechanisms and fail-safe protocols. "
                    "Developers must maintain audit logs for a minimum of five years."
                ),
                source="AI Safety Framework 2024, Section 3.2",
            ),
            Document(
                id="doc-002",
                title="Model Transparency Guidelines",
                content=(
                    "Organisations using AI systems must provide explanations for "
                    "automated decisions affecting individuals. Explanations must be "
                    "meaningful, accurate, and understandable to the affected person. "
                    "Black-box models must be accompanied by post-hoc explanation tools."
                ),
                source="Model Transparency Guidelines, Part II",
            ),
            Document(
                id="doc-003",
                title="Data Governance Standards",
                content=(
                    "Personal data used in AI training must be minimised, pseudonymised "
                    "where possible, and subject to regular audits. Data retention "
                    "periods must be documented and enforced. Cross-border data transfers "
                    "require explicit regulatory approval."
                ),
                source="Data Governance Standards v2.1, Article 7",
            ),
            Document(
                id="doc-004",
                title="Incident Response Protocol",
                content=(
                    "AI system failures must be reported to the regulatory authority "
                    "within 72 hours of detection. Incident reports must include root "
                    "cause analysis, affected population estimates, and remediation steps. "
                    "Systems causing harm must be suspended pending investigation."
                ),
                source="Incident Response Protocol 2023, Clause 9",
            ),
            Document(
                id="doc-005",
                title="Procurement Standards for AI",
                content=(
                    "Public sector procurement of AI systems requires competitive "
                    "tendering above defined thresholds. Vendors must demonstrate "
                    "compliance with safety and transparency requirements. Contracts "
                    "must include provisions for audit access and source code escrow."
                ),
                source="Procurement Standards for AI, Schedule 4",
            ),
        ]
    )

    print(f"Knowledge base: {kb.size} documents loaded.")
    print()

    # -----------------------------------------------------------------------
    # Scenario 1: grounded answer
    # -----------------------------------------------------------------------
    print("Scenario 1: Grounded answer")
    print("-" * 40)
    query_1 = "What are the audit log requirements for AI systems?"
    answer_1 = (
        "Developers must maintain audit logs for a minimum of five years. "
        "Systems must include human oversight mechanisms and fail-safe protocols."
    )
    result_1 = rag_query(query_1, kb, answer_1, top_k=3)
    print(result_1.provenance_trail)
    print()

    # -----------------------------------------------------------------------
    # Scenario 2: hallucinating answer
    # -----------------------------------------------------------------------
    print("Scenario 2: Hallucinating answer (terms absent from sources)")
    print("-" * 40)
    query_2 = "What are the penalties for non-compliance with AI regulations?"
    answer_2 = (
        "Non-compliance attracts fines up to 10 million euros or 4% of global "
        "annual turnover, whichever is higher. Repeat violations trigger criminal "
        "sanctions against senior executives."
    )
    result_2 = rag_query(query_2, kb, answer_2, top_k=3)
    print(result_2.provenance_trail)
    print()

    # -----------------------------------------------------------------------
    # Scenario 3: source mismatch
    # -----------------------------------------------------------------------
    print("Scenario 3: Source mismatch (retrieved docs don't match answer)")
    print("-" * 40)
    query_3 = "How should AI incidents be reported?"
    answer_3 = (
        "Procurement contracts must include audit access provisions and source "
        "code escrow requirements for all AI vendor agreements."
    )
    result_3 = rag_query(query_3, kb, answer_3, top_k=3)
    print(result_3.provenance_trail)
    print()

    print("See: layers/06-rag-explainability.md")


if __name__ == "__main__":  # pragma: no cover
    main()
