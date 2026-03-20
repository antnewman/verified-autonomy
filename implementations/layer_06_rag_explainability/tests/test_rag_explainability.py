"""
Tests for Layer 06: Retrieval-Augmented Generation as Explainability

Test naming convention: test_[function]_[scenario]_[expected_result]
Example: test_groundedness_check_returns_false_when_claim_absent_from_sources

Requirements (see CONTRIBUTING.md):
- 90% minimum coverage
- Edge cases at both boundaries
- Corner cases at intersections of multiple boundaries
- Failure mode tests (correct exceptions, messages, fallbacks)
- No mocking of the system under test
"""

from __future__ import annotations

import pytest

from rag_explainability import (
    Citation,
    Document,
    GroundednessResult,
    KnowledgeBase,
    RAGResult,
    RetrievalResult,
    _tokenise,
    check_groundedness,
    rag_query,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_doc(
    id: str = "doc-001",
    title: str = "Test Document",
    content: str = "the quick brown fox jumps over the lazy dog",
    source: str = "test-source",
) -> Document:
    return Document(id=id, title=title, content=content, source=source)


def _make_kb_with_docs(*docs: Document) -> KnowledgeBase:
    kb = KnowledgeBase()
    for doc in docs:
        kb.add_document(doc)
    return kb


# ---------------------------------------------------------------------------
# Document — frozen dataclass
# ---------------------------------------------------------------------------


class TestDocument:
    def test_document_stores_fields(self) -> None:
        doc = Document(id="x", title="T", content="c", source="s")
        assert doc.id == "x"
        assert doc.title == "T"
        assert doc.content == "c"
        assert doc.source == "s"

    def test_document_is_frozen(self) -> None:
        doc = Document(id="x", title="T", content="c", source="s")
        with pytest.raises(Exception):
            doc.id = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RetrievalResult — frozen dataclass
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def test_retrieval_result_stores_fields(self) -> None:
        doc = _make_doc()
        rr = RetrievalResult(document=doc, relevance_score=0.85)
        assert rr.document is doc
        assert rr.relevance_score == 0.85

    def test_retrieval_result_is_frozen(self) -> None:
        rr = RetrievalResult(document=_make_doc(), relevance_score=0.5)
        with pytest.raises(Exception):
            rr.relevance_score = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Citation — frozen dataclass
# ---------------------------------------------------------------------------


class TestCitation:
    def test_citation_stores_fields(self) -> None:
        c = Citation(
            document_id="d1",
            document_title="Title",
            source="src",
            relevance_score=0.7,
            groundedness_score=0.8,
        )
        assert c.document_id == "d1"
        assert c.document_title == "Title"
        assert c.source == "src"
        assert c.relevance_score == 0.7
        assert c.groundedness_score == 0.8

    def test_citation_is_frozen(self) -> None:
        c = Citation("d", "t", "s", 0.5, 0.5)
        with pytest.raises(Exception):
            c.document_id = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------


class TestTokenise:
    def test_tokenise_removes_stop_words(self) -> None:
        tokens = _tokenise("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_tokenise_lowercases(self) -> None:
        tokens = _tokenise("UPPER lower Mixed")
        assert all(t == t.lower() for t in tokens)

    def test_tokenise_removes_single_chars(self) -> None:
        tokens = _tokenise("a b c dog")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "dog" in tokens

    def test_tokenise_handles_punctuation(self) -> None:
        tokens = _tokenise("hello, world! foo-bar baz.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens
        assert "bar" in tokens
        assert "baz" in tokens

    def test_tokenise_empty_string_returns_empty(self) -> None:
        assert _tokenise("") == []

    def test_tokenise_all_stop_words_returns_empty(self) -> None:
        assert _tokenise("the and or but") == []


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------


class TestKnowledgeBase:
    def test_empty_kb_has_size_zero(self) -> None:
        kb = KnowledgeBase()
        assert kb.size == 0

    def test_add_document_increases_size(self) -> None:
        kb = KnowledgeBase()
        kb.add_document(_make_doc())
        assert kb.size == 1

    def test_add_documents_adds_multiple(self) -> None:
        kb = KnowledgeBase()
        kb.add_documents([_make_doc("a"), _make_doc("b")])
        assert kb.size == 2

    def test_constructor_with_documents(self) -> None:
        docs = [_make_doc("a"), _make_doc("b"), _make_doc("c")]
        kb = KnowledgeBase(documents=docs)
        assert kb.size == 3

    def test_retrieve_empty_kb_returns_empty(self) -> None:
        kb = KnowledgeBase()
        assert kb.retrieve("any query") == []

    def test_retrieve_returns_at_most_top_k(self) -> None:
        docs = [
            _make_doc(id=f"doc-{i}", content=f"machine learning model training epoch {i}")
            for i in range(10)
        ]
        kb = KnowledgeBase(documents=docs)
        results = kb.retrieve("machine learning training", top_k=3)
        assert len(results) <= 3

    def test_retrieve_sorted_by_relevance_descending(self) -> None:
        kb = KnowledgeBase()
        kb.add_document(
            Document(id="a", title="A", content="neural network deep learning", source="s")
        )
        kb.add_document(
            Document(id="b", title="B", content="unrelated content about cooking", source="s")
        )
        results = kb.retrieve("neural network", top_k=2)
        if len(results) >= 2:
            assert results[0].relevance_score >= results[1].relevance_score

    def test_retrieve_min_relevance_filters_low_scores(self) -> None:
        kb = KnowledgeBase()
        kb.add_document(
            Document(id="a", title="A", content="completely unrelated document", source="s")
        )
        results = kb.retrieve("quantum physics nuclear fusion", top_k=5, min_relevance=0.9)
        # High threshold — unrelated doc should be filtered out
        assert all(r.relevance_score >= 0.9 for r in results)

    def test_retrieve_min_relevance_zero_returns_all_matching(self) -> None:
        kb = KnowledgeBase()
        kb.add_document(Document(id="a", title="A", content="python programming language", source="s"))
        results = kb.retrieve("python", top_k=5, min_relevance=0.0)
        assert len(results) >= 1

    def test_retrieve_returns_retrieval_result_instances(self) -> None:
        kb = KnowledgeBase()
        kb.add_document(_make_doc(content="machine learning algorithms"))
        results = kb.retrieve("machine learning", top_k=1)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_result_document_matches_added(self) -> None:
        doc = Document(id="special", title="Special", content="unique content here", source="s")
        kb = KnowledgeBase(documents=[doc])
        results = kb.retrieve("unique content", top_k=1)
        assert len(results) == 1
        assert results[0].document.id == "special"

    def test_rebuild_index_after_add_document(self) -> None:
        kb = KnowledgeBase()
        kb.add_document(Document(id="a", title="A", content="python is great", source="s"))
        # First retrieve — builds index
        r1 = kb.retrieve("python", top_k=1)
        # Add another document — marks dirty
        kb.add_document(Document(id="b", title="B", content="java is verbose", source="s"))
        # Second retrieve — must rebuild
        r2 = kb.retrieve("java", top_k=1)
        assert len(r1) == 1
        assert len(r2) == 1
        assert r2[0].document.id == "b"

    def test_retrieve_top_k_zero_returns_empty(self) -> None:
        kb = KnowledgeBase()
        kb.add_document(_make_doc(content="some content"))
        results = kb.retrieve("some content", top_k=0)
        assert results == []


# ---------------------------------------------------------------------------
# check_groundedness
# ---------------------------------------------------------------------------


class TestCheckGroundedness:
    def _make_source(self, content: str, score: float = 0.9) -> RetrievalResult:
        return RetrievalResult(
            document=Document(id="d", title="T", content=content, source="s"),
            relevance_score=score,
        )

    def test_fully_grounded_answer(self) -> None:
        sources = [self._make_source("machine learning model training data")]
        result = check_groundedness(
            "machine learning model training", sources, threshold=0.5
        )
        assert result.grounded is True
        assert result.overall_score > 0.5

    def test_ungrounded_answer_scores_low(self) -> None:
        sources = [self._make_source("machine learning model training data")]
        result = check_groundedness(
            "quantum physics nuclear fusion reactor", sources, threshold=0.5
        )
        assert result.grounded is False
        assert result.overall_score < 0.5

    def test_empty_answer_returns_zero_score(self) -> None:
        sources = [self._make_source("some content")]
        result = check_groundedness("", sources, threshold=0.5)
        assert result.overall_score == 0.0
        assert result.grounded is False

    def test_empty_sources_returns_zero_score(self) -> None:
        result = check_groundedness("some answer here", [], threshold=0.5)
        assert result.overall_score == 0.0
        assert result.grounded is False

    def test_empty_answer_empty_sources_returns_zero(self) -> None:
        result = check_groundedness("", [], threshold=0.5)
        assert result.overall_score == 0.0
        assert result.grounded is False

    def test_returns_groundedness_result(self) -> None:
        result = check_groundedness("answer", [], threshold=0.5)
        assert isinstance(result, GroundednessResult)

    def test_citations_count_matches_sources(self) -> None:
        sources = [
            self._make_source("content one", score=0.9),
            self._make_source("content two", score=0.8),
        ]
        result = check_groundedness("content", sources, threshold=0.5)
        assert len(result.citations) == len(sources)

    def test_citation_is_citation_instance(self) -> None:
        sources = [self._make_source("machine learning")]
        result = check_groundedness("machine learning", sources, threshold=0.5)
        assert all(isinstance(c, Citation) for c in result.citations)

    def test_ungrounded_terms_are_sorted(self) -> None:
        sources = [self._make_source("unrelated")]
        result = check_groundedness("zoo apple banana", sources, threshold=0.5)
        assert result.ungrounded_terms == sorted(result.ungrounded_terms)

    def test_ungrounded_terms_absent_from_sources(self) -> None:
        sources = [self._make_source("machine learning algorithms")]
        result = check_groundedness("quantum mechanics physics", sources)
        # All non-stop-word answer tokens should be ungrounded
        assert len(result.ungrounded_terms) > 0

    def test_grounded_terms_not_in_ungrounded_list(self) -> None:
        sources = [self._make_source("machine learning model")]
        result = check_groundedness("machine learning", sources, threshold=0.0)
        from rag_explainability import _tokenise as tok
        answer_tokens = set(tok("machine learning"))
        source_tokens = set(tok("machine learning model"))
        grounded = answer_tokens & source_tokens
        for term in grounded:
            assert term not in result.ungrounded_terms

    def test_threshold_boundary_at_exact_score(self) -> None:
        # Token overlap: if answer has 4 meaningful tokens and all 4 are in source
        sources = [self._make_source("alpha beta gamma delta epsilon")]
        result = check_groundedness(
            "alpha beta gamma delta", sources, threshold=1.0
        )
        # overall_score should be 1.0 → grounded
        assert result.grounded is True

    def test_threshold_boundary_just_below(self) -> None:
        sources = [self._make_source("alpha beta gamma delta")]
        result = check_groundedness(
            "alpha beta gamma delta outsider", sources, threshold=1.0
        )
        # 4 of 5 tokens grounded → 0.8 < 1.0 → not grounded
        assert result.grounded is False

    def test_per_citation_groundedness_score_in_range(self) -> None:
        sources = [self._make_source("machine learning training")]
        result = check_groundedness("machine learning", sources)
        for citation in result.citations:
            assert 0.0 <= citation.groundedness_score <= 1.0

    def test_multiple_sources_union_increases_score(self) -> None:
        source_a = self._make_source("machine learning model", score=0.9)
        source_b = self._make_source("training data neural network", score=0.8)
        result_single = check_groundedness(
            "machine learning training neural network", [source_a], threshold=0.0
        )
        result_both = check_groundedness(
            "machine learning training neural network", [source_a, source_b], threshold=0.0
        )
        assert result_both.overall_score >= result_single.overall_score

    def test_explanation_contains_grounded_status(self) -> None:
        sources = [self._make_source("machine learning")]
        result = check_groundedness("machine learning", sources, threshold=0.0)
        assert "GROUNDED" in result.explanation

    def test_explanation_contains_ungrounded_status(self) -> None:
        sources = [self._make_source("unrelated text")]
        result = check_groundedness("quantum physics reactor", sources, threshold=0.9)
        assert "UNGROUNDED" in result.explanation

    def test_stop_words_only_answer_scores_zero(self) -> None:
        sources = [self._make_source("the quick brown fox")]
        result = check_groundedness("the and or but", sources, threshold=0.5)
        assert result.overall_score == 0.0

    def test_default_threshold_is_0_5(self) -> None:
        sources = [self._make_source("alpha beta gamma delta")]
        result_default = check_groundedness("alpha beta gamma delta", sources)
        result_explicit = check_groundedness(
            "alpha beta gamma delta", sources, threshold=0.5
        )
        assert result_default.grounded == result_explicit.grounded


# ---------------------------------------------------------------------------
# rag_query
# ---------------------------------------------------------------------------


class TestRagQuery:
    def _build_kb(self) -> KnowledgeBase:
        return KnowledgeBase(
            documents=[
                Document(
                    id="r1",
                    title="Regulation A",
                    content="safety assessment required before deployment systems",
                    source="Reg A, §1",
                ),
                Document(
                    id="r2",
                    title="Regulation B",
                    content="transparency explanation automated decisions required",
                    source="Reg B, §2",
                ),
            ]
        )

    def test_rag_query_returns_rag_result(self) -> None:
        kb = self._build_kb()
        result = rag_query("safety assessment", kb, "safety assessment required")
        assert isinstance(result, RAGResult)

    def test_rag_query_stores_query(self) -> None:
        kb = self._build_kb()
        result = rag_query("safety assessment", kb, "some answer")
        assert result.query == "safety assessment"

    def test_rag_query_stores_answer(self) -> None:
        kb = self._build_kb()
        result = rag_query("safety", kb, "specific answer text")
        assert result.answer == "specific answer text"

    def test_rag_query_retrieved_documents_not_empty(self) -> None:
        kb = self._build_kb()
        result = rag_query("safety assessment", kb, "some answer")
        assert len(result.retrieved_documents) > 0

    def test_rag_query_retrieved_documents_are_retrieval_results(self) -> None:
        kb = self._build_kb()
        result = rag_query("safety", kb, "answer")
        assert all(isinstance(r, RetrievalResult) for r in result.retrieved_documents)

    def test_rag_query_groundedness_is_groundedness_result(self) -> None:
        kb = self._build_kb()
        result = rag_query("safety", kb, "answer")
        assert isinstance(result.groundedness, GroundednessResult)

    def test_rag_query_provenance_trail_is_string(self) -> None:
        kb = self._build_kb()
        result = rag_query("safety", kb, "answer")
        assert isinstance(result.provenance_trail, str)

    def test_rag_query_provenance_trail_contains_query(self) -> None:
        kb = self._build_kb()
        result = rag_query("unique_query_phrase", kb, "answer")
        assert "unique_query_phrase" in result.provenance_trail

    def test_rag_query_provenance_trail_contains_answer(self) -> None:
        kb = self._build_kb()
        result = rag_query("safety", kb, "specific_answer_token")
        assert "specific_answer_token" in result.provenance_trail

    def test_rag_query_top_k_limits_retrieved(self) -> None:
        kb = self._build_kb()
        result = rag_query("safety assessment transparency", kb, "answer", top_k=1)
        assert len(result.retrieved_documents) <= 1

    def test_rag_query_grounded_answer_detected(self) -> None:
        kb = self._build_kb()
        result = rag_query(
            "safety assessment",
            kb,
            "safety assessment required before deployment systems",
            groundedness_threshold=0.5,
        )
        assert result.groundedness.grounded is True

    def test_rag_query_ungrounded_answer_detected(self) -> None:
        kb = self._build_kb()
        result = rag_query(
            "safety assessment",
            kb,
            "quantum physics nuclear fusion antimatter reactor",
            groundedness_threshold=0.5,
        )
        assert result.groundedness.grounded is False

    def test_rag_query_empty_kb_returns_empty_retrieved(self) -> None:
        kb = KnowledgeBase()
        result = rag_query("any query", kb, "some answer")
        assert result.retrieved_documents == []

    def test_rag_query_groundedness_threshold_respected(self) -> None:
        kb = self._build_kb()
        result_low = rag_query(
            "safety", kb, "safety assessment", groundedness_threshold=0.0
        )
        # Low threshold → grounded; high threshold may not be
        assert result_low.groundedness.grounded is True


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_pipeline_grounded_scenario(self) -> None:
        """End-to-end: grounded answer scores well and is marked grounded."""
        kb = KnowledgeBase()
        kb.add_document(
            Document(
                id="law-001",
                title="AI Safety Law",
                content=(
                    "All AI systems must undergo independent safety assessments. "
                    "Developers must maintain audit logs for five years."
                ),
                source="AI Safety Law §3",
            )
        )
        result = rag_query(
            query="What are the audit log requirements?",
            knowledge_base=kb,
            answer="Developers must maintain audit logs for five years.",
            groundedness_threshold=0.5,
        )
        assert result.groundedness.grounded is True
        assert result.groundedness.overall_score > 0.5
        assert len(result.retrieved_documents) >= 1

    def test_full_pipeline_hallucination_scenario(self) -> None:
        """End-to-end: hallucinated answer scores poorly and is marked ungrounded."""
        kb = KnowledgeBase()
        kb.add_document(
            Document(
                id="law-001",
                title="AI Safety Law",
                content="Audit logs must be retained for five years.",
                source="AI Safety Law §3",
            )
        )
        result = rag_query(
            query="What are the penalties for non-compliance?",
            knowledge_base=kb,
            answer="Criminal sanctions and ten million euro fines apply.",
            groundedness_threshold=0.5,
        )
        assert result.groundedness.grounded is False

    def test_full_pipeline_provenance_trail_structure(self) -> None:
        """Provenance trail must contain key structural elements."""
        kb = KnowledgeBase()
        kb.add_document(
            Document(
                id="d1",
                title="Source Document",
                content="machine learning safety alignment",
                source="ref-001",
            )
        )
        result = rag_query("machine learning safety", kb, "machine learning safety")
        trail = result.provenance_trail
        assert "Provenance Trail" in trail
        assert "Query:" in trail
        assert "Retrieved Sources:" in trail
        assert "Answer:" in trail
        assert "Groundedness:" in trail
