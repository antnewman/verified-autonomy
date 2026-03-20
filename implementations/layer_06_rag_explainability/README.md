# Layer 06: Retrieval-Augmented Generation as Explainability — Implementation

## Quick Start

```bash
make install
make run
```

## Expected Output

```
Layer 06: Retrieval-Augmented Generation as Explainability
============================================================

Pattern: retrieve sources, assess groundedness, surface citations.
No LLM calls — the answer is supplied; the harness evaluates it.

Knowledge base: 5 documents loaded.

Scenario 1: Grounded answer
----------------------------------------
Provenance Trail
----------------------------------------
Query: What are the audit log requirements for AI systems?

Retrieved Sources:
  [1] Procurement Standards for AI (relevance: 0.286)
       Source: Procurement Standards for AI, Schedule 4
  [2] AI Safety Framework 2024 (relevance: 0.204)
       Source: AI Safety Framework 2024, Section 3.2
  [3] Model Transparency Guidelines (relevance: 0.163)
       Source: Model Transparency Guidelines, Part II

Answer: Developers must maintain audit logs for a minimum of five years. Systems must include human oversight mechanisms and fai...

Groundedness: GROUNDED (100.0% of answer tokens found in sources; threshold 50%)

Per-source groundedness:
  Procurement Standards for AI: 25.0%
  AI Safety Framework 2024: 100.0%
  Model Transparency Guidelines: 12.5%

Scenario 2: Hallucinating answer (terms absent from sources)
----------------------------------------
...
Groundedness: UNGROUNDED (5.3% of answer tokens found in sources; threshold 50%)  Ungrounded terms: against, annual, attracts, criminal, euros ... (+13 more)

Scenario 3: Source mismatch (retrieved docs don't match answer)
----------------------------------------
...
Groundedness: UNGROUNDED (21.4% of answer tokens found in sources; threshold 50%)  Ungrounded terms: access, agreements, audit, code, contracts ... (+6 more)

See: layers/06-rag-explainability.md
```

## What This Code Does

Implements a structured RAG-based explainability harness demonstrating how to
ground model outputs in verifiable source documents.  The system takes a query
and a pre-written answer (no LLM calls are made), retrieves the most relevant
documents from the knowledge base using TF-IDF cosine similarity, and evaluates
how well the answer is grounded in those documents using token overlap with
stop-word removal.  Results include an overall groundedness score, per-document
citation scores, a list of ungrounded terms (potential hallucinations), and a
human-readable provenance trail.  Three demo scenarios demonstrate grounded,
hallucinating, and source-mismatch answers against a fictional AI regulatory
knowledge base.

## API

| Symbol | Signature | Description |
|---|---|---|
| `Document` | frozen dataclass | id, title, content, source |
| `RetrievalResult` | frozen dataclass | document, relevance_score (cosine similarity) |
| `Citation` | frozen dataclass | document_id, document_title, source, relevance_score, groundedness_score |
| `GroundednessResult` | frozen dataclass | overall_score, grounded, citations, ungrounded_terms, explanation |
| `RAGResult` | dataclass | query, answer, retrieved_documents, groundedness, provenance_trail |
| `KnowledgeBase.__init__` | `(documents=None)` | Corpus of documents with lazy TF-IDF index |
| `KnowledgeBase.add_document` | `(document) → None` | Add one document; marks index dirty |
| `KnowledgeBase.add_documents` | `(documents) → None` | Add multiple documents |
| `KnowledgeBase.retrieve` | `(query, top_k=3, min_relevance=0.0) → list[RetrievalResult]` | TF-IDF cosine similarity retrieval |
| `check_groundedness` | `(answer, sources, threshold=0.5) → GroundednessResult` | Token-overlap groundedness assessment |
| `rag_query` | `(query, knowledge_base, answer, top_k=3, groundedness_threshold=0.5) → RAGResult` | Full RAG pipeline — retrieve, check, report |

## Testing

```bash
make test
```

61 tests, 100% coverage.  Tests cover: frozen dataclass fields; `_tokenise`
stop-word removal, lowercasing, punctuation handling, empty input;
`KnowledgeBase` size, add/retrieve, lazy index rebuild, top_k, min_relevance;
`check_groundedness` grounded/ungrounded detection, empty answer, empty sources,
boundary thresholds, citation count, sorted ungrounded terms, per-source scores,
multi-source union; `rag_query` fields, groundedness detection, top_k, empty KB;
integration grounded/hallucination/provenance trail structure.

## Dependencies

- `numpy>=1.26` — required by scikit-learn for numerical array operations.
- `scikit-learn>=1.4` — `TfidfVectorizer` for TF-IDF indexing; `cosine_similarity` for retrieval scoring.
