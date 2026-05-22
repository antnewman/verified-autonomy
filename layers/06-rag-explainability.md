# Layer 06: Retrieval-Augmented Generation as Explainability

*Shift trust from model intent to source facticity*

> **Primary author:** Ant Newman
>
> **Production examples:** Reference implementation; field examples from clinical pharmacology, enterprise RAG, and compliance checking

---

## What It Is

Retrieval-Augmented Generation is usually discussed as an accuracy technique — ground the model in source documents so it hallucinates less. This layer reframes RAG as something more fundamental: an explainability and trust mechanism.

When a model's output includes citations to source documents, the trust question changes. Instead of asking "do I trust this model's opaque internal reasoning?" — a question nobody can answer — the reviewer asks "do I trust these source documents?" That is a question a human can answer without understanding how the model works. The trust shifts from model intent to source facticity.

The technique works in three parts:

**Evidence-based retrieval** finds the source documents most relevant to the query. The implementation uses TF-IDF vectorisation with cosine similarity — a well-understood, transparent retrieval method. Each retrieved document carries a relevance score indicating how closely it matches the query. The retrieval step answers: "which sources did the system consult?"

**Groundedness checking** evaluates whether the generated answer is actually supported by the retrieved documents. This is the critical verification step that most RAG implementations skip. The system tokenises the answer and the source documents, removes common stop words, and measures what fraction of substantive answer tokens appear in at least one source. The overall groundedness score tells you how much of the answer is traceable to the sources. Per-document citation scores tell you which specific source supports which claims. And the **ungrounded terms** list identifies the specific tokens in the answer that appear in no source — these are the terms most likely to be hallucinated.

**The provenance trail** ties it all together: query → retrieved documents → answer → groundedness assessment. A human reviewer can follow the chain from question to answer to sources to verification in a single readable output. This is the paper trail that makes the system's reasoning auditable without requiring the reviewer to understand the model.

The groundedness check produces a binary verdict: GROUNDED (the answer is sufficiently supported by the sources) or UNGROUNDED (it is not). But the real value is in the detail — the per-source scores that show which document supports which claims, and the ungrounded terms list that shows exactly where the model went beyond its evidence.

## Why It Matters

Consider a compliance checking system that processes pharmaceutical regulatory submissions. The model reads a submission, retrieves relevant regulatory guidelines from a document store, and produces an assessment: "this submission meets requirements X, Y, and Z, but has a gap in requirement W." If the assessment includes citations — "requirement X is defined in Guideline 2024/15, Section 4.2" — the compliance officer can verify each claim against the cited source. The trust is in the sources, not the model.

In clinical pharmacology, RAG-based compliance systems have been reported to achieve high accuracy in identifying compliance gaps with exact regulatory paragraph citations. The mechanism is the same: the model does not need to be trusted to know the regulation. It needs to retrieve the right document, cite the right paragraph, and produce an assessment that is traceable to those citations. If the citations check out, the assessment can be trusted. If they do not, the gap is visible.

Without groundedness checking, a RAG system can fail in three specific ways that are invisible to the end user:

**Hallucinating beyond context.** The model retrieves relevant documents, reads them, and then generates claims that go beyond what the documents actually say. The citations look plausible — the right documents are referenced — but the specific claim is not in those documents. This is the most dangerous failure mode because it looks authoritative.

**Source mismatch.** The model retrieves documents that are relevant to the general topic but do not actually support the specific claims being made. The retrieval relevance score is high (the documents are about the right topic) but the groundedness score is low (the documents do not contain the specific information cited). The system is citing the right neighbourhood but the wrong address.

**Poor retrieval.** The document store contains the information needed to answer the query, but the retrieval step fails to find it. The model then either refuses to answer (best case) or generates an answer from its internal knowledge without source grounding (worst case). The provenance trail reveals this: zero or low-relevance documents retrieved despite the answer being confident.

The failure mode across all three cases is specific: **the model produces an answer that looks grounded — it has citations, it references sources, it sounds authoritative — but the citations do not actually support the claims.** Without a groundedness check, this is invisible. With one, it is caught.

## Production Example

The implementation is open source in the Verified Autonomy repository at [`implementations/layer_06_rag_explainability/`](https://github.com/antnewman/verified-autonomy/tree/main/implementations/layer_06_rag_explainability).

**Run it yourself in under two minutes:**

```bash
cd implementations/layer_06_rag_explainability
make install
make run
```

**Building a knowledge base and checking groundedness:**

```python
from verified_autonomy.rag_explainability import (
    Document, KnowledgeBase, check_groundedness, rag_query,
)

# Build a document store
kb = KnowledgeBase()
kb.add_documents([
    Document(
        id="doc-001",
        title="AI Safety Framework 2024",
        content="All AI systems deployed in critical infrastructure must undergo "
                "independent safety assessments. Developers must maintain audit "
                "logs for a minimum of five years.",
        source="AI Safety Framework 2024, Section 3.2",
    ),
    # ... additional documents
])

# Run the full pipeline
result = rag_query(
    query="What are the audit log requirements?",
    knowledge_base=kb,
    answer="Developers must maintain audit logs for a minimum of five years.",
    top_k=3,
)

result.groundedness.grounded        # True
result.groundedness.overall_score   # 1.0 — every answer token found in sources
result.groundedness.ungrounded_terms  # [] — nothing unsupported
print(result.provenance_trail)
```

**The three scenarios from `make run`:**

Scenario 1 demonstrates a grounded answer. The query asks about audit log requirements. The answer — "Developers must maintain audit logs for a minimum of five years. Systems must include human oversight mechanisms and fail-safe protocols." — is drawn directly from the source documents. Groundedness score: 100%. Every substantive token in the answer appears in at least one retrieved source.

Scenario 2 demonstrates hallucination beyond context. The query asks about penalties for non-compliance. The answer claims "fines up to 10 million euros" and "criminal sanctions against senior executives" — but no document in the knowledge base contains any information about penalties, fines, or criminal sanctions. Groundedness score: 5.3%. The ungrounded terms list identifies the specific fabricated terms: "criminal," "prosecution," "euros," "annual," and others. These are the hallucinated claims, identified by name.

Scenario 3 demonstrates source mismatch. The query asks about incident reporting, but the answer discusses procurement contracts and source code escrow — content from a different document that was retrieved because of topical similarity but does not address the actual question. The documents are retrieved (the system found sources) but the answer does not match those sources. Groundedness score: 21.4%.

**The provenance trail (from the test suite, 61 tests, 100% coverage):** Every `RAGResult` includes a `provenance_trail` — a human-readable audit string documenting the complete chain: which query was asked, which documents were retrieved (with relevance scores), what answer was assessed, whether it was grounded, and which specific terms were unsupported. The integration tests verify that grounded answers produce GROUNDED verdicts, hallucinating answers produce UNGROUNDED verdicts with non-empty ungrounded terms lists, and the provenance trail contains the expected structural elements.

**A note on the implementation:** This module takes the answer as a parameter rather than generating it, because the implementation does not include an LLM. In production, the answer would be generated by a model using the retrieved documents as context. The groundedness checking and provenance trail infrastructure is identical regardless of how the answer is produced — the harness evaluates any answer against any set of source documents.

**Code ancestry:** This implementation uses scikit-learn's `TfidfVectorizer` for retrieval and token-overlap for groundedness scoring. In production, practitioners would use embedding-based retrieval (e.g. sentence transformers) and more sophisticated groundedness checkers (e.g. NLI-based models). The architecture — retrieve, check, cite, produce provenance trail — is the same at any level of sophistication.

## Limitations

**Only as good as the source document store.** RAG shifts trust from the model to the sources. If the sources are incomplete, outdated, or incorrect, the model will produce grounded answers that are grounded in bad information. A groundedness score of 100% means "the answer is fully supported by the sources" — it does not mean "the answer is correct." Source quality is a prerequisite, not a guarantee.

**Source mismatch is subtle and dangerous.** A model can cite documents that are relevant to the general topic but do not support the specific claim. The retrieval relevance score is high (the document is about the right thing) but the groundedness score is low (the document does not say what the answer claims it says). This failure mode is particularly dangerous because the citations look plausible on a quick review. Only a careful check of whether the cited source actually contains the specific claim catches it — which is exactly what the groundedness checker automates.

**Hallucinating beyond context remains a documented failure mode even with RAG.** The model reads the retrieved documents, generates an answer, and then continues generating beyond what the documents support. The first half of the answer is grounded; the second half is fabricated. The groundedness score catches this (it will be below 100%) but the failure is that the model generated the unsupported claims in the first place. RAG reduces hallucination. It does not eliminate it.

**Token-overlap groundedness is a proxy, not a semantic check.** The implementation measures whether the same words appear in the answer and the sources. It does not measure whether the answer's meaning is supported by the sources. An answer could use different words to express the same claim (paraphrasing), or use the same words to express a different claim (misleading citation). Production groundedness checkers use Natural Language Inference (NLI) models to assess semantic entailment — whether the source actually implies the claim. The token-overlap approach is transparent and dependency-light but less precise.

**Retrieval quality limits everything downstream.** If the retrieval step fails to find the relevant documents, the groundedness check has nothing to verify against. The system might score an answer as ungrounded not because it is fabricated but because the right sources were not retrieved. Retrieval quality — recall (finding all relevant documents) and precision (not retrieving irrelevant ones) — is the foundation that the entire explainability pipeline rests on.

**Computational cost scales with document store size.** TF-IDF retrieval scales well, but embedding-based retrieval (the production standard) requires computing embeddings for all documents and maintaining a vector index. For large document stores (millions of documents), retrieval infrastructure becomes a significant engineering and cost concern. The explainability benefit of RAG comes with an infrastructure obligation.

## Reader Takeaway

RAG is not just an accuracy technique. It is the mechanism that makes an AI system's reasoning auditable by shifting the trust question from "do I trust the model?" to "do I trust these sources?" — a question humans can answer. But groundedness checking is essential: without it, the system can cite the right documents while making unsupported claims, and the failure is invisible. Build three things into any RAG pipeline: evidence-based retrieval with relevance scores, groundedness verification that checks whether the answer is actually supported by the sources, and a provenance trail that a reviewer can follow from question to answer to evidence. The provenance trail is what makes the system's reasoning visible, and visibility is what makes trust possible.

---

*Contributor note: Ant Newman wrote the primary draft and implementation. Uses scikit-learn for TF-IDF retrieval; groundedness scoring implemented from scratch using token overlap with stop-word removal.*
