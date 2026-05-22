# Layer 03: Making Failures Visible

*The potential hallucinations list*

> **Primary author:** Ant Newman
>
> **Production examples:** Generalised from [agent-task-planning](https://github.com/antnewman/pda-platform/tree/main/packages/agent-task-planning) in the PDA Platform

---

## What It Is

Most AI systems have two output categories: results they return and results they discard. This layer introduces a third category: results the system is suspicious of, separated from results it trusts, and made visible to human reviewers.

The technique works in three parts.

**Quality scoring** evaluates each model-generated sample against three dimensions — coherence (internal consistency), relevance (how on-topic the output is), and completeness (how fully it addresses the request) — weighted at 40%, 40%, and 20% respectively. The composite score is then penalised for high semantic entropy: when the model produces markedly different outputs across repeated runs for the same prompt, that variance is a signal that something is wrong. The entropy penalty activates only above 0.5 (a modest amount of variation is tolerated) and scales linearly — at maximum entropy of 1.0, the penalty is 0.25, enough to push a borderline output below the default quality threshold of 0.6.

The formula:

```
entropy_penalty = max(0, semantic_entropy - 0.5) × 0.5
overall = (coherence × 0.4 + relevance × 0.4 + completeness × 0.2) - entropy_penalty
```

**The potential hallucinations list** flags any sample whose semantic entropy exceeds a threshold (default 0.7). These are not suppressed. They are not filtered out. They are published alongside the model's outputs as a separate list: "these are the outputs we are suspicious of." The flag is a signal to downstream consumers that these specific outputs warrant human attention before being acted upon.

**The quality pass rate** reports the fraction of samples that cleared the quality bar. This is a system-level health metric — if the pass rate is dropping over time, the model's reliability is degrading and the system needs attention.

The critical design principle is that the suspicious outputs remain visible. A system that silently filters its bad outputs appears more reliable than it is. A system that shows you which outputs it does not trust gives you the information you need to make the right decision.

## Why It Matters

Consider a system that generates summaries of legal documents. It processes a batch of fifty documents and returns summaries for all fifty. Forty-seven of those summaries are coherent, relevant, and complete. Three are subtly wrong — they contain statements that sound plausible but are not supported by the source document. The model generated them with high variance across repeated samples, but the final output looks clean.

If the system silently returns all fifty summaries, the three bad ones are invisible. They look exactly like the forty-seven good ones. A human reviewer reading all fifty has no signal about which ones to scrutinise. In a legal context, a plausible-sounding summary that misrepresents the source document is more dangerous than an obviously broken one — it will be relied upon because it looks authoritative.

If the system filters out the three suspicious summaries, it returns forty-seven. The reviewer sees a clean set and assumes the system handled everything correctly. The three missing documents are either not noticed or assumed to have been handled elsewhere. The system's apparent reliability is 100%. Its actual reliability is 94%.

If the system returns all fifty summaries plus a separate list — "these three summaries have elevated hallucination risk: document 12, document 31, document 48" — the reviewer knows exactly where to focus attention. The system's honesty is visible. The three flagged summaries might be fine. They might not. But the reviewer has the information they need to check.

The failure mode is specific: **silent filtering creates a false picture of system reliability. Systems that only show their confident outputs appear more reliable than they are. In high-stakes domains, a confident wrong answer that passes silently is more dangerous than a flagged suspicious answer that gets checked.**

## Production Example

The implementation is open source in the Verified Autonomy repository at [`implementations/layer_03_visible_failures/`](https://github.com/antnewman/verified-autonomy/tree/main/implementations/layer_03_visible_failures).

**Run it yourself in under two minutes:**

```bash
cd implementations/layer_03_visible_failures
make install
make run
```

**The core functions:**

```python
from verified_autonomy.visible_failures import (
    compute_quality_score,
    flag_potential_hallucinations,
    compute_pass_rate,
    Sample,
)

# Score individual samples
score = compute_quality_score(
    coherence=0.60,
    relevance=0.65,
    completeness=0.55,
    semantic_entropy=0.75,
)
score.overall           # 0.49 — below the 0.6 threshold
score.passed_threshold  # False

# Build a batch of samples and flag suspicious ones
samples = [
    Sample(id="sample_01", quality=compute_quality_score(0.90, 0.88, 0.85, 0.20)),
    Sample(id="sample_02", quality=compute_quality_score(0.75, 0.80, 0.70, 0.45)),
    Sample(id="sample_03", quality=compute_quality_score(0.60, 0.65, 0.55, 0.75)),
    Sample(id="sample_04", quality=compute_quality_score(0.85, 0.82, 0.78, 0.30)),
    Sample(id="sample_05", quality=compute_quality_score(0.40, 0.35, 0.50, 0.92)),
]

flagged = flag_potential_hallucinations(samples)
# Returns: ["sample_03", "sample_05"]

pass_rate = compute_pass_rate(samples)
# Returns: 0.6 — three of five samples passed
```

**What the output looks like:**

```
Sample quality scores:
  ID            Overall  Pass?  Entropy
  ------------  -------  -----  -------
  sample_01      0.8760    yes     0.20
  sample_02      0.7500    yes     0.45
  sample_03      0.4900     no     0.75
  sample_04      0.8280    yes     0.30
  sample_05      0.1050     no     0.92

Quality pass rate: 60%  (3/5 samples)

Potential hallucinations (2 flagged):
  - sample_03
  - sample_05
```

Notice that the two flagged samples have different failure profiles. Sample 03 has moderate quality dimensions (coherence 0.60, relevance 0.65, completeness 0.55) but high entropy (0.75) — the model was uncertain and produced different outputs across runs. Sample 05 has both low quality and very high entropy (0.92) — the model was uncertain and the output it produced was poor. Both are flagged. Both are visible. A reviewer can prioritise accordingly.

**The entropy penalty in action (from the test suite, 45 tests, 98% coverage):**

The test suite verifies that entropy below 0.5 incurs no penalty at all — `compute_quality_score` with entropy 0.3 and entropy 0.5 produce identical overall scores. Above 0.5, the penalty scales linearly: at entropy 1.0, the penalty is exactly 0.25. The test `test_max_entropy_penalty_is_quarter` confirms this by computing the difference between zero-entropy and max-entropy scores for identical quality dimensions. This means a sample with otherwise acceptable quality (overall 0.7 without penalty) can be pushed below the 0.6 threshold by high entropy alone — the test `test_high_entropy_pushes_borderline_below_threshold` verifies this explicitly.

The hallucination flag uses a strict greater-than comparison: entropy exactly at the threshold (0.7) is not flagged, but 0.701 is. This boundary behaviour is tested explicitly because it determines which outputs appear on the potential hallucinations list.

**Code ancestry:** This implementation is generalised from the `agent-task-planning` package in the [PDA Platform](https://github.com/antnewman/pda-platform), where it was originally used to flag suspicious outputs from AI-assisted project schedule analysis. The quality scoring formula, entropy penalty, and three-category output model are identical; the domain-specific terminology has been removed.

## Limitations

**Requires human capacity to review flagged outputs.** The potential hallucinations list is only useful if someone reads it. In a high-volume system producing thousands of outputs per hour, a 5% flag rate means fifty flagged outputs per hour requiring human attention. If reviewer capacity does not scale with output volume, the flagged list becomes another queue that nobody reads — and the system is back to silent filtering in practice if not in design.

**Alert fatigue if the threshold is too low.** If the entropy threshold is set too aggressively (too low), too many outputs are flagged. Reviewers stop treating the flags seriously. The potential hallucinations list becomes the list that always has items on it, and the signal is lost in the noise. Tuning the threshold for a specific deployment is an empirical process — it requires monitoring the flag rate and the proportion of flagged outputs that turn out to be genuinely problematic. The default threshold of 0.7 is a starting point, not a universal answer.

**Semantic entropy is a proxy, not a guarantee.** High semantic entropy (high variance across repeated model runs) is a reliable proxy for confabulation risk, but it is not a direct measurement of whether the output is correct. A model can produce consistent outputs (low entropy) that are consistently wrong. A model can produce variable outputs (high entropy) where the final selection happens to be correct. The potential hallucinations list catches the high-variance case. It does not catch the confident-but-wrong case — that requires different techniques (see Layer 04: Calibration and Layer 06: RAG as Explainability).

**Commercially uncomfortable.** Showing clients or stakeholders a list of outputs the system is suspicious of makes the system look less reliable than one that only shows its confident results. This is the point. But it creates a commercial tension: a competitor whose system silently filters will appear more capable in a demo, even if it is less trustworthy in production. Organisations that adopt this technique need to be prepared to explain why visible uncertainty is a feature, not a deficiency. The closing line of this field guide — "trust is built not by making the system more confident, but by making it more honest about where it is not" — is the argument to make.

**The quality dimensions are weighted, not learned.** The weights (coherence 40%, relevance 40%, completeness 20%) are fixed in the implementation. In practice, different domains may value these dimensions differently — a creative writing system might weight completeness higher, while a factual extraction system might weight relevance higher. The current implementation does not support configurable weights. Production deployments should parameterise these values based on domain requirements.

**Requires multiple model samples to compute entropy.** Like Layer 02, this technique depends on running the same prompt through the model multiple times to measure variance. The semantic entropy value passed to `compute_quality_score` must come from somewhere — typically from comparing the outputs of multiple runs. This multiplies inference cost. Systems that can only afford a single model pass per input cannot use entropy-based hallucination detection and must rely on other signals.

## Reader Takeaway

Build three output categories, not two. Results the system trusts, results the system rejects, and results the system is suspicious of — published explicitly, visible to human reviewers, never silently filtered. The potential hallucinations list is not a sign of weakness. It is the mechanism by which the system tells you where it needs help. A system that shows its working builds more trust than one that curates its presentation, and the quality pass rate gives you a single metric to monitor system health over time. If that number is dropping, the model is degrading and the system needs attention before anyone notices from the outputs alone.

---

*Contributor note: Ant Newman wrote the primary draft. Implementation generalised from the PDA Platform agent-task-planning package by Ant Newman. Lawrence Rowland contributed the original requirements and conceptual design for the confidence extraction and outlier mining capabilities that underpin this module.*
