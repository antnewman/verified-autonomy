# Layer 03: Making Failures Visible — Implementation

## Quick Start

```bash
make install
make run
```

## Expected Output

```
Layer 03: Making Failures Visible
====================================

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

See: layers/03-visible-failures.md
```

## What This Code Does

`compute_quality_score` evaluates a single model-generated sample against three dimensions (coherence, relevance, completeness) and applies an entropy penalty for high-variance outputs.  `flag_potential_hallucinations` uses semantic entropy as a proxy for confabulation risk and returns the IDs of samples above the threshold — not to suppress them, but to surface them explicitly to downstream consumers.  `compute_pass_rate` reports the fraction of samples that cleared the quality bar.

## API

| Symbol | Kind | Description |
|--------|------|-------------|
| `QualityScore` | `dataclass` | `coherence`, `relevance`, `completeness`, `semantic_entropy`, `overall`, `passed_threshold` |
| `Sample` | `dataclass` | `id`, `quality` |
| `compute_quality_score` | `function` | `(coherence, relevance, completeness, semantic_entropy, threshold=0.6) -> QualityScore` |
| `flag_potential_hallucinations` | `function` | `(samples, entropy_threshold=0.7) -> list[str]` — returns sample IDs at hallucination risk |
| `compute_pass_rate` | `function` | `(samples) -> float` — fraction of samples that passed the quality threshold |

## Testing

```bash
make test
```

Runs pytest with `--cov-fail-under=90`.

## Dependencies

No runtime dependencies.  Dev dependencies: `pytest`, `pytest-cov`, `mypy`, `ruff`.
