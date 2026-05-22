# Layer 02: Outlier Detection as Hard Escalation — Implementation

## Quick Start

```bash
make install
make run
```

## Expected Output

```
Layer 02: Outlier Detection as Hard Escalation
================================================

Scenario 1 — Clean data, high confidence:
  Values:     [10.2, 10.5, 10.1, 10.4, 10.3]
  Consensus:  10.30
  Escalation: none
  Outliers:   0

Scenario 2 — One outlier, otherwise high confidence:
  agent_A: 10.2
  agent_B: 10.5
  agent_C: 10.1
  agent_D: 10.4
  agent_E: 80.0 ← OUTLIER
  Consensus:  10.30
  Escalation: expert_required  (outlier alone triggers EXPERT_REQUIRED)

Scenario 3 — Clean data, very low confidence (0.35):
  Escalation: expert_required  (conf < 0.4 triggers EXPERT_REQUIRED)

Scenario 4 — Moderate confidence (0.55):
  Escalation: detailed_review

See: layers/02-outlier-escalation.md
```

## What This Code Does

`detect_outliers` applies Tukey's IQR fence method to a list of numeric values: any value outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` is flagged as an `OutlierFlag` with its divergence score and a plain-English reason.  The function then routes to one of four escalation tiers using an **OR condition**: outliers *or* low confidence each independently trigger escalation, so the system fails safe.

## API

| Symbol | Kind | Description |
|--------|------|-------------|
| `EscalationLevel` | `Enum` | `NONE`, `SPOT_CHECK`, `DETAILED_REVIEW`, `EXPERT_REQUIRED` |
| `OutlierFlag` | `dataclass` | `label`, `value`, `consensus`, `divergence`, `reason` |
| `DetectionResult` | `dataclass` | `escalation_level`, `consensus`, `confidence`, `outliers`, `has_outliers` |
| `detect_outliers` | `function` | `(values, labels=None, confidence=1.0) -> DetectionResult` |

## Testing

```bash
make test
```

Runs pytest with `--cov-fail-under=90`.

## Dependencies

No runtime dependencies.  Dev dependencies: `pytest`, `pytest-cov`, `mypy`, `ruff`.

> **Note:** `scipy` and `numpy` were considered but are not required — the IQR computation is implemented directly to keep the module self-contained.
