# Layer 01: Inverse Confidence Weighting — Implementation

## Quick Start

```bash
make install
make run
```

## Expected Output

```
Layer 01: Inverse Confidence Weighting
========================================

Scenario 1 — Uniform confidence (0.8 across three fields):
  Aggregate: 0.8000  (expected ≈ 0.8000)

Scenario 2 — One weak field (category = 0.30):
       date: 0.95
   location: 0.92
     amount: 0.88
   category: 0.30
  Plain mean:              0.7625
  Inverse-weighted result: 0.7228  (lower than mean)

Scenario 3 — Low confidence across all fields:
  Aggregate: 0.1500

Scenario 4 — Empty input:
  Aggregate: 0.0000  (expected 0.0)

See: layers/01-confidence-weighting.md
```

## What This Code Does

`inverse_confidence_weight` takes a mapping of field names to confidence scores (0.0–1.0) and returns a single aggregate score.  Instead of averaging, each field is weighted by `2.0 - confidence`: fields the model is least sure about pull the aggregate down, surfacing the weakest signal rather than obscuring it.  A system 95% confident in nine fields but only 30% confident in one critical field will report a lower aggregate than a plain mean would suggest.

## API

| Function | Signature | Description |
|----------|-----------|-------------|
| `inverse_confidence_weight` | `(field_confidences: dict[str, float]) -> float` | Compute an inverse-weighted aggregate confidence score. Returns `0.0` for an empty dict. Raises `ValueError` if any confidence is outside `[0.0, 1.0]`. |

## Testing

```bash
make test
```

Runs pytest with `--cov-fail-under=90`.

## Dependencies

No runtime dependencies.  Dev dependencies: `pytest`, `pytest-cov`, `mypy`, `ruff`.
