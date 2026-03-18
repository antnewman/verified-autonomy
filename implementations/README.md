# Implementations

Each layer of the Verified Autonomy field guide has a self-contained, runnable code implementation.

## Design Principle

Every implementation runs in isolation. No implementation depends on any other layer's code. A reader should be able to go from the field guide to a working terminal in under two minutes.

## Quick Start

Pick any layer and run it:

```bash
cd layer_01_confidence_weighting
make install
make run
```

## The Makefile Contract

Every implementation directory has a Makefile with these targets:

| Target | What it does |
|---|---|
| `make install` | Install dependencies into a virtual environment |
| `make test` | Run the test suite with coverage reporting (90% minimum) |
| `make run` | Run the example and print expected output |
| `make lint` | Run type checking and linting |
| `make clean` | Remove virtual environment and build artefacts |

## Installing as a Package

All nine implementations are also available as a single PyPI package:

```bash
pip install verified-autonomy
```

```python
from verified_autonomy.confidence_weighting import inverse_confidence_weight
from verified_autonomy.outlier_escalation import detect_outliers
```

## Code Ancestry

Layers 01, 02, and 03 are generalised from the `agent-task-planning` package in the [PDA Platform](https://github.com/antnewman/pda-platform). The original implementations are domain-specific (project management data); the implementations here are domain-agnostic, designed for use in any AI system.

## Contributing Code

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full code standards, including the 90% coverage floor, edge and corner case mandate, failure mode test requirements, and the runnable isolation requirement.
