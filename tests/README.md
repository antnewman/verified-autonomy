# Testing Standards

## Running Tests

Each implementation has its own test suite. From any implementation directory:

```bash
make test
```

This runs pytest with coverage reporting and enforces the 90% minimum coverage threshold.

## Test Quality Requirements

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full test quality mandate, including edge case, corner case, and failure mode requirements.

## Root Package Tests

To run all tests for the `verified-autonomy` PyPI package:

```bash
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=90
```
