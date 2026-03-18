# Verified Autonomy MCP Server

> **Status:** Phase two — placeholder scaffold

An MCP (Model Context Protocol) server that exposes trust engineering tools from the Verified Autonomy framework. This allows Claude and other MCP-compatible agents to use verified autonomy techniques directly within an agentic workflow.

## Planned Tools

| Tool | Layer | Description |
|---|---|---|
| `confidence_weight` | 01 | Aggregate confidence scores with inverse weighting |
| `detect_outliers` | 02 | Detect statistical disagreement across model samples |
| `flag_hallucinations` | 03 | Flag outputs below quality threshold |
| `calibrate_confidence` | 04 | Apply conformal prediction to confidence scores |
| `check_guardrails` | 05 | Validate output against deterministic policy rules |
| `check_groundedness` | 06 | Verify RAG output is grounded in source documents |

## Architecture

The MCP server wraps the `verified-autonomy` Python package and exposes each layer's core function as an MCP tool. This means the same code that runs in production also runs inside the agent — there is no separate implementation to maintain.

## Contributing

This is a phase two deliverable. The interface design is in progress. Contributions are welcome once the interface is finalised. See [CONTRIBUTING.md](../CONTRIBUTING.md) for general contribution standards.
