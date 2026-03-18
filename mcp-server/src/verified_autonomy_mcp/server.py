"""
Verified Autonomy MCP Server — entry point.

Planned architecture:
    This server exposes the core functions of the verified-autonomy Python
    package as MCP tools, allowing Claude and other MCP-compatible agents to
    apply trust engineering techniques directly within an agentic workflow.

    Each tool is a thin adapter: it validates inputs, calls the corresponding
    verified_autonomy module function, and returns a structured result. All
    business logic lives in the package — this file is the MCP interface only.

Planned tools:
    confidence_weight     — Layer 01: inverse confidence weighting
    detect_outliers       — Layer 02: IQR-based outlier escalation
    flag_hallucinations   — Layer 03: potential hallucinations list
    calibrate_confidence  — Layer 04: conformal prediction
    check_guardrails      — Layer 05: deterministic policy guardrails
    check_groundedness    — Layer 06: RAG source grounding check
"""

# TODO: Implement MCP server using the MCP Python SDK
# See: https://github.com/modelcontextprotocol/python-sdk
# Each tool should wrap the corresponding verified_autonomy module function.
