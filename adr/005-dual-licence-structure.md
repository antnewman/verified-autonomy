# ADR 005: Dual Licence Structure

## Status

Accepted

## Context

The project contains two distinct types of artefact: written content (field guide layers, case studies, ADRs, synthesis) and executable code (layer implementations, MCP server). A single licence does not serve both well.

CC BY 4.0 is the standard for open knowledge — it maximises reuse while requiring attribution. MIT is the standard for open source code — it permits commercial use and is compatible with the widest range of downstream licences. Applying CC BY 4.0 to code creates ambiguity in commercial deployments; applying MIT to written content weakens the attribution requirement that gives the guide its provenance.

## Decision

CC BY 4.0 for all written content (field guide layers, case studies, ADRs, synthesis, README, CONTRIBUTING). MIT for all code (implementations, MCP server, root package). Licence files are clearly named — `LICENSE-CONTENT` and `LICENSE-CODE` — and the README explains the scope of each.

## Consequences

**Benefits:**
- Contributors and users know exactly what they can do with each type of artefact without ambiguity.
- Written content retains attribution requirements appropriate for a citable reference.
- Code is compatible with commercial use, removing barriers to production adoption.

**Trade-offs:**
- Dual licences add marginal complexity to the contribution process. Contributors must understand which licence applies to their contribution type.

**Alternatives considered:**
- Single MIT licence — does not adequately protect attribution for written content that will be cited and reproduced.
- Single CC BY 4.0 licence — creates ambiguity for commercial code reuse and is not conventional for software packages.
- Apache 2.0 for code — more complex than MIT with no meaningful additional benefit for this project's scope.
