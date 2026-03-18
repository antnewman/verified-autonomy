# Contributing to Verified Autonomy

This document is written with sufficient precision and structure that an AI coding assistant can read it and produce a conformant contribution without ambiguity. Please read it in full before opening a pull request.

---

## 1. Welcome and Scope

This repo is specifically about trust engineering techniques for AI systems in production. Each contribution must address a specific, nameable failure mode in a real production AI system.

**Out of scope:**
- AI ethics philosophy
- AI regulation policy
- Model architecture

...unless directly relevant to a trust engineering technique. This is not a survey, not a benchmark, and not a vendor comparison. If your contribution does not address a specific, verifiable failure mode, it is likely out of scope.

---

## 2. Development Environment Setup

Step-by-step. No assumed knowledge.

**Prerequisites:**
- Python 3.11 or newer — [python.org/downloads](https://www.python.org/downloads/)
- [uv](https://github.com/astral-sh/uv) — fast Python package and project manager

**Steps:**

```bash
# 1. Clone the repository
git clone https://github.com/antnewman/verified-autonomy.git
cd verified-autonomy

# 2. Check out the dev branch — never work on main
git checkout dev

# 3. Install the root package with dev dependencies
uv pip install -e ".[dev]"
```

**Running a single layer implementation:**

Each implementation is entirely self-contained. No other layer's code is required.

```bash
cd implementations/layer_01_confidence_weighting
make install
make test
```

Run any other layer the same way — replace `layer_01_confidence_weighting` with the relevant directory name.

---

## 3. Branch Naming

| Pattern | Purpose | Example |
|---|---|---|
| `feat/[descriptive-name]` | New features or content | `feat/layer-01-confidence-weighting` |
| `fix/[descriptive-name]` | Bug fixes or corrections | `fix/layer-04-coverage-formula` |
| `docs/[descriptive-name]` | Documentation changes | `docs/update-contributing-guide` |

**Rules:**
- All branches are cut from `dev`, never from `main`
- All merges go to `dev`, never directly to `main`

---

## 4. Pull Request Requirements

Every pull request must include:

1. A description of what changed and why
2. Which layer(s) or section(s) are affected
3. How it was tested (for code) or reviewed (for content)
4. Any limitations or known issues
5. Confirmation that the contribution meets the standards in this document

Use the pull request template at `.github/pull_request_template.md`. Pull requests that do not use the template will be returned with a request to resubmit.

---

## 5. Content Standards — The Section Template

Every layer section must follow this template exactly. Deviations will be rejected.

| Field | Requirement |
|---|---|
| **What it is** | Plain language description. No assumed knowledge. |
| **Why it matters** | The specific failure mode this technique addresses. Must be grounded in a real, named example. No hypotheticals. |
| **Production example** | At least one concrete, verifiable, real-world implementation. Open source code with file and line number references preferred. Named case studies acceptable. No hypotheticals. |
| **Limitations** | Honest about failure modes and when not to use the technique. A limitations section that only describes minor inconveniences is not acceptable. Every technique has real failure modes — name them. |
| **Reader takeaway** | One or two sentences. What should the reader be able to do or think differently after reading this section? |

### Content Evidence Requirements

- Every claim must be verifiable. If a statistic is cited, it must be traceable to a named source.
- Every production example must be real. Named organisations, named studies, named tools.
- If the example is from your own deployment and cannot be publicly named, describe it with enough specificity to be credible (e.g. "a global financial institution processing over one million daily transactions").
- The guide must be able to stand up to expert review. Before submitting, ask: would a senior ML engineer or production AI practitioner find anything here that is oversimplified, misleading, or incorrect? If yes, fix it before opening the pull request.

---

## 6. Writing for Video

Every layer section should be written with a clear walkthrough narrative arc:

**problem → technique → code/example → limitation**

This structure will serve as the natural script for the companion video series (phase two). When writing, ask yourself: could someone walk through this section on camera with the code or case study visible and have it flow naturally as a 2–3 minute explanation?

If the section does not have this narrative flow, it needs restructuring before merge. A section that reads as a reference document rather than a guided explanation will be returned for revision.

---

## 7. Code Standards

### Runnable Isolation Requirement

Every layer's code example must run with a single command from its own directory, with no dependency on any other layer's code. The reader must be able to go from the field guide to a working terminal in under two minutes.

Each implementation directory must have:
- Its own `pyproject.toml` with minimal, self-contained dependencies
- Its own `Makefile` (see contract below)
- Its own `README.md` with expected output documented

This is non-negotiable. A code contribution that requires navigating the wider repository to run will not be merged.

### The Makefile Contract

Every implementation directory must have a `Makefile` with exactly these targets:

| Target | What it does |
|---|---|
| `make install` | Install dependencies into a virtual environment |
| `make test` | Run the test suite with coverage reporting (90% minimum) |
| `make run` | Run the example and print expected output |
| `make lint` | Run type checking and linting |
| `make clean` | Remove virtual environment and build artefacts |

### Test Coverage

**90% is the floor, not the target.** Coverage is measured per file and overall. A pull request that drops coverage below 90% will not be merged.

### Test Quality Requirements (All Mandatory)

**Edge cases must be explicitly tested.** Inputs at the boundary of the expected range: minimum value, maximum value, empty input, single-element input. Every function that operates on a range must have tests at both boundaries.

**Corner cases must be explicitly tested.** Inputs at the intersection of multiple boundary conditions simultaneously — for example, an empty list with a zero confidence threshold. Corner cases must be named, documented, and tested.

**Happy path tests are necessary but not sufficient.** A test suite that only tests correct inputs is a demonstration, not a test suite.

**Failure mode tests are required.** Every function that can fail must have tests that verify it fails correctly: the right exception type, the right error message, the right fallback behaviour.

**Test names must be descriptive.** The test name must describe the scenario, not the function.

- Not acceptable: `test_confidence_score`
- Acceptable: `test_confidence_score_returns_zero_when_all_samples_diverge`

**No mocking the system under test.** Mocking external dependencies (APIs, file systems, databases) is acceptable. Mocking the function or class being tested is not.

### Code Quality Requirements

- All code must be typed — Python type hints throughout, including return types
- All public functions must have docstrings describing parameters, return values, and raised exceptions
- No commented-out code in merged pull requests
- No print statements or debug logging left in merged code
- All imports must be used
- Functions must do one thing — if a function does two things, it should be two functions

### The Production Standard

This is not academic code. The Verified Autonomy field guide claims to describe production-grade trust engineering. The code must meet that standard. If a contribution would not pass a production code review at a senior engineer level, it should not be merged.

---

## 8. Review Process

| Merge direction | Required approvals |
|---|---|
| `feat → dev` | At least one core team member |
| `dev → main` | All three core team members: Ant Newman, Shanti Greene, Malia Hosseini |

- No force pushes to `dev` or `main` under any circumstances
- Pull requests will be reviewed within five working days. If a reviewer cannot meet that timeline, they comment on the pull request to set expectations.
- Malia Hosseini (technical reviewer) will reject non-conformant contributions with written reasons. Contributors are expected to address the feedback and resubmit.

---

## 9. How to Raise an Issue

Issue templates are provided for four contribution types. Use the appropriate template — issues without sufficient detail will be closed with a request to resubmit.

| Template | When to use |
|---|---|
| [Content correction](.github/ISSUE_TEMPLATE/content-correction.md) | An inaccuracy, oversimplification, or misleading claim |
| [New layer proposal](.github/ISSUE_TEMPLATE/new-layer-proposal.md) | A proposed new layer or trust engineering technique |
| [Code contribution](.github/ISSUE_TEMPLATE/code-contribution.md) | A new or improved implementation |
| [Case study contribution](.github/ISSUE_TEMPLATE/case-study-contribution.md) | A real-world production case study |

---

## 10. Attribution

All contributors who have a pull request merged into `main` will be listed in [CONTRIBUTORS.md](CONTRIBUTORS.md). Each layer section carries a contributor note identifying who wrote the primary draft and who provided the production examples.
