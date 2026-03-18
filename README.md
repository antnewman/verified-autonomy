# Verified Autonomy

**A Field Guide to Engineering Trust in AI Systems**

*Trust is built not by making the system more confident, but by making it more honest about where it is not.*

---

## What This Is

An open source field guide structured around a nine-layer defence-in-depth architecture for trust in AI systems. Each layer addresses a specific failure mode. Together they form a comprehensive framework. Written by practitioners, anchored in production code and verified case studies.

---

## The Nine Layers

| Layer | Title | Subtitle |
|---|---|---|
| 01 | Inverse Confidence Weighting | Surface your weakest signal, not your average |
| 02 | Outlier Detection as Hard Escalation | Statistical disagreement is a signal, not noise |
| 03 | Making Failures Visible | The potential hallucinations list |
| 04 | Calibration and Conformal Prediction | The only confidence score with a mathematical guarantee |
| 05 | Deterministic Guardrails | The constitutional court that holds final power |
| 06 | Retrieval-Augmented Generation as Explainability | Shift trust from model intent to source facticity |
| 07 | Adversarial Testing | Find where it breaks before someone else does |
| 08 | Cryptographic Audit Trails | Proving what the system did |
| 09 | Formal Verification | The gold standard nobody can fully use yet |

---

## The Central Argument

Trust is not a feature. It is an emergent property of a defence-in-depth architecture where each layer compensates for the failure modes of the others.

---

## Quick Start — Run Any Layer's Code Example

```bash
cd implementations/layer_01_confidence_weighting
make run
```

---

## Install as a Python Package

```bash
pip install verified-autonomy
```

```python
from verified_autonomy.confidence_weighting import inverse_confidence_weight
```

---

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md). The guide is designed for community contribution — every layer can be improved with better production examples, better code, or better limitations analysis.

---

## Ownership and Transfer Intent

This repository is initially hosted on Ant Newman's personal GitHub account. It is the explicit intent to transfer ownership to a neutral third-party organisation (e.g. github.com/verified-autonomy) as the project matures and community adoption warrants it. This is stated prominently so contributors know from day one that this will not become personal IP.

---

## Licence

Dual-licensed:

- Written content: [CC BY 4.0](LICENSE-CONTENT)
- Code: [MIT](LICENSE-CODE)

---

## Authors

**Ant Newman** and **Shanti Greene**. Technical reviewer: **Malia Hosseini**. Originated from the *Not Another AI* podcast episode on AI trust (Austin, Texas, 2026). Piyanka Jain credited in acknowledgements.

---

## Citation

```
Newman, A. & Greene, S. (2026). Verified Autonomy: A Field Guide to Engineering Trust
in AI Systems. https://github.com/antnewman/verified-autonomy
```

---

## Versioning

This project uses semantic versioning. See [CHANGELOG.md](CHANGELOG.md) for release history.

---

## Links

- [Hosted site](https://verified-autonomy.netlify.app)
- [Field guide layers](layers/)
- [Runnable implementations](implementations/)
- [Case studies](case-studies/)
