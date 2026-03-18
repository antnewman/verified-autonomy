# ADR 007: External Dependencies for Layer 04

## Status

Accepted

## Context

Layers 01–03 carry zero runtime dependencies beyond the Python standard library. This was a deliberate choice: the techniques those layers demonstrate (confidence weighting, IQR-based outlier detection, visible failures) can be implemented cleanly with built-in data structures and the `statistics` module. Zero-dependency implementations have the lowest possible barrier to adoption.

Layer 04 demonstrates calibration and conformal prediction. Two of its three techniques require external libraries:

- **Temperature scaling** requires numerical optimisation: finding the scalar T that minimises negative log-likelihood over a calibration set. This is a bounded 1-D minimisation problem. The standard tool in the Python ecosystem is `scipy.optimize.minimize_scalar`. Reimplementing a robust bounded 1-D minimiser from scratch (e.g. Brent's method) would add several hundred lines of numerical code with no educational value — the technique being demonstrated is temperature scaling, not root-finding.

- **The demonstration and integration tests** require a classifier and synthetic dataset. `GradientBoostingClassifier` and `make_classification` from `scikit-learn` provide a realistic overconfident model and a reproducible five-class dataset. These are already available in any ML environment where this guide is relevant.

- **Array operations** throughout — ECE computation, conformity scores, quantiles — require efficient array arithmetic. `numpy` is a transitive dependency of `scikit-learn` and is effectively universal in ML environments.

The conformal prediction core (calibrate\_conformal, conformal\_predict, evaluate\_coverage) is implemented from scratch with no CP library (no MAPIE, no nonconformist). This keeps the mechanism fully transparent, consistent with the field guide's pedagogical goal.

## Decision

Allow `numpy`, `scikit-learn`, and `scipy` as runtime dependencies for Layer 04. Do not introduce any conformal prediction library; implement the algorithm directly.

## Consequences

**Benefits:**
- The layer produces a realistic, convincing demonstration: actual overconfidence (ECE = 0.077), meaningful temperature reduction (T = 2.27), and a formal coverage guarantee verified on held-out data (91.6% ≥ 90%).
- Temperature optimisation is numerically robust via scipy's Brent-method implementation, without adding educational noise.
- The conformal prediction code is self-contained and readable — the guarantee is visible in the source, not hidden behind a library abstraction.

**Trade-offs:**
- Layer 04 is the first layer to require `pip install`. The `make install` step handles this automatically, and the dependencies are universally available in ML environments. The two-minute goal from ADR 006 is preserved.
- `scikit-learn` and `scipy` are heavier packages than the standard library. This is unavoidable for a realistic demonstration; toy implementations would undermine the field guide's credibility.

**Alternatives considered:**
- Reimplementing Brent's method from scratch — rejected. Adds complexity with no pedagogical value; scipy's implementation is battle-tested.
- Using a conformal prediction library (MAPIE) — rejected. The algorithm is simple enough to implement from scratch (seven lines for the threshold computation), and transparency is more valuable than convenience here.
- Using `statsmodels` for temperature optimisation — rejected. `scipy` is already required and is the lower-level, more appropriate tool for a single bounded scalar minimisation.
