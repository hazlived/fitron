# Changelog

All notable changes to FITRON will be documented in this file.

## [1.1.0] - 2026-11-05

### Added
- **Exact Simplex Projection Optimizer (`pip_model.core.optimizer`)**:
  - Implemented exact $\mathcal{O}(m \log m)$ Euclidean projection onto probability simplex $\Delta^{m-1}$ ($\sum w_j = 1, w_j \ge 0$).
  - Added coordinate descent and projected simplex optimization with strict boundary clamping, replacing unconstrained perturbations.
- **Ruspini Fuzzy Quantile Partition of Unity Completeness (`pip_model.core.fuzzy`)**:
  - Proved and implemented Theorem 2: Ruspini completeness ($\mu_{low}(x) + \mu_{mid}(x) + \mu_{high}(x) \equiv 1.0$) across all features.
  - Added exact indicator preservation for discrete and binary features, resolving quantile knot collapse.
- **Vectorized Bounded TOPSIS & Orientation Inferencing (`pip_model.core.mcdm`)**:
  - Vectorized relative closeness formulation proved to be bounded in $[0, 1]$ (Theorem 3).
  - Added global data-driven Spearman rank correlation to infer criterion polarity (+1 benefit, -1 cost).
- **Adaptive EWMA Memory with Wasserstein-1 Drift Tracking (`pip_model.core.memory`)**:
  - Formulated A-EWMA memory combining exponential criteria weighting with Wasserstein-1 feature distribution drift distance.
  - Added recency decay penalties to prioritize recent reliable operational iterations.
- **Graded Operational Ground-Truth Utility (`pip_model.core.reward`)**:
  - Resolved Theorem 1 (homogeneous positive ranking degeneracy under pure binary labels) via graded operational ground-truth utility $g_i = y_i \cdot \psi(\mathbf{u}_i)$.
  - Added Expected Calibration Error (ECE), Brier score, and graded NDCG calculation.
- **Comprehensive Mathematical Foundations Test Suite (`tests/test_mathematical_foundations.py`)**:
  - 10 rigorous tests validating Theorems 1–4, Ruspini sum-to-one, simplex projection, TOPSIS bounds, and drift detection.

### Changed
- Deprecated unconstrained hill climbing perturbations in favor of exact Euclidean simplex projection.
- Base decision tree classifier updated with balanced class weighting and calibrated Platt/Isotonic probability estimation.
- Package version bumped to `1.1.0`.

## [1.0.3] - 2026-04-26

### Added
- Support for customizable confidence floor fallback and iteration metric tracking.

## [1.0.0] - 2026-04-15

### Added
- Generic support for tabular binary problems beyond the original loan-focused example.
- Automatic threshold tuning based on balanced accuracy and F1.
- Confidence-floor fallback for low-confidence selections.
- Schema validation and explicit target mapping support.
- Iteration metrics export for deployment observability.
- Regression tests covering objective score consistency, threshold tuning, schema validation, and generic string-label targets.

### Changed
- Promoted the package from an early release to the first stable major release.
- Updated the README to describe supported problem types and the PyPI update workflow.
- Updated the example script to use a generic non-loan dataset.

### Fixed
- Removed the practical dependency on a single loan dataset example.
- Reduced deployment fragility from fixed-threshold decisioning.
- Reduced runtime overhead by limiting expensive hyperparameter tuning to the first iteration in the demo flow.
