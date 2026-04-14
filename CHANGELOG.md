# Changelog

All notable changes to FITRON will be documented in this file.

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
