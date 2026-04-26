# FITRON

[![PyPI version](https://img.shields.io/pypi/v/fitron.svg)](https://pypi.org/project/fitron/)
[![Python](https://img.shields.io/pypi/pyversions/fitron.svg)](https://pypi.org/project/fitron/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Repository](https://img.shields.io/badge/GitHub-hazlived%2Ffitron-black.svg)](https://github.com/hazlived/fitron)

FITRON is a Python library for ranking-first decision intelligence on tabular binary problems.

Instead of stopping at yes/no prediction, FITRON combines classification quality and multi-criteria ranking so you can prioritize records with interpretable, feature-level explanations.

Current release: `1.0.2`

- PyPI: https://pypi.org/project/fitron/
- Repository: https://github.com/hazlived/fitron

## Table of Contents

- [Why FITRON](#why-fitron)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core API](#core-api)
- [Function Reference](#function-reference)
- [Coding Guide](#coding-guide)
- [Common Workflows](#common-workflows)
- [Testing and Development](#testing-and-development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Why FITRON

Many production pipelines output only probabilities or binary labels. FITRON adds a ranking layer designed for operational decision flows where prioritization matters.

FITRON integrates:

1. Fuzzy feature transformation for smoother numeric representation.
2. Decision-tree backbone for robust tabular performance and feature importance.
3. TOPSIS multi-criteria ranking for transparent candidate prioritization.
4. Memory-guided adaptive weights for iterative refinement.

## Key Features

- Ranking-first output with best-candidate selection and scored alternatives.
- Interpretable explanations tied to high-impact transformed features.
- Native support for binary labels as numeric or string values.
- Flexible API: class-based workflow and functional shortcuts.
- Iteration metrics export for reproducibility and analysis.
- Practical deployment safeguards such as confidence floor fallback.

## Installation

### End users

```bash
python -m pip install -U fitron
```

### From source

```bash
git clone https://github.com/hazlived/fitron.git
cd fitron
python -m pip install -U pip setuptools wheel
python -m pip install .
```

### Editable install (development)

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -e .[dev]
```

### Verify installation

```bash
python -c "import fitron; from fitron import FITRONModel; print('fitron import OK')"
```

## Quick Start

Use this sequence for a reliable first run.

1. Install FITRON:

```bash
python -m pip install -U fitron
```

2. Run a minimal end-to-end example:

```python
import pandas as pd
from fitron import FITRONModel

sample = pd.DataFrame(
    {
        "income": [50000, 20000, 75000, 43000, 60000, 47000, 71000, 25000, 55000, 38000, 65000, 30000],
        "risk": [0.2, 0.8, 0.3, 0.5, 0.4, 0.6, 0.25, 0.9, 0.35, 0.7, 0.2, 0.75],
        "credit_score": [710, 520, 760, 640, 700, 650, 750, 500, 720, 580, 740, 550],
        "employment_years": [5, 1, 10, 3, 7, 4, 9, 1, 6, 2, 8, 2],
        "target": [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    }
)

model = FITRONModel(iterations=5, random_state=42)
result = model.fit(sample, target="target")

print("Best candidate index:", result.best_index)
print("Best score:", round(float(result.best_score), 4))
print("Test accuracy:", round(float(result.test_accuracy), 4))
print("\nExplanation:")
for item in result.explanation:
    print(" -", item)
```

3. Optional, from repository root:

```bash
python examples/run_demo.py
```

Expected behavior: prints best candidate index, score, train/test accuracy, and explanation lines.

## Core API

### Class API

#### `FITRONModel`

Main iterative model for fit-rank-explain workflows.

Constructor:

```python
FITRONModel(
    iterations=20,
    random_state=42,
    decision_threshold=0.5,
    objective_classification_weight=0.65,
    confidence_floor=0.55,
)
```

Primary methods:

- `fit(df, target, ...)` -> train and optimize over iterations.
- `rank(df, target, ...)` -> score/rank using current learned state.
- `explain()` -> explanation list from last run.

### Functional API

- `fit(df, target, ...)`
- `rank(df, target, weights=None, memory=None, ...)`
- `explain(result)`
- `update_memory(memory, weights, score, best_idx)`

### Returned result object

`fit()` and `rank()` return `IterationResult`, including:

- predictions and ranking scores
- candidate indices and best index
- objective/classification/ranking quality metrics
- train/test accuracy and threshold metrics
- final weights and explanation list

## Function Reference

Use this table as a fast lookup for what each public entry point does.

| Function / Method | Input | Output | What it does | Use when |
|---|---|---|---|---|
| `FITRONModel(...)` | tuning parameters | model instance | Configures iterative FITRON engine state. | You need reusable model state across multiple runs. |
| `FITRONModel.fit(df, target, ...)` | training dataframe + target | `IterationResult` | Trains and optimizes ranking/classification over iterations. | First pass on a dataset; baseline + learned weights. |
| `FITRONModel.rank(df, target, ...)` | dataframe + target + existing model state | `IterationResult` | Ranks using learned memory/weights with optional retuning. | Re-ranking new candidate batches after initial fit. |
| `FITRONModel.explain()` | none | `list[str]` | Returns explanation entries from the latest model run. | You want quick explanation access from the model instance. |
| `fit(df, target, ...)` | dataframe + target | `IterationResult` | One-shot convenience wrapper around class workflow. | Fast scripting without managing class lifecycle. |
| `rank(df, target, weights=None, memory=None, ...)` | dataframe + target + optional state | `IterationResult` | One-shot ranking call with optional prior state injection. | Stateless pipelines or external state management. |
| `explain(result)` | `IterationResult` | `list[str]` | Extracts explanation strings from a run result. | You already have a result object and need text explanation. |
| `update_memory(memory, weights, score, best_idx)` | memory + optimization stats | `None` | Updates memory history and best-known weight trajectory. | Manual/custom control loops around FITRON logic. |

## Coding Guide

This is the recommended implementation pattern for production-style usage.

1. Define your tabular schema and map the target to binary values.
2. Start with conservative defaults (`iterations=10-20`, `random_state=42`).
3. Run `fit()` once and log metrics for calibration.
4. Use `rank()` for future batches while reusing model state.
5. Persist and monitor threshold/F1/balanced-accuracy trends.
6. Tune `decision_threshold`, `confidence_floor`, and `iterations` only after metric review.

Reference skeleton:

```python
from fitron import FITRONModel

# 1) Create model once
model = FITRONModel(
    iterations=20,
    random_state=42,
    decision_threshold=0.5,
    confidence_floor=0.55,
)

# 2) Initial training + ranking
train_result = model.fit(
    df=train_df,
    target="decision",
    target_map={"reject": 0, "approve": 1},
    metrics_output_path="./iteration_metrics.csv",
)

# 3) Operational ranking on new data
batch_result = model.rank(
    df=batch_df,
    target="decision",
    target_map={"reject": 0, "approve": 1},
)

# 4) Surface result for downstream systems
best_idx = batch_result.best_index
best_score = float(batch_result.best_score)
explanation = batch_result.explanation
```

Production notes:

- Keep `target_map` explicit for string labels to avoid silent mapping mistakes.
- Prefer class API for long-running services so memory/weights evolve consistently.
- Export iteration metrics in CI/staging and compare drift before changing thresholds.

## Common Workflows

### String target mapping

```python
from fitron import fit

result = fit(
    df=data,
    target="decision",
    target_map={"reject": 0, "approve": 1},
    iterations=10,
)
```

### Save per-iteration metrics

```python
from fitron import FITRONModel

model = FITRONModel(iterations=20)
result = model.fit(
    df=data,
    target="decision",
    metrics_output_path="./iteration_metrics.csv",
)
```

### Multi-pass refinement

```python
from fitron import FITRONModel

model = FITRONModel(iterations=5, random_state=42)
first = model.fit(train_data, target="decision")
second = model.rank(new_data, target="decision")
```

## Testing and Development

Run tests:

```bash
pytest -q
```

Release notes: [CHANGELOG.md](CHANGELOG.md)

## Troubleshooting

### `ModuleNotFoundError: No module named 'fitron'`

Cause: package installed in a different interpreter than the one running your command.

Fix:

```bash
python -c "import sys; print(sys.executable)"
python -m pip install -U fitron
```

### NumPy/Pandas build errors on Windows

Cause: non-standard Python distributions may miss compatible wheels and try source builds.

Fix:

- Use standard CPython from python.org.
- Create a fresh virtual environment.
- Install with that venv interpreter directly.

```powershell
python -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install -U pip
.\.venv-win\Scripts\python.exe -m pip install -U fitron
```

### Poor ranking quality or frequent fallback

- Increase dataset size and feature quality.
- Verify `target_map` for string labels.
- Tune `decision_threshold` and `confidence_floor`.
- Increase `iterations` for more adaptive refinement.

## License

MIT
