# TRACE

TRACE (Tree-based Ranking with Adaptive Criteria Evolution) is a Python package for hybrid decision intelligence under uncertainty.

The package combines:

- fuzzy feature transformation,
- decision tree learning,
- TOPSIS multi-criteria ranking,
- memory-guided adaptive weight updates,
- interpretable ranking outputs.

Package name on PyPI: `trace`  
Import path: `pip_model`

## Why TRACE

Most pipelines stop at classification. TRACE extends that flow into adaptive decision ranking.

1. Learns a predictive backbone with a decision tree.
2. Ranks valid candidates with TOPSIS and learned criteria weights.
3. Adapts weights over iterations using memory and exploration.
4. Produces feature-level explanations for selected options.

## Core Workflow

1. Preprocess tabular data and encode features.
2. Fuzzify numeric features into low/mid/high memberships.
3. Train decision tree model and estimate feature importances.
4. Filter valid candidates using model predictions.
5. Rank candidates with TOPSIS.
6. Update weights using adaptive exploration and memory feedback.
7. Return best option, score, and explanation.

## Installation

### End users

```bash
pip install trace
```

### From source

```bash
pip install .
```

### Editable install (development)

```bash
pip install -e .[dev]
```

## Environment Setup

Use a standard CPython virtual environment.

Windows:

```powershell
python -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install -U pip setuptools wheel
.\.venv-win\Scripts\python.exe -m pip install -e .[dev]
```

If PowerShell blocks script activation, use:

```cmd
.venv-win\Scripts\activate.bat
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .[dev]
```

## Quick Start

```python
import pandas as pd
from pip_model import TRACEModel

sample = pd.DataFrame(
    {
        "income": [50000, 20000, 75000, 43000],
        "risk": [0.2, 0.8, 0.3, 0.5],
        "credit_score": [710, 520, 760, 640],
        "target": [1, 0, 1, 1],
    }
)

model = TRACEModel(iterations=5, random_state=42)
result = model.fit(sample, target="target")

print("Best index:", result.best_index)
print("Best score:", result.best_score)
print("Explanation:", result.explanation)
```

## Public API

- `TRACEModel`
- `fit(df, target, ...)`
- `rank(df, target, ...)`
- `explain(result)`
- `update_memory(memory, weights, score, best_idx)`

## Run Tests

```bash
pytest -q
```

## Build and Publish

Build distribution artifacts:

```bash
python -m build
```

Validate artifacts:

```bash
python -m twine check dist/*
```

Upload:

```bash
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```

## Project Structure

```text
src/pip_model/
  core/
  api.py
  pipeline.py
tests/
examples/
pyproject.toml
```

## Authors

- zibransheikh
- hazlived

## License

MIT
