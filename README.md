# FITRON

FITRON is a Python package for hybrid decision intelligence under uncertainty.

Current release: `1.0.0`

The package combines:

- fuzzy feature transformation,
- decision tree learning,
- TOPSIS multi-criteria ranking,
- memory-guided adaptive weight updates,
- interpretable ranking outputs.

Package name on PyPI: `fitron`  
Import path: `fitron`

## What FITRON Supports

FITRON is designed for generic tabular decision problems where the target is binary and the feature space contains a mix of numeric and categorical variables.

It is a good fit when:

- the target can be encoded as 0/1, yes/no, true/false, approved/rejected, or another binary pair,
- the data contains structured rows with feature columns that can be normalized, encoded, and ranked,
- the goal is not only prediction, but also interpretable candidate ranking.

It is not a fit for:

- multi-class targets without adaptation,
- unstructured text or image-only problems,
- tasks where row-level ranking is not meaningful.

## Why FITRON

Most pipelines stop at classification. FITRON extends that flow into adaptive decision ranking.

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
pip install fitron
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

More detailed setup for Linux/macOS:

1. Confirm Python 3.10+ is available:

```bash
python3 --version
```

2. If `venv` is missing on Linux, install it first (example for Debian/Ubuntu):

```bash
sudo apt update
sudo apt install -y python3-venv
```

3. Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Upgrade packaging tools and install FITRON in editable mode:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -e .[dev]
```

5. Verify installation:

```bash
python -c "from fitron import FITRONModel; print('ok')"
pytest -q
```

Notes:

- On some macOS setups, `python3` may be installed via Homebrew and located under `/opt/homebrew/bin/python3` (Apple Silicon) or `/usr/local/bin/python3` (Intel).
- Use `deactivate` when you are done working in the virtual environment.

## Quick Start

```python
import pandas as pd
from fitron import FITRONModel, fit

sample = pd.DataFrame(
    {
        "income": [50000, 20000, 75000, 43000],
        "risk": [0.2, 0.8, 0.3, 0.5],
        "credit_score": [710, 520, 760, 640],
        "target": [1, 0, 1, 1],
    }
)

model = FITRONModel(iterations=5, random_state=42)
result = model.fit(sample, target="target")

print("Best index:", result.best_index)
print("Best score:", result.best_score)
print("Explanation:", result.explanation)
```

You can also pass binary string targets and explicit label mappings:

```python
result = fit(
    df,
    target="status",
    target_map={"reject": 0, "approve": 1},
)
```

## Public API

- `FITRONModel`
- `fit(df, target, ...)`
- `rank(df, target, ...)`
- `explain(result)`
- `update_memory(memory, weights, score, best_idx)`

## Run Tests

```bash
pytest -q
```

## Build and Publish

Release notes and version history are tracked in [CHANGELOG.md](CHANGELOG.md).

### Update flow for PyPI

1. Update the version in [pyproject.toml](pyproject.toml).
2. Update [CHANGELOG.md](CHANGELOG.md) with the release summary.
3. Run the test suite:

```bash
pytest -q
```

4. Build the package:

```bash
python -m build
```

5. Verify the generated distributions:

```bash
python -m twine check dist/*
```

6. Upload to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

7. Confirm the package installs from TestPyPI, then upload to the real PyPI index:

```bash
python -m twine upload dist/*
```

8. Tag the release in git so the published version is easy to trace:

```bash
git tag v1.0.0
git push origin v1.0.0
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
