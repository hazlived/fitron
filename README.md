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

Here's a complete working example with sufficient data to train the model effectively:

```python
import pandas as pd
from fitron import FITRONModel

# Create sample data with sufficient rows for training
sample = pd.DataFrame(
    {
        "income": [50000, 20000, 75000, 43000, 60000, 47000, 71000, 25000, 55000, 38000, 65000, 30000],
        "risk": [0.2, 0.8, 0.3, 0.5, 0.4, 0.6, 0.25, 0.9, 0.35, 0.7, 0.2, 0.75],
        "credit_score": [710, 520, 760, 640, 700, 650, 750, 500, 720, 580, 740, 550],
        "employment_years": [5, 1, 10, 3, 7, 4, 9, 1, 6, 2, 8, 2],
        "target": [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    }
)

# Initialize and fit the model
model = FITRONModel(iterations=5, random_state=42)
result = model.fit(sample, target="target")

# Access results
print("Best candidate index:", result.best_index)
print("Best score:", result.best_score)
print("Test accuracy:", result.test_accuracy)
print("\nExplanation for best candidate:")
for explanation in result.explanation:
    print(" -", explanation)
```

**Output:**
```
Best candidate index: 4
Best score: 0.87
Test accuracy: 0.83

Explanation for best candidate:
 - credit_score: 0.7341 (importance: 0.3872)
 - income: 0.6523 (importance: 0.2914)
 - employment_years: 0.5890 (importance: 0.1632)
 - risk: 0.4241 (importance: 0.1582)
```

### Using String Targets

FITRON automatically handles string labels through explicit mapping:

```python
data = pd.DataFrame(
    {
        "feature_a": [10, 20, 30, 40, 50, 60, 70, 80],
        "feature_b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "decision": ["reject", "approve", "approve", "reject", "approve", "approve", "reject", "approve"],
    }
)

from fitron import fit

result = fit(
    data,
    target="decision",
    target_map={"reject": 0, "approve": 1},
    iterations=10,
)

print("Best index:", result.best_index)
print("Explanation:", result.explanation)
```

---

## API Reference

### Core Classes

#### `FITRONModel`

Main class for building and iterating over decision-ranking workflows. Maintains memory and weights across multiple iterations.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iterations` | `int` | 20 | Number of iterations to run during `fit()`. Each iteration refines weights and finds better candidates. |
| `random_state` | `int` | 42 | Random seed for reproducibility. Affects train/test splits, fuzzy profile fitting, and decision tree training. |
| `decision_threshold` | `float` | 0.5 | Probability threshold for binary classification. Candidates above this are preferred. |
| `objective_classification_weight` | `float` | 0.65 | Weight balancing classification quality (0.65) vs ranking quality (0.35) in the objective score. Range: [0, 1]. |
| `confidence_floor` | `float` | 0.55 | Minimum confidence score for a candidate to be selected. If no candidate meets this, falls back to highest probability. |

**Methods:**

##### `fit(df, target, ...)`

Train the model on data and perform multi-iteration optimization.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | **Required** | Input data containing features and target column. Must have at least 5 rows for reliable results. |
| `target` | `str` | **Required** | Name of the binary target column. Values can be numeric (0/1) or strings (mapped via `target_map`). |
| `iterations` | `int` | 20 | Override the constructor's iteration count (optional). |
| `drop_columns` | `list[str]` \| None | None | Feature columns to exclude from processing. Useful for removing IDs, timestamps, etc. |
| `criterion_types` | `list[str]` \| None | None | MCDM criterion type for each feature: `"benefit"` (higher is better) or `"cost"` (lower is better). If None, all treated as benefits. |
| `expected_feature_columns` | `list[str]` \| None | None | Explicit list of feature columns to use. If None, all columns except target are used. |
| `target_map` | `dict[str, int]` \| None | None | Mapping for non-numeric target values. Example: `{"reject": 0, "approve": 1}`. |
| `metrics_output_path` | `str` \| None | None | Path to save iteration metrics as CSV. If provided, creates iteration_metrics.csv with scores and accuracies. |

**Returns:** `IterationResult` object containing predictions, rankings, best candidate, and explanations.

**Example:**

```python
result = model.fit(
    df=loan_data,
    target="approved",
    iterations=20,
    target_map={"denied": 0, "approved": 1},
    metrics_output_path="./metrics.csv",
)
```

##### `rank(df, target, ...)`

Rank candidates using previously fitted weights and memory without retraining from scratch.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | **Required** | Input data to rank. Must have same features as training data. |
| `target` | `str` | **Required** | Name of the binary target column. |
| `drop_columns` | `list[str]` \| None | None | Columns to exclude. |
| `criterion_types` | `list[str]` \| None | None | MCDM criterion types for each feature. |
| `expected_feature_columns` | `list[str]` \| None | None | Explicit feature columns to use. |
| `target_map` | `dict[str, int]` \| None | None | Mapping for non-numeric target values. |
| `tune_hyperparameters` | `bool` | False | If True, retunes decision tree hyperparameters (slower, more accurate). |

**Returns:** `IterationResult` object.

**Example:**

```python
# After fit(), use same model to rank new data
new_candidates = pd.DataFrame({...})
rank_result = model.rank(
    df=new_candidates,
    target="approved",
    target_map={"denied": 0, "approved": 1},
)
print("Ranked order:", rank_result.candidate_indices)
print("Best choice index:", rank_result.best_index)
```

---

### API Functions

#### `fit(df, target, ...)`

Convenience function wrapper around `FITRONModel.fit()`. Creates a model internally.

**Parameters:** Same as `FITRONModel.fit()` plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | `int` | 42 | Random seed. |
| `decision_threshold` | `float` | 0.5 | Classification threshold. |
| `objective_classification_weight` | `float` | 0.65 | Classification weight in objective. |
| `confidence_floor` | `float` | 0.55 | Minimum confidence for selection. |

**Returns:** `IterationResult`

**Example:**

```python
from fitron import fit

result = fit(
    df=data,
    target="approved",
    iterations=10,
    target_map={"no": 0, "yes": 1},
)
```

---

#### `rank(df, target, weights=None, memory=None, ...)`

Convenience function to rank candidates without creating a model instance.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | **Required** | Candidates to rank. |
| `target` | `str` | **Required** | Binary target column name. |
| `weights` | `np.ndarray` \| None | None | Feature weights from a previous fit. If None, weights are initialized from feature importances. |
| `memory` | `Memory` \| None | None | Memory object from a previous fit to blend learned patterns. |
| `drop_columns` | `list[str]` \| None | None | Columns to exclude. |
| `criterion_types` | `list[str]` \| None | None | MCDM criterion types. |
| `expected_feature_columns` | `list[str]` \| None | None | Explicit features to use. |
| `tune_hyperparameters` | `bool` | False | Retune decision tree hyperparameters. |
| `target_map` | `dict[str, int]` \| None | None | Target value mapping. |

**Returns:** `IterationResult`

**Example:**

```python
from fitron import rank

result = rank(
    df=candidates,
    target="approved",
    weights=previous_weights,
    memory=previous_memory,
    target_map={"rejected": 0, "approved": 1},
)
```

---

#### `explain(result)`

Extract explanation strings from an `IterationResult`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `result` | `IterationResult` | Result object from `fit()` or `rank()`. |

**Returns:** `list[str]` - Feature explanations in importance order.

**Example:**

```python
from fitron import explain

explanations = explain(result)
print("Top features explaining the selection:")
for exp in explanations:
    print(f"  • {exp}")
```

---

#### `update_memory(memory, weights, score, best_idx)`

Manually update a memory object with new weights and scores.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory` | `Memory` | Memory object to update. |
| `weights` | `np.ndarray` | Feature weights to store. |
| `score` | `float` | Objective score achieved. |
| `best_idx` | `int` | Index of the best candidate selected. |

**Returns:** `None` (modifies memory in place)

**Example:**

```python
from fitron import update_memory, Memory
import numpy as np

memory = Memory()
weights = np.array([0.2, 0.3, 0.25, 0.25])
update_memory(memory, weights, score=0.85, best_idx=5)
```

---

### Data Classes

#### `IterationResult`

Dataclass containing all outputs from a single `fit()` or `rank()` call.

| Attribute | Type | Description |
|-----------|------|-------------|
| `predictions` | `np.ndarray` | Binary predictions (0 or 1) for each row in test set. |
| `scores` | `np.ndarray` | Final ranking scores for candidate rows. Higher = better. |
| `candidate_indices` | `list[int]` | Indices of rows selected as valid candidates. |
| `best_index` | `int` | Index of the highest-scoring candidate (best choice). |
| `best_score` | `float` | Final ranking score of the best candidate. |
| `objective_score` | `float` | Composite objective score across all iterations. |
| `weights` | `np.ndarray` | Final optimized feature weights used for ranking. |
| `train_accuracy` | `float` | Accuracy of decision tree on training set. |
| `test_accuracy` | `float` | Accuracy of decision tree on test set. |
| `classification_quality` | `float` | Quality metric for classification (0 to 1). |
| `ranking_quality` | `float` | Quality metric for ranking (0 to 1). |
| `threshold_balanced_accuracy` | `float` | Balanced accuracy at decision threshold. |
| `threshold_f1` | `float` | F1 score at decision threshold. |
| `fallback_triggered` | `bool` | True if fallback to highest probability was used (confidence below floor). |
| `model` | `object` | Underlying decision tree model (sklearn DecisionTreeClassifier). |
| `explanation` | `list[str]` | Top-N feature explanations for best candidate. |

**Example:**

```python
result = model.fit(data, target="approved")

print(f"Best candidate: {result.best_index}")
print(f"Score: {result.best_score:.4f}")
print(f"Test accuracy: {result.test_accuracy:.4f}")
print(f"Fallback used: {result.fallback_triggered}")
print(f"Ranking scores: {result.scores}")
```

---

#### `Memory`

Dataclass for tracking optimization history and best weights across iterations.

| Attribute | Type | Description |
|-----------|------|-------------|
| `history` | `list[dict]` | Log of all updates. Each entry contains: `weights`, `score`, `improvement`, `ema_improvement`, `best_idx`. |
| `best_weights` | `np.ndarray` \| None | Feature weights that achieved the best score. |
| `best_score` | `float` | The highest objective score achieved. |
| `max_history` | `int` | Maximum entries to keep in history (default: 200). Older entries are removed. |
| `ema_improvement` | `float` | Exponential moving average of score improvements. |

**Methods:**

##### `update(weights, score, best_idx)`

Record new weights and score in memory.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `weights` | `np.ndarray` | Feature weights to store. |
| `score` | `float` | Objective score. |
| `best_idx` | `int` | Index of selected candidate. |

**Example:**

```python
from fitron import Memory

memory = Memory()
memory.update(weights=np.array([0.25, 0.35, 0.2, 0.2]), score=0.82, best_idx=3)

# Later, retrieve best weights
best_w = memory.get_best()
print(f"Best weights: {best_w}")
print(f"Best score: {memory.best_score}")
```

##### `get_best()`

Retrieve the feature weights that achieved the best score.

**Returns:** `np.ndarray | None` - Best weights or None if memory is empty.

---

## Common Workflows

### Workflow 1: Single-Pass Ranking

Train once and get the best candidate:

```python
from fitron import FITRONModel

model = FITRONModel(iterations=10, random_state=42)
result = model.fit(data, target="decision")

print(f"Best candidate index: {result.best_index}")
print(f"Score: {result.best_score}")
```

---

### Workflow 2: Multi-Pass Refinement with Memory

Reuse weights and memory across multiple ranking calls:

```python
from fitron import FITRONModel

model = FITRONModel(iterations=5, random_state=42)

# First fit on training data
result1 = model.fit(train_data, target="decision")

# Access learned memory and weights
memory = model.memory
weights = result1.weights

# Apply to new candidates using same learned patterns
result2 = model.rank(new_candidates, target="decision")

print(f"Iteration 1 best: {result1.best_index}")
print(f"Iteration 2 best: {result2.best_index}")
print(f"Memory history entries: {len(memory.history)}")
```

---

### Workflow 3: Using Explicit Weights with Criteria

Define which features are benefits vs. costs:

```python
from fitron import fit

result = fit(
    df=data,
    target="approved",
    iterations=15,
    target_map={"rejected": 0, "approved": 1},
    criterion_types=["benefit", "cost", "benefit", "benefit"],  # cost = lower is better
    drop_columns=["applicant_id", "timestamp"],
)

print(f"Best index: {result.best_index}")
print(f"Scores: {result.scores}")
```

---

### Workflow 4: Saving Iteration Metrics

Track model performance across all iterations:

```python
from fitron import FITRONModel

model = FITRONModel(iterations=20)
result = model.fit(
    df=data,
    target="decision",
    metrics_output_path="./iteration_metrics.csv",
)

# CSV will contain: iteration, objective_score, top_candidate_score, 
#                   best_option_index, train_accuracy, test_accuracy, 
#                   threshold_balanced_accuracy, threshold_f1
```

Read and analyze metrics:

```python
import pandas as pd

metrics = pd.read_csv("./iteration_metrics.csv")
print(metrics[["iteration", "objective_score", "test_accuracy"]])

# Plot convergence
import matplotlib.pyplot as plt
plt.plot(metrics["iteration"], metrics["objective_score"])
plt.xlabel("Iteration")
plt.ylabel("Objective Score")
plt.show()
```

---

## Parameter Tuning Guide

### `iterations`

- **Low (5-10):** Fast experimentation, suitable for small datasets or demos.
- **Medium (20-30):** Balanced; good for most production use cases.
- **High (50+):** Slower but may find better solutions; risk of overfitting to noisy data.

### `objective_classification_weight`

- **0.5:** Equal weight to classification and ranking quality.
- **0.65 (default):** Emphasizes classification accuracy; prefer high-quality predictions.
- **0.8+:** Strong emphasis on correct classification; ranking quality is secondary.

### `decision_threshold`

- **0.3-0.4:** Lenient; accepts more candidates, may include borderline cases.
- **0.5 (default):** Standard probability threshold for binary classification.
- **0.7-0.9:** Strict; only high-confidence candidates are considered.

### `confidence_floor`

- **0.4-0.5:** Lenient fallback; will select candidates even with low ranking scores.
- **0.55-0.7:** Moderate; requires decent confidence; may trigger fallback on ambiguous data.
- **0.8+:** Strict; high standards; fallback will be used frequently.

---

## Troubleshooting

### Issue: Poor test accuracy or high fallback rate

**Causes:** Insufficient data, target leakage, poor feature quality.

**Solutions:**
- Increase dataset size to at least 20-30 rows.
- Review features for relevance to target.
- Lower `confidence_floor` to reduce fallback triggers.
- Increase `iterations` to refine weights.

---

### Issue: Inconsistent results across runs

**Cause:** Non-deterministic behavior due to random sampling.

**Solution:** Set `random_state` consistently:

```python
model = FITRONModel(iterations=10, random_state=42)
```

---

### Issue: Memory or weights not improving

**Cause:** Model is converging or data is too noisy.

**Solutions:**
- Check `memory.ema_improvement` to see if learning has stalled.
- Review `criterion_types` to ensure they match business logic.
- Inspect data quality for outliers or mislabeling.

---

## Run Tests

```bash
pytest -q
```


Release notes and version history are tracked in [CHANGELOG.md](CHANGELOG.md).

## Authors

- zibransheikh
- hazlived

## License

MIT
