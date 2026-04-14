from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .core.adaptive import blend_with_memory, initialize_weights, normalize_weights, perturb_weights
from .core.decision_tree import predict, train_decision_tree
from .core.fuzzy import fuzzify_df
from .core.mcdm import topsis
from .core.memory import Memory
from .core.preprocessor import preprocess_data
from .core.reward import compute_accuracy


@dataclass
class IterationResult:
    predictions: np.ndarray
    scores: np.ndarray
    candidate_indices: list[int]
    best_index: int
    best_score: float
    weights: np.ndarray
    train_accuracy: float
    test_accuracy: float
    explanation: list[str]


def _generate_explanation(
    X_fuzzy: pd.DataFrame,
    feature_weights: np.ndarray,
    index: int,
    top_n: int = 5,
) -> list[str]:
    row = X_fuzzy.iloc[index]
    feature_data = list(zip(X_fuzzy.columns, feature_weights))
    feature_data.sort(key=lambda x: x[1], reverse=True)

    out: list[str] = []
    for name, importance in feature_data[:top_n]:
        out.append(f"{name}: {row[name]:.4f} (weight: {importance:.4f})")
    return out


def run_iteration(
    df: pd.DataFrame,
    target: str,
    weights: np.ndarray | None = None,
    memory: Memory | None = None,
    iteration: int = 0,
    random_state: int = 42,
    drop_columns: list[str] | None = None,
) -> IterationResult:
    X, y = preprocess_data(df, target, drop_columns=drop_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    X_train_fuzzy = fuzzify_df(X_train)
    X_test_fuzzy = fuzzify_df(X_test)

    model = train_decision_tree(X_train_fuzzy, y_train, random_state=random_state)

    train_preds = predict(model, X_train_fuzzy)
    test_preds = predict(model, X_test_fuzzy)

    train_acc = compute_accuracy(train_preds, y_train.to_numpy())
    test_acc = compute_accuracy(test_preds, y_test.to_numpy())

    if weights is None:
        weights = initialize_weights(model.feature_importances_)
    else:
        weights = normalize_weights(weights)

    if memory is not None and memory.get_best() is not None:
        weights = blend_with_memory(weights, memory)

    preds = train_preds
    valid_indices = [i for i, p in enumerate(preds) if p == 1]
    if len(valid_indices) < 3:
        valid_indices = list(range(len(preds)))

    filtered_matrix = X_train_fuzzy.iloc[valid_indices].to_numpy()

    rng = np.random.default_rng(random_state + iteration)

    scores = topsis(filtered_matrix, weights)
    scores = np.sqrt(np.clip(scores, 1e-12, None))

    row_std = np.std(filtered_matrix, axis=1)
    scores = scores + 0.2 * row_std
    scores = scores + rng.normal(0, 0.001, size=len(scores))

    best_idx_local = int(np.argmax(scores))
    best_idx_global = valid_indices[best_idx_local]

    if memory is not None:
        visit_count = sum(1 for h in memory.history if h.get("best_idx") == best_idx_global)
        if visit_count > 5:
            scores[best_idx_local] *= 0.9
            best_idx_local = int(np.argmax(scores))
            best_idx_global = valid_indices[best_idx_local]

    current_score = float(scores[best_idx_local])

    new_weights = perturb_weights(weights, scale=0.05, rng=rng)
    if rng.random() < 0.05:
        new_weights = rng.dirichlet(np.ones_like(weights))

    new_scores = topsis(filtered_matrix, new_weights)
    new_scores = np.sqrt(np.clip(new_scores, 1e-12, None))
    new_score = float(np.max(new_scores))

    if memory is not None and len(memory.history) >= 3:
        recent = [h["improvement"] for h in memory.history[-3:]]
        avg_improvement = float(np.mean(recent))
        explore_prob = 0.35 if avg_improvement < 0.01 else 0.1
    else:
        explore_prob = 0.2

    if (new_score > current_score) or (rng.random() < explore_prob):
        weights = new_weights

    if memory is not None and len(memory.history) >= 5:
        recent_scores = [h["score"] for h in memory.history[-5:]]
        if max(recent_scores) - min(recent_scores) < 0.02:
            weights = perturb_weights(weights, scale=0.1, rng=rng)

    weights = normalize_weights(weights)

    if memory is not None:
        memory.update(weights, current_score, best_idx_global)

    explanation = _generate_explanation(X_train_fuzzy, weights, best_idx_global)

    return IterationResult(
        predictions=preds,
        scores=scores,
        candidate_indices=valid_indices,
        best_index=best_idx_global,
        best_score=current_score,
        weights=weights,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        explanation=explanation,
    )


class FITRONModel:
    def __init__(self, iterations: int = 20, random_state: int = 42) -> None:
        self.iterations = iterations
        self.random_state = random_state
        self.memory = Memory()
        self.weights: np.ndarray | None = None
        self.last_result: IterationResult | None = None
        self.score_history: list[float] = []
        self.global_best_score: float = float("-inf")
        self.global_best_index: int | None = None

    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        drop_columns: list[str] | None = None,
    ) -> IterationResult:
        for i in range(self.iterations):
            result = run_iteration(
                df=df,
                target=target,
                weights=self.weights,
                memory=self.memory,
                iteration=i,
                random_state=self.random_state,
                drop_columns=drop_columns,
            )

            self.weights = result.weights
            self.last_result = result
            self.score_history.append(result.best_score)

            if result.best_score > self.global_best_score:
                self.global_best_score = result.best_score
                self.global_best_index = result.best_index

        if self.last_result is None:
            raise RuntimeError("fit completed without producing a result")
        return self.last_result

    def rank(
        self,
        df: pd.DataFrame,
        target: str,
        drop_columns: list[str] | None = None,
    ) -> IterationResult:
        result = run_iteration(
            df=df,
            target=target,
            weights=self.weights,
            memory=self.memory,
            iteration=0,
            random_state=self.random_state,
            drop_columns=drop_columns,
        )
        self.last_result = result
        self.weights = result.weights
        return result

    def explain(self) -> list[str]:
        if self.last_result is None:
            raise RuntimeError("no result available; call fit() or rank() first")
        return self.last_result.explanation
