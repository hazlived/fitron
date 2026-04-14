from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .core.adaptive import blend_with_memory, initialize_weights, normalize_weights, perturb_weights
from .core.decision_tree import predict, predict_proba_positive, train_decision_tree
from .core.fuzzy import fit_fuzzy_profile, transform_fuzzy
from .core.mcdm import topsis
from .core.memory import Memory
from .core.preprocessor import preprocess_data
from .core.reward import (
    compute_accuracy,
    compute_classification_quality,
    compute_objective_score,
    compute_ranking_quality,
    summarize_threshold_metrics,
    find_best_threshold,
)


@dataclass
class IterationResult:
    predictions: np.ndarray
    scores: np.ndarray
    candidate_indices: list[int]
    best_index: int
    best_score: float
    objective_score: float
    weights: np.ndarray
    train_accuracy: float
    test_accuracy: float
    classification_quality: float
    ranking_quality: float
    threshold_balanced_accuracy: float
    threshold_f1: float
    fallback_triggered: bool
    model: object
    explanation: list[str]


def _generate_explanation(
    X_fuzzy: pd.DataFrame,
    feature_importances: np.ndarray,
    index: int,
    top_n: int = 5,
) -> list[str]:
    row = X_fuzzy.iloc[index]

    if len(feature_importances) != len(X_fuzzy.columns):
        feature_importances = np.ones(len(X_fuzzy.columns), dtype=float) / len(X_fuzzy.columns)

    feature_data = list(zip(X_fuzzy.columns, feature_importances))
    feature_data.sort(key=lambda x: x[1], reverse=True)

    out: list[str] = []
    for name, importance in feature_data[:top_n]:
        out.append(f"{name}: {row[name]:.4f} (importance: {importance:.4f})")
    return out


def _select_candidate_rows(probs: np.ndarray, min_candidates: int = 3) -> np.ndarray:
    positive_rows = np.where(probs >= 0.5)[0]
    if len(positive_rows) >= min_candidates:
        return positive_rows

    top_k = max(min_candidates, int(np.ceil(0.2 * len(probs))))
    top_k = min(top_k, len(probs))
    if top_k == len(probs):
        return np.arange(len(probs))
    return np.argsort(probs)[-top_k:]


def _optimize_weights(
    matrix: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    criterion_types: list[str],
    rounds: int = 2,
    step: float = 0.03,
    max_optimized_dims: int = 30,
) -> np.ndarray:
    best = normalize_weights(initial_weights.copy())

    def objective(w: np.ndarray) -> float:
        rank_scores = topsis(matrix, w, criterion_types=criterion_types)
        rank_quality = compute_ranking_quality(labels, rank_scores)
        concentration_penalty = float(np.sum(w ** 2))
        return rank_quality - 0.05 * concentration_penalty

    best_score = objective(best)

    if len(best) > max_optimized_dims:
        optimize_idx = np.argsort(best)[-max_optimized_dims:]
    else:
        optimize_idx = np.arange(len(best))

    for _ in range(rounds):
        improved = False
        for i in optimize_idx:
            for direction in (-1.0, 1.0):
                trial = best.copy()
                trial[i] = max(1e-6, trial[i] + direction * step)
                trial = normalize_weights(trial)
                trial_score = objective(trial)
                if trial_score > best_score:
                    best = trial
                    best_score = trial_score
                    improved = True
        if not improved:
            step *= 0.5

    return best


def tune_decision_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    balance_weight: float = 0.6,
    tune_hyperparameters: bool = True,
    threshold_grid: np.ndarray | None = None,
) -> dict[str, float]:
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    fuzzy_profile = fit_fuzzy_profile(X_train)
    X_train_fuzzy = transform_fuzzy(X_train, fuzzy_profile)
    X_val_fuzzy = transform_fuzzy(X_val, fuzzy_profile)

    model, _ = train_decision_tree(
        X_train_fuzzy,
        y_train,
        random_state=random_state,
        tune_hyperparameters=tune_hyperparameters,
    )

    val_probs = predict_proba_positive(model, X_val_fuzzy)
    return find_best_threshold(y_val.to_numpy(), val_probs, thresholds=threshold_grid, balance_weight=balance_weight)


def run_iteration(
    df: pd.DataFrame,
    target: str,
    weights: np.ndarray | None = None,
    memory: Memory | None = None,
    iteration: int = 0,
    random_state: int = 42,
    drop_columns: list[str] | None = None,
    criterion_types: list[str] | None = None,
    decision_threshold: float = 0.5,
    objective_classification_weight: float = 0.65,
    confidence_floor: float = 0.55,
    expected_feature_columns: list[str] | None = None,
    tune_hyperparameters: bool = True,
    target_map: dict[str, int] | None = None,
) -> IterationResult:
    X, y = preprocess_data(
        df,
        target,
        drop_columns=drop_columns,
        target_map=target_map,
        required_columns=expected_feature_columns,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    fuzzy_profile = fit_fuzzy_profile(X_train)
    X_train_fuzzy = transform_fuzzy(X_train, fuzzy_profile)
    X_test_fuzzy = transform_fuzzy(X_test, fuzzy_profile)

    model, feature_importances = train_decision_tree(
        X_train_fuzzy,
        y_train,
        random_state=random_state,
        tune_hyperparameters=tune_hyperparameters,
    )

    train_preds = predict(model, X_train_fuzzy, threshold=decision_threshold)
    test_preds = predict(model, X_test_fuzzy, threshold=decision_threshold)
    test_probs = predict_proba_positive(model, X_test_fuzzy)

    train_acc = compute_accuracy(train_preds, y_train.to_numpy())
    test_acc = compute_accuracy(test_preds, y_test.to_numpy())
    classification_quality = compute_classification_quality(y_test.to_numpy(), test_preds, test_probs)
    threshold_stats = summarize_threshold_metrics(y_test.to_numpy(), test_probs, threshold=decision_threshold)

    if weights is None:
        weights = initialize_weights(feature_importances)
    else:
        weights = normalize_weights(weights)

    if memory is not None and memory.get_best() is not None:
        weights = blend_with_memory(weights, memory)

    candidate_rows = _select_candidate_rows(test_probs, min_candidates=3)
    candidate_indices = X_test_fuzzy.index[candidate_rows].tolist()
    filtered_matrix = X_test_fuzzy.iloc[candidate_rows].to_numpy()
    filtered_labels = y_test.to_numpy()[candidate_rows]
    filtered_probs = test_probs[candidate_rows]

    if criterion_types is None:
        criterion_types = ["benefit"] * filtered_matrix.shape[1]

    weights = _optimize_weights(
        filtered_matrix,
        filtered_labels,
        weights,
        criterion_types=criterion_types,
        rounds=3,
        step=0.04,
    )

    topsis_scores = topsis(filtered_matrix, weights, criterion_types=criterion_types)
    prob_min = float(np.min(filtered_probs))
    prob_ptp = float(np.ptp(filtered_probs))
    prob_scaled = (filtered_probs - prob_min) / (prob_ptp + 1e-12)
    scores = 0.8 * topsis_scores + 0.2 * prob_scaled

    ranking_quality = compute_ranking_quality(filtered_labels, scores)
    objective_score = compute_objective_score(
        classification_quality,
        ranking_quality,
        classification_weight=objective_classification_weight,
    )

    best_idx_local = int(np.argmax(scores))
    best_idx_global = int(candidate_indices[best_idx_local])
    fallback_triggered = False

    if float(scores[best_idx_local]) < float(confidence_floor):
        best_idx_local = int(np.argmax(filtered_probs))
        best_idx_global = int(candidate_indices[best_idx_local])
        fallback_triggered = True

    if memory is not None:
        visit_count = sum(1 for h in memory.history if h.get("best_idx") == best_idx_global)
        if visit_count > 5:
            scores[best_idx_local] *= 0.9
            best_idx_local = int(np.argmax(scores))
            best_idx_global = int(candidate_indices[best_idx_local])

    current_score = float(scores[best_idx_local])

    weights = normalize_weights(weights)

    if memory is not None:
        memory.update(weights, objective_score, best_idx_global)

    explanation = _generate_explanation(
        X_test_fuzzy,
        feature_importances,
        index=candidate_rows[best_idx_local],
    )

    return IterationResult(
        predictions=test_preds,
        scores=scores,
        candidate_indices=candidate_indices,
        best_index=best_idx_global,
        best_score=current_score,
        objective_score=objective_score,
        weights=weights,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        classification_quality=classification_quality,
        ranking_quality=ranking_quality,
        threshold_balanced_accuracy=threshold_stats["balanced_accuracy"],
        threshold_f1=threshold_stats["f1"],
        fallback_triggered=fallback_triggered,
        model=model,
        explanation=explanation,
    )


class FITRONModel:
    def __init__(
        self,
        iterations: int = 20,
        random_state: int = 42,
        decision_threshold: float = 0.5,
        objective_classification_weight: float = 0.65,
        confidence_floor: float = 0.55,
    ) -> None:
        self.iterations = iterations
        self.random_state = random_state
        self.decision_threshold = decision_threshold
        self.objective_classification_weight = objective_classification_weight
        self.confidence_floor = confidence_floor
        self.memory = Memory()
        self.weights: np.ndarray | None = None
        self.last_result: IterationResult | None = None
        self.score_history: list[float] = []
        self.top_score_history: list[float] = []
        self.global_best_score: float = float("-inf")
        self.global_best_index: int | None = None
        self.iteration_records: list[dict[str, float | int]] = []

    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        drop_columns: list[str] | None = None,
        criterion_types: list[str] | None = None,
        expected_feature_columns: list[str] | None = None,
        metrics_output_path: str | None = None,
        target_map: dict[str, int] | None = None,
    ) -> IterationResult:
        if expected_feature_columns is None:
            expected_feature_columns = [c for c in df.columns if c != target]

        self.iteration_records = []
        for i in range(self.iterations):
            result = run_iteration(
                df=df,
                target=target,
                weights=self.weights,
                memory=self.memory,
                iteration=i,
                random_state=self.random_state + i,
                drop_columns=drop_columns,
                criterion_types=criterion_types,
                decision_threshold=self.decision_threshold,
                objective_classification_weight=self.objective_classification_weight,
                confidence_floor=self.confidence_floor,
                expected_feature_columns=expected_feature_columns,
                tune_hyperparameters=(i == 0),
                target_map=target_map,
            )

            self.weights = result.weights
            self.last_result = result
            self.score_history.append(result.objective_score)
            self.top_score_history.append(result.best_score)

            self.iteration_records.append(
                {
                    "iteration": i + 1,
                    "objective_score": float(result.objective_score),
                    "top_candidate_score": float(result.best_score),
                    "best_option_index": int(result.best_index),
                    "train_accuracy": float(result.train_accuracy),
                    "test_accuracy": float(result.test_accuracy),
                    "threshold_balanced_accuracy": float(result.threshold_balanced_accuracy),
                    "threshold_f1": float(result.threshold_f1),
                }
            )

            if result.objective_score > self.global_best_score:
                self.global_best_score = result.objective_score
                self.global_best_index = result.best_index

        if self.last_result is None:
            raise RuntimeError("fit completed without producing a result")

        if metrics_output_path is not None:
            output_path = Path(metrics_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.iteration_records).to_csv(output_path, index=False)

        return self.last_result

    def rank(
        self,
        df: pd.DataFrame,
        target: str,
        drop_columns: list[str] | None = None,
        criterion_types: list[str] | None = None,
        expected_feature_columns: list[str] | None = None,
        tune_hyperparameters: bool = False,
        target_map: dict[str, int] | None = None,
    ) -> IterationResult:
        if expected_feature_columns is None:
            expected_feature_columns = [c for c in df.columns if c != target]

        result = run_iteration(
            df=df,
            target=target,
            weights=self.weights,
            memory=self.memory,
            iteration=0,
            random_state=self.random_state,
            drop_columns=drop_columns,
            criterion_types=criterion_types,
            decision_threshold=self.decision_threshold,
            objective_classification_weight=self.objective_classification_weight,
            confidence_floor=self.confidence_floor,
            expected_feature_columns=expected_feature_columns,
            tune_hyperparameters=tune_hyperparameters,
            target_map=target_map,
        )
        self.last_result = result
        self.weights = result.weights
        return result

    def explain(self) -> list[str]:
        if self.last_result is None:
            raise RuntimeError("no result available; call fit() or rank() first")
        return self.last_result.explanation
