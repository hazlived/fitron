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
from .core.mcdm import infer_criterion_types, topsis
from .core.memory import Memory
from .core.optimizer import optimize_weights_simplex, project_simplex
from .core.preprocessor import preprocess_data
from .core.reward import (
    compute_accuracy,
    compute_classification_quality,
    compute_objective_score,
    compute_ranking_quality,
    find_best_threshold,
    summarize_threshold_metrics,
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

    top_k = max(min_candidates, int(np.ceil(0.20 * len(probs))))
    top_k = min(top_k, len(probs))
    if top_k == len(probs):
        return np.arange(len(probs))
    return np.argsort(probs)[-top_k:]


def _optimize_weights(
    matrix: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    criterion_types: list[str],
    rounds: int = 3,
    step: float = 0.04,
    max_optimized_dims: int = 30,
) -> np.ndarray:
    """Simplex-projected criteria weight optimization."""
    return optimize_weights_simplex(
        matrix=matrix,
        utility_labels=labels,
        initial_weights=initial_weights,
        criterion_types=criterion_types,
        rounds=rounds,
        step=step,
        reg_lambda=0.05,
        max_optimized_dims=max_optimized_dims,
    )


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
        test_size=0.20,
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
        weights = project_simplex(weights)

    if memory is not None and memory.get_best() is not None:
        weights = blend_with_memory(weights, memory)

    candidate_rows = _select_candidate_rows(test_probs, min_candidates=3)
    candidate_indices = [int(idx) for idx in X_test_fuzzy.index[candidate_rows]]
    filtered_matrix = X_test_fuzzy.iloc[candidate_rows].to_numpy()
    filtered_labels = y_test.to_numpy()[candidate_rows]
    filtered_probs = test_probs[candidate_rows]

    if criterion_types is None:
        criterion_types = infer_criterion_types(filtered_matrix, filtered_probs)

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
    raw_scores = 0.80 * topsis_scores + 0.20 * prob_scaled

    if memory is not None:
        recency_factors = memory.compute_recency_penalty(candidate_indices, lambda_rec=0.10)
        scores = raw_scores * recency_factors
    else:
        scores = raw_scores

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

    current_score = float(scores[best_idx_local])
    weights = project_simplex(weights)

    if memory is not None:
        memory.update(
            weights=weights,
            score=objective_score,
            best_idx=best_idx_global,
            batch_scores=scores,
        )

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

            if result.objective_score > self.global_best_score:
                self.global_best_score = result.objective_score
                self.global_best_index = result.best_index

            self.iteration_records.append(
                {
                    "iteration": i + 1,
                    "objective_score": result.objective_score,
                    "top_candidate_score": result.best_score,
                    "best_option_index": result.best_index,
                    "train_accuracy": result.train_accuracy,
                    "test_accuracy": result.test_accuracy,
                    "threshold_balanced_accuracy": result.threshold_balanced_accuracy,
                    "threshold_f1": result.threshold_f1,
                }
            )

        if metrics_output_path is not None:
            out_file = Path(metrics_output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.iteration_records).to_csv(out_file, index=False)

        assert self.last_result is not None
        return self.last_result
