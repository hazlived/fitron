from __future__ import annotations

import numpy as np
import pandas as pd

from .core.memory import Memory
from .pipeline import FITRONModel, IterationResult, run_iteration


def fit(
    df: pd.DataFrame,
    target: str,
    iterations: int = 20,
    random_state: int = 42,
    decision_threshold: float = 0.5,
    objective_classification_weight: float = 0.65,
    confidence_floor: float = 0.55,
    drop_columns: list[str] | None = None,
    criterion_types: list[str] | None = None,
    expected_feature_columns: list[str] | None = None,
    metrics_output_path: str | None = None,
    target_map: dict[str, int] | None = None,
) -> IterationResult:
    model = FITRONModel(
        iterations=iterations,
        random_state=random_state,
        decision_threshold=decision_threshold,
        objective_classification_weight=objective_classification_weight,
        confidence_floor=confidence_floor,
    )
    return model.fit(
        df=df,
        target=target,
        drop_columns=drop_columns,
        criterion_types=criterion_types,
        expected_feature_columns=expected_feature_columns,
        metrics_output_path=metrics_output_path,
        target_map=target_map,
    )


def rank(
    df: pd.DataFrame,
    target: str,
    weights: np.ndarray | None = None,
    memory: Memory | None = None,
    random_state: int = 42,
    decision_threshold: float = 0.5,
    objective_classification_weight: float = 0.65,
    confidence_floor: float = 0.55,
    drop_columns: list[str] | None = None,
    criterion_types: list[str] | None = None,
    expected_feature_columns: list[str] | None = None,
    tune_hyperparameters: bool = False,
    target_map: dict[str, int] | None = None,
) -> IterationResult:
    return run_iteration(
        df=df,
        target=target,
        weights=weights,
        memory=memory,
        random_state=random_state,
        drop_columns=drop_columns,
        criterion_types=criterion_types,
        decision_threshold=decision_threshold,
        objective_classification_weight=objective_classification_weight,
        confidence_floor=confidence_floor,
        expected_feature_columns=expected_feature_columns,
        tune_hyperparameters=tune_hyperparameters,
        target_map=target_map,
    )


def explain(result: IterationResult) -> list[str]:
    return result.explanation


def update_memory(memory: Memory, weights: np.ndarray, score: float, best_idx: int) -> None:
    memory.update(weights=weights, score=score, best_idx=best_idx)
