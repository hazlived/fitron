from __future__ import annotations

import numpy as np
import pandas as pd

from .core.memory import Memory
from .pipeline import IterationResult, TRACEModel, run_iteration


def fit(
    df: pd.DataFrame,
    target: str,
    iterations: int = 20,
    random_state: int = 42,
    drop_columns: list[str] | None = None,
) -> IterationResult:
    model = TRACEModel(iterations=iterations, random_state=random_state)
    return model.fit(df=df, target=target, drop_columns=drop_columns)


def rank(
    df: pd.DataFrame,
    target: str,
    weights: np.ndarray | None = None,
    memory: Memory | None = None,
    random_state: int = 42,
    drop_columns: list[str] | None = None,
) -> IterationResult:
    return run_iteration(
        df=df,
        target=target,
        weights=weights,
        memory=memory,
        random_state=random_state,
        drop_columns=drop_columns,
    )


def explain(result: IterationResult) -> list[str]:
    return result.explanation


def update_memory(memory: Memory, weights: np.ndarray, score: float, best_idx: int) -> None:
    memory.update(weights=weights, score=score, best_idx=best_idx)
