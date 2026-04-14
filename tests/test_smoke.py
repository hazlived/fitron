import numpy as np
import pandas as pd

from pip_model import FITRONModel, Memory, rank, update_memory


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "income": [50000, 20000, 80000, 32000, 61000, 45000, 72000, 26000, 53000, 68000],
            "age": [25, 22, 41, 27, 33, 35, 45, 24, 31, 39],
            "risk": [0.2, 0.8, 0.1, 0.7, 0.4, 0.5, 0.3, 0.9, 0.4, 0.2],
            "credit_score": [710, 520, 790, 560, 690, 640, 740, 500, 670, 730],
            "target": [1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
        }
    )


def test_model_fit_returns_result() -> None:
    df = _sample_df()
    model = FITRONModel(iterations=3, random_state=7)
    result = model.fit(df, target="target")

    assert isinstance(result.best_index, int)
    assert np.isfinite(result.best_score)
    assert len(result.explanation) > 0
    assert result.weights.ndim == 1


def test_rank_and_memory_update() -> None:
    df = _sample_df()
    memory = Memory()
    result = rank(df, target="target", memory=memory, random_state=11)

    update_memory(memory, result.weights, result.best_score, result.best_index)
    assert len(memory.history) >= 1
    assert memory.get_best() is not None
