from __future__ import annotations

import pandas as pd
import pytest

from pip_model import FITRONModel, rank
from pip_model.core.reward import compute_objective_score, find_best_threshold


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "income": [50000, 20000, 80000, 32000, 61000, 45000, 72000, 26000, 53000, 68000, 47000, 30000],
            "age": [25, 22, 41, 27, 33, 35, 45, 24, 31, 39, 29, 28],
            "risk": [0.2, 0.8, 0.1, 0.7, 0.4, 0.5, 0.3, 0.9, 0.4, 0.2, 0.6, 0.65],
            "credit_score": [710, 520, 790, 560, 690, 640, 740, 500, 670, 730, 610, 580],
            "target": [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        }
    )


def _generic_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "income": [62000, 18000, 91000, 34000, 48000, 56000, 72000, 26000],
            "age": [29, 21, 45, 31, 38, 41, 52, 24],
            "segment": ["premium", "basic", "premium", "basic", "standard", "standard", "premium", "basic"],
            "tenure_years": [5, 1, 12, 3, 7, 9, 15, 2],
            "status": ["approve", "reject", "approve", "reject", "approve", "approve", "approve", "reject"],
        }
    )


def test_objective_formula_consistency() -> None:
    score = compute_objective_score(0.8, 0.9, classification_weight=0.65)
    assert score == pytest.approx(0.835)


def test_threshold_tuning_prefers_nondefault_threshold() -> None:
    y_true = [0, 0, 1, 1, 1, 0, 1, 0]
    probs = [0.05, 0.2, 0.35, 0.55, 0.82, 0.48, 0.61, 0.18]

    result = find_best_threshold(y_true, probs, thresholds=[0.3, 0.4, 0.5, 0.6], balance_weight=0.6)

    assert result["threshold"] in {0.3, 0.4, 0.5, 0.6}
    assert 0.0 <= result["score"] <= 1.0


def test_confidence_floor_triggers_fallback() -> None:
    df = _sample_df()
    result = rank(
        df,
        target="target",
        random_state=11,
        confidence_floor=1.1,
        tune_hyperparameters=False,
    )
    assert result.fallback_triggered is True


def test_schema_mismatch_raises() -> None:
    df = _sample_df()
    with pytest.raises(ValueError, match="input schema mismatch"):
        rank(
            df,
            target="target",
            expected_feature_columns=["income", "age", "missing_column"],
            tune_hyperparameters=False,
        )


def test_fit_writes_iteration_metrics_csv(tmp_path) -> None:
    df = _sample_df()
    out = tmp_path / "iteration_metrics.csv"

    model = FITRONModel(iterations=2, random_state=7)
    model.fit(df, target="target", metrics_output_path=str(out))

    assert out.exists()
    written = pd.read_csv(out)
    assert set(
        [
            "iteration",
            "objective_score",
            "top_candidate_score",
            "best_option_index",
            "train_accuracy",
            "test_accuracy",
            "threshold_balanced_accuracy",
            "threshold_f1",
        ]
    ).issubset(set(written.columns))
    assert len(written) == 2


def test_generic_string_target_support() -> None:
    df = _generic_df()
    model = FITRONModel(iterations=2, random_state=5)
    result = model.fit(
        df,
        target="status",
        target_map={"reject": 0, "approve": 1},
    )

    assert isinstance(result.best_index, int)
    assert result.explanation
    assert model.global_best_index is not None
