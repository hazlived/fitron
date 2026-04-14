from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd


def preprocess_data(
    df: pd.DataFrame,
    target: str,
    drop_columns: Sequence[str] | None = None,
    target_map: Mapping[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    if target_map is None:
        target_map = {"Y": 1, "N": 0}

    if y.dtype == "object":
        y = y.map(target_map)
        if y.isnull().any():
            raise ValueError("target contains unmapped values; provide target_map")

    if drop_columns:
        X = X.drop(columns=list(drop_columns), errors="ignore")

    for col in X.columns:
        numeric = pd.to_numeric(X[col], errors="coerce")
        if numeric.notnull().any():
            X[col] = numeric.fillna(numeric.mean())
        else:
            mode_series = X[col].mode(dropna=True)
            fallback = mode_series.iloc[0] if not mode_series.empty else "unknown"
            X[col] = X[col].fillna(fallback)

    X = pd.get_dummies(X, dtype=float)
    X = X.astype(float)

    return X, y.astype(int)
