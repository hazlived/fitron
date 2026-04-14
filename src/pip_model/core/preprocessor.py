from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def preprocess_data(
    df: pd.DataFrame,
    target: str,
    drop_columns: Sequence[str] | None = None,
    target_map: Mapping[str, int] | None = None,
    required_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    if required_columns is not None:
        missing = [col for col in required_columns if col not in X.columns]
        if missing:
            raise ValueError(f"input schema mismatch, missing columns: {missing}")

    if y.dtype == "object" or str(y.dtype).startswith("category"):
        if target_map is None:
            normalized = y.astype(str).str.strip().str.lower()
            uniques = sorted(set(normalized.dropna().unique()))
            if len(uniques) != 2:
                raise ValueError("target must be binary for this pipeline; provide target_map")

            positive_tokens = {"1", "true", "yes", "y", "approved", "pass", "positive"}
            if uniques[0] in positive_tokens and uniques[1] not in positive_tokens:
                target_map = {uniques[0]: 1, uniques[1]: 0}
            elif uniques[1] in positive_tokens and uniques[0] not in positive_tokens:
                target_map = {uniques[0]: 0, uniques[1]: 1}
            else:
                target_map = {uniques[0]: 0, uniques[1]: 1}
            y = normalized.map(target_map)
        else:
            y = y.map(target_map)

        if y.isnull().any():
            raise ValueError("target contains unmapped values; provide target_map")

    y = pd.to_numeric(y, errors="coerce")
    if y.isnull().any():
        raise ValueError("target contains non-numeric values after mapping")

    unique_target = set(pd.Series(y).dropna().unique().tolist())
    if len(unique_target) != 2 or not unique_target.issubset({0, 1}):
        raise ValueError("target must be binary and encoded as 0/1")

    if drop_columns:
        X = X.drop(columns=list(drop_columns), errors="ignore")

    auto_drop: list[str] = []
    row_count = len(X)
    for col in X.columns:
        if not (X[col].dtype == "object" or str(X[col].dtype).startswith("category")):
            continue

        unique_count = int(X[col].nunique(dropna=True))
        unique_ratio = unique_count / max(1, row_count)
        if unique_count > 50 and unique_ratio > 0.3:
            auto_drop.append(col)

    if auto_drop:
        X = X.drop(columns=auto_drop, errors="ignore")

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(float).fillna(float(X[col].median()))
            continue

        numeric = pd.to_numeric(X[col], errors="coerce")
        numeric_ratio = float(numeric.notnull().mean())

        if numeric_ratio >= 0.8:
            X[col] = numeric.fillna(float(numeric.median()))
        else:
            X[col] = X[col].astype(str).replace({"nan": np.nan}).fillna("__missing__")

    X = pd.get_dummies(X, dtype=float)
    X = X.astype(float)

    return X, y.astype(int)
