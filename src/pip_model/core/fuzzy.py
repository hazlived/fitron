from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_col(col: pd.Series) -> pd.Series:
    col = col.fillna(col.mean())
    den = col.max() - col.min()
    if den == 0:
        return pd.Series(np.zeros(len(col)), index=col.index)
    return (col - col.min()) / den


def fuzzify_df(X: pd.DataFrame) -> pd.DataFrame:
    fuzzy_data: dict[str, pd.Series] = {}

    for col in X.columns:
        norm = _normalize_col(X[col])
        fuzzy_data[f"{col}_low"] = 1 - norm
        fuzzy_data[f"{col}_mid"] = 1 - (norm - 0.5).abs()
        fuzzy_data[f"{col}_high"] = norm

    return pd.DataFrame(fuzzy_data, index=X.index)
