from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FuzzyProfile:
    mins: pd.Series
    maxs: pd.Series
    mids: pd.Series


def fit_fuzzy_profile(X: pd.DataFrame) -> FuzzyProfile:
    X_num = X.astype(float)
    mins = X_num.min(axis=0)
    maxs = X_num.max(axis=0)
    mids = X_num.median(axis=0)

    # Keep denominator valid for constant columns.
    maxs = pd.Series(np.where((maxs - mins) == 0, mins + 1.0, maxs), index=maxs.index)
    return FuzzyProfile(mins=mins, maxs=maxs, mids=mids)


def transform_fuzzy(X: pd.DataFrame, profile: FuzzyProfile) -> pd.DataFrame:
    X_num = X.astype(float)
    fuzzy_data: dict[str, pd.Series] = {}

    for col in X_num.columns:
        col_vals = X_num[col].fillna(float(profile.mids[col]))
        denom = float(profile.maxs[col] - profile.mins[col])
        norm = (col_vals - float(profile.mins[col])) / denom
        norm = norm.clip(0.0, 1.0)

        fuzzy_data[f"{col}_low"] = 1.0 - norm
        fuzzy_data[f"{col}_mid"] = (1.0 - (norm - 0.5).abs() * 2.0).clip(0.0, 1.0)
        fuzzy_data[f"{col}_high"] = norm

    return pd.DataFrame(fuzzy_data, index=X.index)


def fuzzify_df(X: pd.DataFrame, profile: FuzzyProfile | None = None) -> pd.DataFrame:
    if profile is None:
        profile = fit_fuzzy_profile(X)
    return transform_fuzzy(X, profile)
