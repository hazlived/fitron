from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class FuzzyProfile:
    """Robust Quantile Knots for Ruspini Fuzzy Partition.

    Knots (a, b, c) correspond to regularized quartiles or interval knots,
    guaranteeing strict ordering a < b < c and partition of unity completeness:
    mu_low(x) + mu_mid(x) + mu_high(x) == 1.0 for all x in R.
    Supports continuous attributes, zero-inflated distributions, and binary indicators.
    """
    a: pd.Series
    b: pd.Series
    c: pd.Series
    is_binary: pd.Series | None = None


def fit_fuzzy_profile(
    X: pd.DataFrame,
    epsilon_tol: float = 1e-7,
) -> FuzzyProfile:
    """Fit robust quantile knots for each numeric column in X.

    Regularization ensures strict knot ordering a < b < c even for
    constant, degenerate, or zero-inflated distributions, and preserves
    binary indicator features without variance collapse.
    """
    X_num = X.astype(float)
    a_dict: dict[str, float] = {}
    b_dict: dict[str, float] = {}
    c_dict: dict[str, float] = {}
    is_bin_dict: dict[str, bool] = {}

    for col in X_num.columns:
        vals = X_num[col].dropna().to_numpy()
        if len(vals) == 0:
            a_dict[col] = 0.0
            b_dict[col] = 0.5
            c_dict[col] = 1.0
            is_bin_dict[col] = False
            continue

        uvals = np.unique(vals)
        if len(uvals) <= 2:
            is_bin_dict[col] = True
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))
            a_dict[col] = min_v
            b_dict[col] = (min_v + max_v) / 2.0 if max_v > min_v else min_v + 0.5
            c_dict[col] = max_v if max_v > min_v else min_v + 1.0
        else:
            is_bin_dict[col] = False
            q25 = float(np.percentile(vals, 25))
            q50 = float(np.percentile(vals, 50))
            q75 = float(np.percentile(vals, 75))

            if q25 == q75:
                min_v = float(np.min(vals))
                max_v = float(np.max(vals))
                if max_v > min_v:
                    a_dict[col] = min_v
                    b_dict[col] = (min_v + max_v) / 2.0
                    c_dict[col] = max_v
                else:
                    a_dict[col] = min_v
                    b_dict[col] = min_v + 0.5
                    c_dict[col] = min_v + 1.0
            else:
                scale = max(abs(q50), 1.0)
                delta = max(epsilon_tol, 1e-4 * scale)
                a = q25
                b = max(q50, a + delta)
                c = max(q75, b + delta)
                a_dict[col] = a
                b_dict[col] = b
                c_dict[col] = c

    cols = X_num.columns
    return FuzzyProfile(
        a=pd.Series(a_dict, index=cols),
        b=pd.Series(b_dict, index=cols),
        c=pd.Series(c_dict, index=cols),
        is_binary=pd.Series(is_bin_dict, index=cols),
    )


def transform_fuzzy(X: pd.DataFrame, profile: FuzzyProfile) -> pd.DataFrame:
    """Transform continuous numerical attributes into strict Ruspini fuzzy memberships.

    mu_low(x):
      1.0                           if x <= a
      (b - x) / (b - a)             if a < x < b
      0.0                           if x >= b

    mu_mid(x):
      0.0                           if x <= a or x >= c
      (x - a) / (b - a)             if a < x <= b
      (c - x) / (c - b)             if b < x < c

    mu_high(x):
      0.0                           if x <= b
      (x - b) / (c - b)             if b < x < c
      1.0                           if x >= c

    Theorem 2 guarantees: mu_low(x) + mu_mid(x) + mu_high(x) == 1.0 for all x.
    """
    X_num = X.astype(float)
    fuzzy_data: dict[str, pd.Series] = {}

    for col in X_num.columns:
        a_val = float(profile.a[col])
        b_val = float(profile.b[col])
        c_val = float(profile.c[col])
        is_bin = bool(profile.is_binary[col]) if profile.is_binary is not None and col in profile.is_binary else False

        # Fill missing values with median knot b_val
        col_vals = X_num[col].fillna(b_val).to_numpy()

        if is_bin:
            span = c_val - a_val
            if span > 1e-12:
                norm_val = np.clip((col_vals - a_val) / span, 0.0, 1.0)
                low = 1.0 - norm_val
                mid = np.zeros_like(norm_val)
                high = norm_val
            else:
                low = np.zeros_like(col_vals)
                mid = np.ones_like(col_vals)
                high = np.zeros_like(col_vals)
        else:
            denom_low = max(b_val - a_val, 1e-12)
            denom_high = max(c_val - b_val, 1e-12)

            low = np.where(
                col_vals <= a_val,
                1.0,
                np.where(col_vals < b_val, (b_val - col_vals) / denom_low, 0.0),
            )
            low = np.clip(low, 0.0, 1.0)

            high = np.where(
                col_vals <= b_val,
                0.0,
                np.where(col_vals < c_val, (col_vals - b_val) / denom_high, 1.0),
            )
            high = np.clip(high, 0.0, 1.0)

            # Enforce exact Ruspini partition of unity
            mid = np.clip(1.0 - (low + high), 0.0, 1.0)

        fuzzy_data[f"{col}_low"] = pd.Series(low, index=X.index)
        fuzzy_data[f"{col}_mid"] = pd.Series(mid, index=X.index)
        fuzzy_data[f"{col}_high"] = pd.Series(high, index=X.index)

    return pd.DataFrame(fuzzy_data, index=X.index)


def fuzzify_df(X: pd.DataFrame, profile: FuzzyProfile | None = None) -> pd.DataFrame:
    """Convenience interface to fit and transform a DataFrame."""
    if profile is None:
        profile = fit_fuzzy_profile(X)
    return transform_fuzzy(X, profile)
