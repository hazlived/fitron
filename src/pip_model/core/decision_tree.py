from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .reward import compute_brier_score, compute_expected_calibration_error


def train_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    tune_hyperparameters: bool = True,
    calibration_method: str = "sigmoid",
) -> tuple[object, np.ndarray]:
    """Train CART decision tree with cross-validation tuning and probability calibration."""
    class_counts = y.value_counts()
    imbalance = (class_counts.max() / max(1, class_counts.min())) > 1.5 if len(class_counts) > 1 else False
    cw = "balanced" if imbalance else None

    base = DecisionTreeClassifier(random_state=random_state, class_weight=cw)

    if not tune_hyperparameters:
        base.set_params(max_depth=5, min_samples_leaf=5)
        base.fit(X, y)
        fitted = base
        importances = base.feature_importances_.copy()
        return _calibrate_if_possible(fitted, X, y, method=calibration_method), importances

    min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0
    if min_class_count < 2:
        base.set_params(max_depth=5, min_samples_leaf=5)
        base.fit(X, y)
        fitted = base
        importances = base.feature_importances_.copy()
        return _calibrate_if_possible(fitted, X, y, method=calibration_method), importances

    cv_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = "f1" if len(class_counts) == 2 else "accuracy"

    grid = {
        "max_depth": [3, 5, 7, None],
        "min_samples_leaf": [1, 3, 8],
        "class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=base,
        param_grid=grid,
        scoring=scoring,
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y)
    fitted = search.best_estimator_
    importances = fitted.feature_importances_.copy()
    return _calibrate_if_possible(fitted, X, y, method=calibration_method), importances


def _calibrate_if_possible(model, X: pd.DataFrame, y: pd.Series, method: str = "sigmoid"):
    class_counts = y.value_counts()
    min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0
    if min_class_count < 3:
        return model

    cv_splits = min(3, min_class_count)
    calibrator = CalibratedClassifierCV(estimator=model, method=method, cv=cv_splits)
    calibrator.fit(X, y)
    return calibrator


def predict(model, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    probs = predict_proba_positive(model, X)
    return (probs >= float(threshold)).astype(int)


def predict_proba_positive(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        return np.ones(len(X), dtype=float)

    class_labels = list(model.classes_)
    if 1 in class_labels:
        idx = class_labels.index(1)
    else:
        idx = int(np.argmax(class_labels))
    return proba[:, idx]


def evaluate_generalization_cv(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_splits: int = 5,
    calibration_method: str = "sigmoid",
) -> dict[str, float]:
    """Stratified cross-validation evaluating classification, discrimination, and calibration (ECE/Brier)."""
    class_counts = y.value_counts()
    min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0
    if min_class_count < 2:
        raise ValueError("Not enough data in minority class for stratified CV")

    splits = min(n_splits, min_class_count)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)

    bal_scores = []
    f1_scores = []
    auc_scores = []
    ap_scores = []
    ece_scores = []
    brier_scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model, _ = train_decision_tree(
            X_train,
            y_train,
            random_state=random_state,
            tune_hyperparameters=False,
            calibration_method=calibration_method,
        )
        probs = predict_proba_positive(model, X_test)
        preds = (probs >= 0.5).astype(int)

        y_test_arr = y_test.to_numpy()
        bal_scores.append(float(balanced_accuracy_score(y_test_arr, preds)))
        f1_scores.append(float(f1_score(y_test_arr, preds, zero_division=0)))

        if len(np.unique(y_test_arr)) > 1:
            auc_scores.append(float(roc_auc_score(y_test_arr, probs)))
            ap_scores.append(float(average_precision_score(y_test_arr, probs)))
        else:
            auc_scores.append(0.5)
            ap_scores.append(float(np.mean(y_test_arr)))

        ece_scores.append(compute_expected_calibration_error(y_test_arr, probs))
        brier_scores.append(compute_brier_score(y_test_arr, probs))

    return {
        "cv_splits": float(splits),
        "balanced_accuracy_mean": float(np.mean(bal_scores)),
        "balanced_accuracy_std": float(np.std(bal_scores)),
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "auc_mean": float(np.mean(auc_scores)),
        "auc_std": float(np.std(auc_scores)),
        "ap_mean": float(np.mean(ap_scores)),
        "ap_std": float(np.std(ap_scores)),
        "ece_mean": float(np.mean(ece_scores)),
        "ece_std": float(np.std(ece_scores)),
        "brier_mean": float(np.mean(brier_scores)),
        "brier_std": float(np.std(brier_scores)),
    }
