from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, ndcg_score, roc_auc_score


def compute_accuracy(preds: np.ndarray, y_true: np.ndarray) -> float:
    correct = np.sum(preds == y_true)
    return float(correct / len(y_true))


def compute_classification_quality(
    y_true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> float:
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)
    probs = np.asarray(probs)

    balanced = float(balanced_accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, zero_division=0))

    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, probs))
        ap = float(average_precision_score(y_true, probs))
    else:
        auc = 0.5
        ap = float(np.mean(y_true))

    return 0.35 * balanced + 0.25 * f1 + 0.2 * auc + 0.2 * ap


def compute_ranking_quality(labels: np.ndarray, ranking_scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    ranking_scores = np.asarray(ranking_scores, dtype=float)

    if labels.size == 0:
        return 0.0

    if np.all(labels == labels[0]):
        return 0.5

    ndcg = float(ndcg_score(labels.reshape(1, -1), ranking_scores.reshape(1, -1), k=min(5, labels.size)))
    ap = float(average_precision_score(labels, ranking_scores))
    return 0.6 * ndcg + 0.4 * ap


def compute_objective_score(
    classification_quality: float,
    ranking_quality: float,
    classification_weight: float = 0.65,
) -> float:
    classification_weight = float(classification_weight)
    if classification_weight < 0.0 or classification_weight > 1.0:
        raise ValueError("classification_weight must be between 0 and 1")

    ranking_weight = 1.0 - classification_weight
    return classification_weight * float(classification_quality) + ranking_weight * float(ranking_quality)


def summarize_threshold_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    threshold = float(threshold)
    preds = (np.asarray(probs) >= threshold).astype(int)
    return {
        "threshold": threshold,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }


def find_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray | None = None,
    balance_weight: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    if thresholds is None:
        thresholds = np.linspace(0.25, 0.75, 51)

    balance_weight = float(balance_weight)
    if balance_weight < 0.0 or balance_weight > 1.0:
        raise ValueError("balance_weight must be between 0 and 1")

    best: dict[str, float] | None = None
    for threshold in thresholds:
        metrics = summarize_threshold_metrics(y_true, probs, threshold=float(threshold))
        score = balance_weight * metrics["balanced_accuracy"] + (1.0 - balance_weight) * metrics["f1"]
        candidate = {
            "threshold": float(threshold),
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1": metrics["f1"],
            "score": float(score),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        raise ValueError("no thresholds available for tuning")
    return best
