from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    ndcg_score,
    roc_auc_score,
)


def compute_reward(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Compute standard classification accuracy."""
    preds = np.asarray(preds)
    y_true = np.asarray(y_true)
    if len(y_true) == 0:
        return 0.0
    return float(np.sum(preds == y_true) / len(y_true))


compute_accuracy = compute_reward


def compute_expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if len(y_true) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (probs >= low) & (probs < high) if i < n_bins - 1 else (probs >= low) & (probs <= high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(probs[mask])
            ece += (bin_count / total) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Compute Brier Score."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if len(y_true) == 0:
        return 0.0
    return float(brier_score_loss(y_true, probs))


def compute_classification_quality(
    y_true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> float:
    """Compute comprehensive classification quality index."""
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

    return float(0.35 * balanced + 0.25 * f1 + 0.20 * auc + 0.20 * ap)


def compute_ranking_quality(
    labels: np.ndarray,
    ranking_scores: np.ndarray,
    k: int = 5,
) -> float:
    """Compute multi-criteria candidate ranking quality.

    Supports graded operational utility labels g_i = y_i * psi(u_i) as well as binary labels.
    """
    labels = np.asarray(labels, dtype=float)
    ranking_scores = np.asarray(ranking_scores, dtype=float)

    if labels.size == 0:
        return 0.0
    if np.all(labels == labels[0]):
        return 0.5

    trunc_k = min(k, labels.size)
    ndcg = float(
        ndcg_score(
            labels.reshape(1, -1),
            ranking_scores.reshape(1, -1),
            k=trunc_k,
        )
    )

    # If labels are binary or graded, compute average precision
    # For graded labels, threshold at median positive for AP calculation
    pos_mask = labels > 0
    if np.any(pos_mask) and len(np.unique(pos_mask)) > 1:
        ap = float(average_precision_score(pos_mask.astype(int), ranking_scores))
    else:
        ap = float(np.mean(pos_mask)) if pos_mask.size > 0 else 0.5

    return float(0.60 * ndcg + 0.40 * ap)


def compute_objective_score(
    classification_quality: float,
    ranking_quality: float,
    classification_weight: float = 0.65,
) -> float:
    """Compute unified joint objective score."""
    classification_weight = float(classification_weight)
    if not (0.0 <= classification_weight <= 1.0):
        raise ValueError("classification_weight must be between 0.0 and 1.0")
    ranking_weight = 1.0 - classification_weight
    return float(classification_weight * classification_quality + ranking_weight * ranking_quality)


def summarize_threshold_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Summarize threshold operating metrics."""
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
    thresholds: list[float] | np.ndarray | None = None,
    balance_weight: float = 0.6,
) -> dict[str, float]:
    """Find optimal decision threshold maximizing balanced accuracy and F1."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    if thresholds is None:
        thresholds = np.linspace(0.25, 0.75, 51)

    balance_weight = float(balance_weight)
    if not (0.0 <= balance_weight <= 1.0):
        raise ValueError("balance_weight must be between 0.0 and 1.0")

    best = None
    for th in thresholds:
        metrics = summarize_threshold_metrics(y_true, probs, threshold=float(th))
        score = balance_weight * metrics["balanced_accuracy"] + (1.0 - balance_weight) * metrics["f1"]
        candidate = {
            "threshold": float(th),
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1": metrics["f1"],
            "score": float(score),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        raise ValueError("No thresholds available for tuning.")
    return best


def compute_pareto_knee(
    classification_quality: float,
    ranking_quality: float,
    alpha_grid: np.ndarray | None = None,
) -> float:
    """Derive optimal operating parameter alpha via Utopia Distance Knee-Point Selection.

    Solves:
        alpha* = argmin_{alpha} sqrt((1 - Q_cls)^2 + (1 - Q_rank)^2)
    """
    if alpha_grid is None:
        alpha_grid = np.linspace(0.1, 0.9, 81)

    best_dist = np.inf
    best_alpha = 0.5

    for a in alpha_grid:
        dist = np.sqrt((1.0 - classification_quality) ** 2 + (1.0 - ranking_quality) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_alpha = float(a)

    return best_alpha
