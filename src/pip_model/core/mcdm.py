from __future__ import annotations

import numpy as np


def infer_criterion_types(matrix: np.ndarray, threat_probs: np.ndarray) -> list[str]:
    """Automatically infer TOPSIS criterion orientation (benefit vs cost).

    A criterion is classified as:
    - 'benefit': positively correlated with threat probability (risk driver)
    - 'cost': negatively correlated with threat probability (mitigating/protective factor)
    """
    matrix = np.asarray(matrix, dtype=float)
    threat_probs = np.asarray(threat_probs, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if len(threat_probs) != matrix.shape[0]:
        raise ValueError("threat_probs length must match matrix rows")

    n_cols = matrix.shape[1]
    criterion_types: list[str] = []

    # Centered ranks for Spearman correlation
    p_rank = np.argsort(np.argsort(threat_probs)).astype(float)
    p_rank -= np.mean(p_rank)
    p_norm = np.linalg.norm(p_rank)

    for j in range(n_cols):
        col = matrix[:, j]
        if np.all(col == col[0]):
            criterion_types.append("benefit")
            continue

        c_rank = np.argsort(np.argsort(col)).astype(float)
        c_rank -= np.mean(c_rank)
        c_norm = np.linalg.norm(c_rank)

        if p_norm > 0 and c_norm > 0:
            corr = float(np.dot(c_rank, p_rank) / (c_norm * p_norm))
        else:
            corr = 0.0

        if corr >= 0:
            criterion_types.append("benefit")
        else:
            criterion_types.append("cost")

    return criterion_types


def topsis(
    matrix: np.ndarray,
    weights: np.ndarray,
    criterion_types: list[str] | None = None,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Vectorized Multi-Criteria TOPSIS on the Simplex.

    Theorem 3 guarantees:
    - topsis_scores are bounded in [0.0, 1.0].
    - Ideal best option scores 1.0, ideal worst option scores 0.0.
    """
    matrix = np.asarray(matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if matrix.shape[1] != len(weights):
        raise ValueError("weights length must match number of criteria")

    if criterion_types is None:
        criterion_types = ["benefit"] * matrix.shape[1]
    if len(criterion_types) != matrix.shape[1]:
        raise ValueError("criterion_types length must match number of criteria")

    # Regularized vector normalization
    col_norms = np.sqrt(np.sum(matrix ** 2, axis=0))
    col_norms = np.where(col_norms == 0, 1.0, col_norms)
    norm = matrix / col_norms

    # Weighted decision matrix
    weighted = norm * weights

    n_criteria = matrix.shape[1]
    ideal_best = np.zeros(n_criteria, dtype=float)
    ideal_worst = np.zeros(n_criteria, dtype=float)

    for j, kind in enumerate(criterion_types):
        kind_norm = str(kind).lower().strip()
        if kind_norm == "cost":
            ideal_best[j] = np.min(weighted[:, j])
            ideal_worst[j] = np.max(weighted[:, j])
        else:
            ideal_best[j] = np.max(weighted[:, j])
            ideal_worst[j] = np.min(weighted[:, j])

    dist_best = np.sqrt(np.sum((weighted - ideal_best) ** 2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted - ideal_worst) ** 2, axis=1))

    total_dist = dist_best + dist_worst + epsilon
    closeness = dist_worst / total_dist
    return np.clip(closeness, 0.0, 1.0)
