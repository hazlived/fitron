from __future__ import annotations

import numpy as np


def topsis(
    matrix: np.ndarray,
    weights: np.ndarray,
    criterion_types: list[str] | None = None,
) -> np.ndarray:
    matrix = np.array(matrix, dtype=float)
    weights = np.array(weights, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if matrix.shape[1] != len(weights):
        raise ValueError("weights length must match number of criteria")

    if criterion_types is None:
        criterion_types = ["benefit"] * matrix.shape[1]
    if len(criterion_types) != matrix.shape[1]:
        raise ValueError("criterion_types length must match number of criteria")

    denom = np.sqrt((matrix ** 2).sum(axis=0))
    denom = np.where(denom == 0, 1.0, denom)
    norm = matrix / denom

    weighted = norm * weights

    ideal_best = np.zeros(matrix.shape[1], dtype=float)
    ideal_worst = np.zeros(matrix.shape[1], dtype=float)

    for j, kind in enumerate(criterion_types):
        kind_norm = kind.lower().strip()
        if kind_norm == "cost":
            ideal_best[j] = np.min(weighted[:, j])
            ideal_worst[j] = np.max(weighted[:, j])
        else:
            ideal_best[j] = np.max(weighted[:, j])
            ideal_worst[j] = np.min(weighted[:, j])

    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    return dist_worst / (dist_best + dist_worst + 1e-12)
