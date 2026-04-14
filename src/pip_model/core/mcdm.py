from __future__ import annotations

import numpy as np


def topsis(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = np.array(matrix, dtype=float)
    weights = np.array(weights, dtype=float)

    denom = np.sqrt((matrix ** 2).sum(axis=0))
    denom = np.where(denom == 0, 1.0, denom)
    norm = matrix / denom

    weighted = norm * weights

    ideal_best = np.max(weighted, axis=0)
    ideal_worst = np.min(weighted, axis=0)

    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    return dist_worst / (dist_best + dist_worst + 1e-12)
