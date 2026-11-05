from __future__ import annotations

import numpy as np
from .mcdm import topsis
from .reward import compute_ranking_quality


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Exact Euclidean projection of vector v onto the probability simplex.

    Delta^{m-1} = {w in R^m : w_j >= 0, sum_{j=1}^m w_j = 1}

    Algorithm: Wang & Carreira-Perpinan (2013); Duchi et al. (2008).
    Time Complexity: O(m log m) due to sorting.
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("Vector v must be 1-dimensional.")
    m = len(v)
    if m == 0:
        return v
    if m == 1:
        return np.array([1.0], dtype=float)

    # Sort descending
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    indices = np.arange(1, m + 1)
    cond = u * indices > (cssv - 1.0)
    rho = np.nonzero(cond)[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)

    w = np.maximum(v - theta, 0.0)
    # Numerical safeguard to ensure exact sum of 1.0
    w_sum = np.sum(w)
    if w_sum > 0:
        w /= w_sum
    else:
        w = np.ones(m, dtype=float) / m
    return w


def initialize_weights(
    feature_importances: np.ndarray,
    alpha: float = 0.5,
    max_cap: float = 0.25,
) -> np.ndarray:
    """Initialize criteria weights from tree feature importances.

    Blends normalized, capped tree importances with the uniform prior and projects onto simplex.
    """
    feat_imp = np.asarray(feature_importances, dtype=float)
    m = len(feat_imp)
    if m == 0:
        return feat_imp

    tree_weights = np.clip(feat_imp, 0.0, max_cap)
    if np.sum(tree_weights) > 0:
        tree_weights /= np.sum(tree_weights)
    else:
        tree_weights = np.ones(m, dtype=float) / m

    uniform = np.ones(m, dtype=float) / m
    blended = alpha * tree_weights + (1.0 - alpha) * uniform
    return project_simplex(blended)


def optimize_weights_simplex(
    matrix: np.ndarray,
    utility_labels: np.ndarray,
    initial_weights: np.ndarray,
    criterion_types: list[str] | None = None,
    rounds: int = 3,
    step: float = 0.04,
    reg_lambda: float = 0.05,
    max_optimized_dims: int = 30,
) -> np.ndarray:
    """Projected coordinate optimization of criteria weights on the probability simplex.

    Maximizes:
        L(w) = Q_rank(w; matrix, utility_labels) - (reg_lambda / 2) * ||w - w_0||_2^2
    subject to:
        w in Delta^{m-1}
    """
    matrix = np.asarray(matrix, dtype=float)
    utility_labels = np.asarray(utility_labels, dtype=float)
    w_0 = project_simplex(initial_weights.copy())
    best_w = w_0.copy()

    if criterion_types is None:
        criterion_types = ["benefit"] * matrix.shape[1]

    def evaluate_objective(w: np.ndarray) -> float:
        topsis_scores = topsis(matrix, w, criterion_types=criterion_types)
        rank_q = compute_ranking_quality(utility_labels, topsis_scores)
        penalty = 0.5 * reg_lambda * float(np.sum((w - w_0) ** 2))
        return float(rank_q - penalty)

    best_score = evaluate_objective(best_w)
    m = len(best_w)

    if m > max_optimized_dims:
        optimize_indices = np.argsort(best_w)[-max_optimized_dims:]
    else:
        optimize_indices = np.arange(m)

    curr_step = float(step)
    for _ in range(rounds):
        improved = False
        for idx in optimize_indices:
            for direction in (-1.0, 1.0):
                trial = best_w.copy()
                trial[idx] += direction * curr_step
                # Project back to probability simplex
                trial = project_simplex(trial)
                trial_score = evaluate_objective(trial)

                if trial_score > best_score + 1e-6:
                    best_w = trial
                    best_score = trial_score
                    improved = True

        if not improved:
            curr_step *= 0.5

    return best_w

