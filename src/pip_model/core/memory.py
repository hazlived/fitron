from __future__ import annotations

import numpy as np
from .optimizer import project_simplex


def compute_wasserstein_drift(
    current_scores: np.ndarray,
    reference_scores: np.ndarray,
) -> float:
    """Compute empirical 1-Wasserstein distance between current and reference score distributions."""
    curr = np.sort(np.asarray(current_scores, dtype=float))
    ref = np.sort(np.asarray(reference_scores, dtype=float))
    if len(curr) == 0 or len(ref) == 0:
        return 0.0

    # Quantile interpolation
    grid = np.linspace(0.01, 0.99, 50)
    q_curr = np.quantile(curr, grid)
    q_ref = np.quantile(ref, grid)
    return float(np.mean(np.abs(q_curr - q_ref)))


class Memory:
    """Adaptive Criteria Memory with A-EWMA and Concept Drift Awareness.

    Manages persistent criteria weights on the probability simplex Delta^{m-1},
    adapting to non-stationary distributions via Wasserstein-1 drift detection.
    """

    def __init__(
        self,
        max_history: int = 200,
        eta_min: float = 0.20,
        eta_max: float = 0.60,
        drift_threshold: float = 0.15,
    ) -> None:
        self.history: list[dict] = []
        self.best_weights: np.ndarray | None = None
        self.best_score: float = -np.inf
        self.max_history: int = max_history
        self.ema_improvement: float = 0.0

        self.eta_min = eta_min
        self.eta_max = eta_max
        self.drift_threshold = drift_threshold

        self.persistent_weights: np.ndarray | None = None
        self.reference_scores: np.ndarray | None = None
        self.selection_counts: dict[int, int] = {}

    def get_selection_count(self, idx: int) -> int:
        return self.selection_counts.get(idx, 0)

    def compute_recency_penalty(
        self,
        candidate_indices: list[int] | np.ndarray,
        lambda_rec: float = 0.10,
    ) -> np.ndarray:
        """Compute continuous information-theoretic recency penalty factors.

        penalty_factor = exp(-lambda_rec * n_t(i) / (mean_n + 1))
        Replaces arbitrary score docking (score *= 0.9 if visit > 5).
        """
        counts = np.array([self.get_selection_count(int(idx)) for idx in candidate_indices], dtype=float)
        mean_count = float(np.mean(counts)) if len(counts) > 0 else 0.0
        factors = np.exp(-lambda_rec * counts / (mean_count + 1.0))
        return factors

    def update(
        self,
        weights: np.ndarray,
        score: float,
        best_idx: int,
        batch_scores: np.ndarray | None = None,
    ) -> None:
        weights = project_simplex(weights)
        improvement = 0.0
        if self.history:
            improvement = score - self.history[-1]["score"]

        self.ema_improvement = 0.8 * self.ema_improvement + 0.2 * float(improvement)

        # Drift detection & Adaptive learning rate
        drift_distance = 0.0
        eta = self.eta_min

        if batch_scores is not None and len(batch_scores) > 0:
            batch_arr = np.asarray(batch_scores, dtype=float)
            if self.reference_scores is not None:
                drift_distance = compute_wasserstein_drift(batch_arr, self.reference_scores)
                # Sigmoidal scaling of learning rate based on drift magnitude
                drift_ratio = drift_distance / (self.drift_threshold + 1e-12)
                eta = self.eta_min + (self.eta_max - self.eta_min) / (1.0 + np.exp(-3.0 * (drift_ratio - 1.0)))
            else:
                self.reference_scores = batch_arr.copy()

        # Update persistent memory via A-EWMA
        if self.persistent_weights is None:
            self.persistent_weights = weights.copy()
        else:
            self.persistent_weights = project_simplex(
                (1.0 - eta) * self.persistent_weights + eta * weights
            )

        # Track candidate selection
        best_idx = int(best_idx)
        self.selection_counts[best_idx] = self.selection_counts.get(best_idx, 0) + 1

        self.history.append(
            {
                "weights": weights.copy(),
                "persistent_weights": self.persistent_weights.copy(),
                "score": float(score),
                "improvement": float(improvement),
                "ema_improvement": float(self.ema_improvement),
                "best_idx": best_idx,
                "drift_distance": float(drift_distance),
                "eta": float(eta),
            }
        )

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

        if score > self.best_score:
            self.best_score = float(score)
            self.best_weights = weights.copy()

    def get_best(self) -> np.ndarray | None:
        return self.best_weights

    def get_persistent(self) -> np.ndarray | None:
        return self.persistent_weights
