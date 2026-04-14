from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Memory:
    history: list[dict] = field(default_factory=list)
    best_weights: np.ndarray | None = None
    best_score: float = float("-inf")

    def update(self, weights: np.ndarray, score: float, best_idx: int) -> None:
        improvement = 0.0
        if self.history:
            improvement = score - self.history[-1]["score"]

        self.history.append(
            {
                "weights": weights.copy(),
                "score": float(score),
                "improvement": float(improvement),
                "best_idx": int(best_idx),
            }
        )

        if score > self.best_score:
            self.best_score = float(score)
            self.best_weights = weights.copy()

    def get_best(self) -> np.ndarray | None:
        return self.best_weights
