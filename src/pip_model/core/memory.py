from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Memory:
    history: list[dict] = field(default_factory=list)
    best_weights: np.ndarray | None = None
    best_score: float = float("-inf")
    max_history: int = 200
    ema_improvement: float = 0.0

    def update(self, weights: np.ndarray, score: float, best_idx: int) -> None:
        improvement = 0.0
        if self.history:
            improvement = score - self.history[-1]["score"]

        self.ema_improvement = 0.8 * self.ema_improvement + 0.2 * float(improvement)

        self.history.append(
            {
                "weights": weights.copy(),
                "score": float(score),
                "improvement": float(improvement),
                "ema_improvement": float(self.ema_improvement),
                "best_idx": int(best_idx),
            }
        )

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

        if score > self.best_score:
            self.best_score = float(score)
            self.best_weights = weights.copy()

    def get_best(self) -> np.ndarray | None:
        return self.best_weights
