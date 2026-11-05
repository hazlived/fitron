from __future__ import annotations

import numpy as np

from .memory import Memory
from .optimizer import initialize_weights, project_simplex


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Project weights onto the probability simplex."""
    return project_simplex(weights)


def blend_with_memory(
    weights: np.ndarray,
    memory: Memory,
    memory_factor: float = 0.4,
) -> np.ndarray:
    """Convex combination with memory state, projected onto the probability simplex."""
    best = memory.get_best()
    if best is None:
        return project_simplex(weights)
    mixed = memory_factor * best + (1.0 - memory_factor) * weights
    return project_simplex(mixed)


def perturb_weights(
    weights: np.ndarray,
    scale: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bounded exploration perturbation projected onto probability simplex."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, scale, size=len(weights))
    perturbed = weights + noise
    return project_simplex(perturbed)


__all__ = [
    "normalize_weights",
    "initialize_weights",
    "blend_with_memory",
    "perturb_weights",
    "project_simplex",
]
