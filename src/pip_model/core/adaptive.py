from __future__ import annotations

import numpy as np

from .memory import Memory


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(weights, 1e-6, None)
    return weights / np.sum(weights)


def initialize_weights(
    feature_importances: np.ndarray,
    alpha: float = 0.5,
    max_cap: float = 0.25,
) -> np.ndarray:
    tree_weights = np.clip(feature_importances, 0, max_cap)
    equal_weights = np.ones_like(tree_weights) / len(tree_weights)
    weights = alpha * tree_weights + (1 - alpha) * equal_weights
    return normalize_weights(weights)


def blend_with_memory(weights: np.ndarray, memory: Memory, memory_factor: float = 0.4) -> np.ndarray:
    best = memory.get_best()
    if best is None:
        return normalize_weights(weights)
    mixed = memory_factor * best + (1 - memory_factor) * weights
    return normalize_weights(mixed)


def perturb_weights(
    weights: np.ndarray,
    scale: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, scale, size=len(weights))
    return normalize_weights(np.abs(weights + noise))
