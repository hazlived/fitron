from __future__ import annotations

import numpy as np


def compute_accuracy(preds: np.ndarray, y_true: np.ndarray) -> float:
    correct = np.sum(preds == y_true)
    return float(correct / len(y_true))
