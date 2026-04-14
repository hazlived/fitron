from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int = 4,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def predict(model: DecisionTreeClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)
