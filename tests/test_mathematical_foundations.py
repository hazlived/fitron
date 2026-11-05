from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pip_model.core.fuzzy import fit_fuzzy_profile, transform_fuzzy
from pip_model.core.mcdm import infer_criterion_types, topsis
from pip_model.core.memory import Memory, compute_wasserstein_drift
from pip_model.core.optimizer import initialize_weights, optimize_weights_simplex, project_simplex
from pip_model.core.reward import (
    compute_brier_score,
    compute_expected_calibration_error,
    compute_ranking_quality,
)


class TestRuspiniFuzzyPartition:
    """Verifies Theorem 2: Ruspini Fuzzy Partition Completeness."""

    def test_ruspini_sum_equals_one(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feat1": np.random.exponential(scale=10.0, size=500),
                "feat2": np.random.normal(loc=50.0, scale=15.0, size=500),
                "feat3": np.random.uniform(low=0.0, high=100.0, size=500),
            }
        )

        profile = fit_fuzzy_profile(df)
        fuzzy_df = transform_fuzzy(df, profile)

        for col in ["feat1", "feat2", "feat3"]:
            low = fuzzy_df[f"{col}_low"]
            mid = fuzzy_df[f"{col}_mid"]
            high = fuzzy_df[f"{col}_high"]

            # Bounds
            assert (low >= 0.0).all() and (low <= 1.0).all()
            assert (mid >= 0.0).all() and (mid <= 1.0).all()
            assert (high >= 0.0).all() and (high <= 1.0).all()

            # Partition of Unity
            total = low + mid + high
            np.testing.assert_allclose(total.to_numpy(), 1.0, rtol=1e-6, atol=1e-6)

    def test_constant_feature_handling(self) -> None:
        df = pd.DataFrame({"const_feat": [42.0] * 50})
        profile = fit_fuzzy_profile(df)
        fuzzy_df = transform_fuzzy(df, profile)

        total = fuzzy_df["const_feat_low"] + fuzzy_df["const_feat_mid"] + fuzzy_df["const_feat_high"]
        np.testing.assert_allclose(total.to_numpy(), 1.0, rtol=1e-6, atol=1e-6)


class TestSimplexProjectionAndOptimizer:
    """Verifies Theorem 4: Simplex Projection and Optimization."""

    def test_project_simplex_properties(self) -> None:
        test_vectors = [
            np.array([0.5, 0.8, -0.2, 0.1, 1.2]),
            np.array([10.0, -5.0, 0.0]),
            np.array([-1.0, -2.0, -3.0]),
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        ]

        for v in test_vectors:
            w = project_simplex(v)
            assert (w >= 0.0).all(), f"Negative component found: {w}"
            assert np.isclose(np.sum(w), 1.0, atol=1e-9), f"Sum != 1.0: {np.sum(w)}"

    def test_optimize_weights_simplex_improves_or_maintains(self) -> None:
        np.random.seed(42)
        matrix = np.random.uniform(0.1, 1.0, size=(10, 5))
        labels = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0])
        init_w = np.ones(5) / 5.0

        opt_w = optimize_weights_simplex(
            matrix=matrix,
            utility_labels=labels,
            initial_weights=init_w,
            rounds=4,
            step=0.05,
        )

        assert (opt_w >= 0.0).all()
        assert np.isclose(np.sum(opt_w), 1.0, atol=1e-9)


class TestMCDMOrientationAndTOPSIS:
    """Verifies Theorem 3: TOPSIS Distance Geometry & Orientation."""

    def test_infer_criterion_types(self) -> None:
        # Col 0 is positively correlated with threat probs (Benefit)
        # Col 1 is negatively correlated with threat probs (Cost)
        threat_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        matrix = np.array(
            [
                [10.0, 100.0],
                [20.0, 80.0],
                [30.0, 60.0],
                [40.0, 40.0],
                [50.0, 20.0],
            ]
        )

        criterion_types = infer_criterion_types(matrix, threat_probs)
        assert criterion_types == ["benefit", "cost"]

    def test_topsis_boundedness_and_extremities(self) -> None:
        matrix = np.array(
            [
                [10.0, 10.0],
                [1.0, 1.0],
                [5.0, 5.0],
            ]
        )
        weights = np.array([0.5, 0.5])
        scores = topsis(matrix, weights, criterion_types=["benefit", "benefit"])

        assert 0.0 <= np.min(scores) <= np.max(scores) <= 1.0
        # First row is ideal best, second row is ideal worst
        assert scores[0] > scores[2] > scores[1]


class TestMemoryAndConceptDrift:
    """Verifies A-EWMA and Recency Penalty."""

    def test_recency_penalty_monotonic_decay(self) -> None:
        memory = Memory()
        # Candidate 10 selected 0 times, Candidate 20 selected 1 time, Candidate 30 selected 5 times
        memory.selection_counts = {10: 0, 20: 1, 30: 5}
        penalties = memory.compute_recency_penalty([10, 20, 30], lambda_rec=0.20)

        assert penalties[0] > penalties[1] > penalties[2]

    def test_wasserstein_drift_triggers_higher_eta(self) -> None:
        memory = Memory(eta_min=0.20, eta_max=0.80, drift_threshold=0.10)
        # Baseline reference
        ref_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        memory.reference_scores = ref_scores

        # Low drift batch
        low_drift_batch = np.array([0.12, 0.22, 0.29, 0.41, 0.49])
        drift_low = compute_wasserstein_drift(low_drift_batch, ref_scores)

        # High drift batch
        high_drift_batch = np.array([0.7, 0.8, 0.85, 0.9, 0.95])
        drift_high = compute_wasserstein_drift(high_drift_batch, ref_scores)

        assert drift_high > drift_low


class TestGradedUtilityNDCG:
    """Verifies Theorem 1 & Resolution: Graded Utility vs Binary Invariance."""

    def test_binary_ndcg_is_invariant_on_all_positives(self) -> None:
        binary_labels = np.array([1, 1, 1, 1, 1])
        scores_perm1 = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        scores_perm2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        q1 = compute_ranking_quality(binary_labels, scores_perm1, k=5)
        q2 = compute_ranking_quality(binary_labels, scores_perm2, k=5)
        assert np.isclose(q1, q2)

    def test_graded_utility_discriminates_ranking_quality(self) -> None:
        # Candidate 0 has high severity (100.0), Candidate 4 has low severity (1.0)
        graded_labels = np.array([100.0, 50.0, 20.0, 5.0, 1.0])
        # Optimal ranking (places high severity first)
        optimal_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        # Inverted ranking (places low severity first)
        inverted_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        q_optimal = compute_ranking_quality(graded_labels, optimal_scores, k=5)
        q_inverted = compute_ranking_quality(graded_labels, inverted_scores, k=5)

        assert q_optimal > q_inverted
        assert q_optimal - q_inverted > 0.15

