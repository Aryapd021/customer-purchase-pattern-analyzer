"""Unit and Mathematical Verification Tests for AnalyticalProfilingModule."""

import numpy as np
import pytest

from src.linear_algebra.engine import LinearAlgebraEngine
from src.profiling.profiler import (
    AnalyticalProfilingModule,
    CustomerProfileCluster,
    ProfilingResult,
)
from src.ingestion.synthetic_generator import SyntheticCustomerDataGenerator


class TestAnalyticalProfilingModule:
    """Mathematical validation for customer subspace grouping, centroid re-projection, and archetype synthesis."""

    @pytest.fixture
    def synthetic_data(self):
        generator = SyntheticCustomerDataGenerator(random_state=42)
        return generator.generate(n_customers=60, n_categories=8, rank=3, noise_level=0.0)

    @pytest.fixture
    def linear_algebra_results(self, synthetic_data):
        engine = LinearAlgebraEngine()
        spec = engine.compute_spectral_decomposition(synthetic_data.matrix)
        proj = engine.compute_orthogonal_projection(spec, k=3)
        return spec, proj

    def test_euclidean_distance_matrix(self):
        """Validates Euclidean distance matrix computation in subspace:
        d(y_i, y_j) = ||y_i - y_j||_2
        """
        Y = np.array([
            [0.0, 0.0],
            [3.0, 4.0],
            [1.0, 1.0],
        ])
        D = AnalyticalProfilingModule.compute_euclidean_distance_matrix(Y)

        assert D.shape == (3, 3)
        # d(0, 1) = sqrt(3^2 + 4^2) = 5.0
        assert D[0, 1] == pytest.approx(5.0)
        assert D[1, 0] == pytest.approx(5.0)
        # d(i, i) = 0.0
        np.testing.assert_allclose(np.diag(D), [0.0, 0.0, 0.0])
        # Symmetry
        np.testing.assert_allclose(D, D.T)

    def test_reprojection_formula_exact(self):
        """Validates exact reconstruction identity:
        c_orig = c_proj * V_k^T + mu
        """
        V_k = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])  # shape (3, 2)
        mu = np.array([10.0, 20.0, 30.0])
        c_proj = np.array([[2.0, -5.0]])  # shape (1, 2)

        c_orig = AnalyticalProfilingModule.reproject_centroids_to_category_space(
            projected_centroids=c_proj,
            V_k=V_k,
            mean_vector=mu,
        )

        expected = np.array([[12.0, 15.0, 30.0]])
        np.testing.assert_allclose(c_orig, expected, atol=1e-10)

    def test_customer_clustering_and_archetype_profiling(self, synthetic_data, linear_algebra_results):
        """Tests full end-to-end customer profiling pipeline."""
        spec, proj = linear_algebra_results
        m, n = synthetic_data.matrix.shape
        K = 3

        profiler = AnalyticalProfilingModule(random_state=42)
        res = profiler.profile_customers(
            projected_coordinates=proj.projected_coordinates,
            top_eigenvectors=proj.top_eigenvectors,
            mean_vector=spec.mean_vector,
            n_clusters=K,
            customer_ids=synthetic_data.customer_ids,
            category_names=synthetic_data.category_names,
        )

        # 1. Structural checks
        assert len(res.clusters) == K
        assert res.labels.shape == (m,)
        assert res.projected_centroids.shape == (K, 3)
        assert res.category_centroids.shape == (K, n)
        assert res.total_inertia >= 0.0

        # 2. Complete customer partition: every customer assigned to exactly one cluster
        total_customers = sum(c.customer_count for c in res.clusters)
        assert total_customers == m
        total_pct = sum(c.percentage for c in res.clusters)
        assert total_pct == pytest.approx(100.0)

        # 3. Valid centroid re-projection check
        for g, cluster in enumerate(res.clusters):
            expected_c_orig = np.maximum(
                0.0, np.dot(cluster.projected_centroid, proj.top_eigenvectors.T) + spec.mean_vector
            )
            np.testing.assert_allclose(cluster.category_centroid, expected_c_orig, atol=1e-10)
            assert len(cluster.dominant_categories) == 3
            assert isinstance(cluster.archetype_name, str)
            assert len(cluster.archetype_name) > 0

        # 4. DataFrame export
        df = res.to_dataframe()
        assert len(df) == K
        assert "Archetype" in df.columns
        assert "Customers" in df.columns
