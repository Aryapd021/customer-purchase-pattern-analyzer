"""Unit and Mathematical Verification Tests for PCA, Covariance, Spectral Decomposition,

and Subspace Orthogonal Projection.
"""

import numpy as np
import pytest

from src.linear_algebra.engine import (
    LinearAlgebraEngine,
    SpectralResult,
    ProjectionResult,
)
from src.ingestion.synthetic_generator import SyntheticCustomerDataGenerator


class TestPCACovarianceSpectral:
    """Mathematical validation for Covariance formulation and Spectral Decomposition."""

    @pytest.fixture
    def engine(self) -> LinearAlgebraEngine:
        return LinearAlgebraEngine(tolerance=1e-10)

    @pytest.fixture
    def benchmark_matrix(self) -> np.ndarray:
        """Hand-calculated benchmark matrix from Masterplan §4:
        A_test = [[4, 0, 3],
                  [0, 5, 0],
                  [8, 0, 6]]
        m = 3, n = 3
        """
        return np.array([
            [4.0, 0.0, 3.0],
            [0.0, 5.0, 0.0],
            [8.0, 0.0, 6.0],
        ], dtype=np.float64)

    def test_mean_centering_and_covariance_benchmark(self, engine, benchmark_matrix):
        """Verifies column mean centering and sample covariance against exact analytical values:
        mu = [4.0, 5/3, 3.0]
        X = [[ 0, -5/3,  0],
             [-4, 10/3, -3],
             [ 4, -5/3,  3]]
        C = [[ 16.0,  -10.0,  12.0],
             [-10.0,   25/3,  -7.5],
             [ 12.0,   -7.5,   9.0]]
        """
        X, mu = engine.compute_mean_centered_matrix(benchmark_matrix)
        expected_mu = np.array([4.0, 5.0 / 3.0, 3.0])
        expected_X = np.array([
            [0.0, -5.0 / 3.0, 0.0],
            [-4.0, 10.0 / 3.0, -3.0],
            [4.0, -5.0 / 3.0, 3.0],
        ])

        np.testing.assert_allclose(mu, expected_mu, atol=1e-12)
        np.testing.assert_allclose(X, expected_X, atol=1e-12)
        # Column sums must be 0
        np.testing.assert_allclose(np.mean(X, axis=0), [0.0, 0.0, 0.0], atol=1e-12)

        C = engine.compute_covariance_matrix(X)
        expected_C = np.array([
            [16.0, -10.0, 12.0],
            [-10.0, 25.0 / 3.0, -7.5],
            [12.0, -7.5, 9.0],
        ])

        np.testing.assert_allclose(C, expected_C, atol=1e-12)
        # Symmetry check
        np.testing.assert_allclose(C, C.T, atol=1e-12)

    def test_eigen_solvers_and_theoretical_identities(self, engine, benchmark_matrix):
        """Validates Jacobi and QR algorithms against theoretical identities from Masterplan §4.2:
        1. Trace(C) == sum(lambda_i)
        2. det(C) == prod(lambda_i)
        3. V^T * V == I
        4. C * v_i == lambda_i * v_i
        """
        for method in ["jacobi", "qr", "characteristic"]:
            spec = engine.compute_spectral_decomposition(benchmark_matrix, method=method)

            eigenvalues = spec.eigenvalues
            eigenvectors = spec.eigenvectors
            C = spec.covariance_matrix

            # 1. Sorted in descending order
            assert np.all(np.diff(eigenvalues) <= 1e-10), f"Eigenvalues not sorted in descending order for {method}"
            assert len(eigenvalues) == 3

            # 2. Trace Identity: Trace(C) == sum(lambda_i)
            # Trace(C) = 16 + 25/3 + 9 = 100/3 ~ 33.333333
            expected_trace = 100.0 / 3.0
            np.testing.assert_allclose(np.sum(eigenvalues), expected_trace, atol=1e-7)
            np.testing.assert_allclose(np.trace(C), expected_trace, atol=1e-7)

            # 3. Determinant Identity: det(C) == prod(lambda_i)
            # Since rank(C) = 2, det(C) == 0, and smallest eigenvalue must be 0
            np.testing.assert_allclose(np.prod(eigenvalues), 0.0, atol=1e-7)
            assert eigenvalues[2] < 1e-6, f"Smallest eigenvalue should be 0, got {eigenvalues[2]}"

            # 4. Orthonormality: V^T * V == I_3
            np.testing.assert_allclose(np.matmul(eigenvectors.T, eigenvectors), np.eye(3), atol=1e-7)

            # 5. Eigenpair equation: C * v_i == lambda_i * v_i
            for i in range(3):
                v_i = eigenvectors[:, i]
                lam_i = eigenvalues[i]
                res = np.matmul(C, v_i) - lam_i * v_i
                np.testing.assert_allclose(res, np.zeros(3), atol=1e-7)

            # 6. Verification helper check
            verif = engine.verify_spectral_identities(C, eigenvalues, eigenvectors, tol=1e-6)
            assert verif["trace_match"]
            assert verif["orthonormal"]
            assert verif["eigenpair_valid"]

    def test_cumulative_variance_ratio(self, engine, benchmark_matrix):
        """Verifies cumulative variance ratios R(k) = sum_{1..k} lambda_i / sum_{1..n} lambda_j."""
        spec = engine.compute_spectral_decomposition(benchmark_matrix)

        var_ratios = spec.variance_ratios
        cum_ratios = spec.cumulative_variance_ratios

        # Non-negative and sums to 1.0
        assert np.all(var_ratios >= -1e-12)
        np.testing.assert_allclose(np.sum(var_ratios), 1.0, atol=1e-12)
        np.testing.assert_allclose(cum_ratios[-1], 1.0, atol=1e-12)

        # Since lambda_3 = 0, first two components capture 100% of variance!
        np.testing.assert_allclose(cum_ratios[1], 1.0, atol=1e-6)
        assert var_ratios[2] < 1e-6


class TestSubspaceOrthogonalProjection:
    """Mathematical validation for Orthogonal Subspace Projection P and Coordinates Y."""

    @pytest.fixture
    def engine(self) -> LinearAlgebraEngine:
        return LinearAlgebraEngine(tolerance=1e-10)

    @pytest.fixture
    def benchmark_matrix(self) -> np.ndarray:
        return np.array([
            [4.0, 0.0, 3.0],
            [0.0, 5.0, 0.0],
            [8.0, 0.0, 6.0],
        ], dtype=np.float64)

    def test_orthogonal_projection_matrix_properties(self, engine, benchmark_matrix):
        """Verifies mathematical properties of orthogonal projection operator P = V_k * V_k^T:
        1. Symmetry: P^T == P
        2. Idempotence: P^2 == P
        3. Rank(P) == k
        4. Trace(P) == k
        """
        for k in [1, 2, 3]:
            proj = engine.compute_orthogonal_projection(benchmark_matrix, k=k)
            P = proj.projection_matrix

            assert P.shape == (3, 3)

            # 1. Symmetry
            np.testing.assert_allclose(P, P.T, atol=1e-10)

            # 2. Idempotence: P * P == P
            np.testing.assert_allclose(np.matmul(P, P), P, atol=1e-10)

            # 3. Rank(P) == k
            assert int(np.linalg.matrix_rank(P)) == k

            # 4. Trace(P) == k
            np.testing.assert_allclose(np.trace(P), float(k), atol=1e-10)

    def test_low_dimensional_customer_coordinates(self, engine, benchmark_matrix):
        """Verifies low-dimensional coordinates Y = X * V_k in R^{m x k}."""
        m, n = benchmark_matrix.shape
        for k in [1, 2]:
            proj = engine.compute_orthogonal_projection(benchmark_matrix, k=k)
            Y = proj.projected_coordinates

            assert Y.shape == (m, k)
            # Reconstructed centered matrix X_hat = Y * V_k^T
            X_hat = proj.reconstructed_centered
            assert X_hat.shape == (m, n)

            # Reconstructed original matrix A_hat = X_hat + 1 * mu^T
            A_hat = proj.reconstructed_original
            assert A_hat.shape == (m, n)

    def test_information_loss_identity(self, engine, benchmark_matrix):
        """Validates the fundamental PCA identity:
        Reconstruction Error L(k) = ||X - X_hat||_F^2 / ||X||_F^2 == 1 - R(k).
        """
        spec = engine.compute_spectral_decomposition(benchmark_matrix)
        for k in [1, 2, 3]:
            proj = engine.compute_orthogonal_projection(spec, k=k)
            expected_loss = 1.0 - spec.cumulative_variance_ratios[k - 1]
            np.testing.assert_allclose(proj.reconstruction_error, expected_loss, atol=1e-6)

        # For k=2, since lambda_3 = 0, reconstruction error must be 0!
        proj_k2 = engine.compute_orthogonal_projection(spec, k=2)
        np.testing.assert_allclose(proj_k2.reconstruction_error, 0.0, atol=1e-6)
        # Original matrix is perfectly reconstructed from k=2 components!
        np.testing.assert_allclose(proj_k2.reconstructed_original, benchmark_matrix, atol=1e-6)

    def test_integration_synthetic_pca_projection(self, engine):
        """End-to-end integration: generate synthetic dataset with rank r=3 in R^{60 x 8}.
        Verify that k=3 preserves 100% of variance and exact reconstruction.
        """
        generator = SyntheticCustomerDataGenerator(random_state=42)
        m, n, r = 60, 8, 3
        synth = generator.generate(n_customers=m, n_categories=n, rank=r, noise_level=0.0)

        spec = engine.compute_spectral_decomposition(synth.matrix)
        assert len(spec.eigenvalues) == n

        # Since matrix rank is 3, covariance rank is <= 3, eigenvalues 4..8 should be effectively 0
        for idx in range(r, n):
            assert spec.eigenvalues[idx] < 1e-6, f"Eigenvalue {idx} should be 0, got {spec.eigenvalues[idx]}"

        # Cumulative variance at k=3 must be 1.0 (100%)
        np.testing.assert_allclose(spec.cumulative_variance_ratios[r - 1], 1.0, atol=1e-6)

        # Projection onto k=3 subspace
        proj = engine.compute_orthogonal_projection(spec, k=r)
        assert proj.projected_coordinates.shape == (m, r)
        assert proj.projection_matrix.shape == (n, n)
        np.testing.assert_allclose(proj.reconstruction_error, 0.0, atol=1e-6)
        np.testing.assert_allclose(proj.reconstructed_original, synth.matrix, atol=1e-6)
