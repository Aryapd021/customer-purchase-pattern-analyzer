"""Unit Tests for VerificationEngine.

Cross-validates all numerical linear algebra computations using analytic properties,
hand-calculated matrices, and standard reference routines from scipy.linalg.
"""

import numpy as np
import pytest

from src.verification.engine import (
    VerificationEngine,
    BENCHMARK_MATRIX_A_TEST,
    BENCHMARK_EXPECTED_RREF,
    BenchmarkReport,
    IdentityCheckResult,
    ScipyComparisonResult,
)
from src.linear_algebra.engine import LinearAlgebraEngine
from src.ingestion.synthetic_generator import SyntheticCustomerDataGenerator


class TestBenchmarkMatrixVerification:
    """Rigorous tests specifically on the hardcoded benchmark matrix A_test from Masterplan §4:

    A_test = [[4, 0, 3],
              [0, 5, 0],
              [8, 0, 6]]
    """

    @pytest.fixture
    def verifier(self) -> VerificationEngine:
        return VerificationEngine(tolerance=1e-10)

    @pytest.fixture
    def A_test(self) -> np.ndarray:
        return VerificationEngine.get_benchmark_matrix()

    def test_row3_is_two_times_row1_and_rank_and_pivot_columns(self, verifier, A_test):
        """Asserts that:
        1. Row 3 is equal to 2 * Row 1
        2. Rank r = 2
        3. Pivot columns are Column 1 and Column 2 (0-indexed [0, 1])
        """
        row1 = A_test[0]
        row3 = A_test[2]

        # 1. Assert Row 3 == 2 * Row 1
        np.testing.assert_allclose(row3, 2.0 * row1, atol=1e-12)

        # 2. Assert Rank r = 2
        engine = LinearAlgebraEngine(tolerance=1e-10)
        rref_res = engine.compute_rref(A_test)
        assert rref_res.rank == 2, f"Expected rank 2, got {rref_res.rank}"

        # 3. Assert Pivot Columns are Column 1 and Column 2 (0-indexed: [0, 1])
        assert rref_res.pivot_columns == [0, 1], f"Expected pivot columns [0, 1], got {rref_res.pivot_columns}"
        assert rref_res.free_columns == [2], f"Expected free column [2], got {rref_res.free_columns}"

        # VerificationEngine report check
        report = verifier.verify_benchmark_properties()
        assert report.row3_is_2x_row1
        assert report.rank == 2
        assert report.pivot_columns == [0, 1]

    def test_rref_exact_match(self, verifier, A_test):
        """Asserts the RREF output exactly matches:
        Row 1 [1, 0, 0.75], Row 2 [0, 1, 0], Row 3 [0, 0, 0]
        """
        engine = LinearAlgebraEngine(tolerance=1e-10)
        rref_res = engine.compute_rref(A_test)

        expected_rref = np.array([
            [1.0, 0.0, 0.75],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float64)

        np.testing.assert_allclose(rref_res.rref_matrix, expected_rref, atol=1e-12)

        # VerificationEngine report check
        report = verifier.verify_benchmark_properties()
        assert report.rref_matches
        assert report.rref_max_diff < 1e-12

    def test_inner_product_and_cosine_similarity_exact(self, verifier, A_test):
        """Asserts:
        1. Inner Product between Customer 1 and Customer 3 is exactly 50.0:
           <a_1, a_3> = 4*8 + 0*0 + 3*6 = 32 + 18 = 50.0
        2. Cosine Similarity between Customer 1 and Customer 3 is exactly 1.0:
           ||a_1||_2 = sqrt(4^2 + 3^2) = 5.0
           ||a_3||_2 = sqrt(8^2 + 6^2) = 10.0
           S_{1, 3} = 50.0 / (5.0 * 10.0) = 1.0
        """
        engine = LinearAlgebraEngine(tolerance=1e-10)

        # 1. Inner Product
        ip = engine.vector_dot_product(A_test[0], A_test[2])
        assert ip == pytest.approx(50.0, abs=1e-12)

        # 2. Vector Norms
        sim_res = engine.compute_similarity_engine(A_test)
        np.testing.assert_allclose(sim_res.vector_norms, [5.0, 5.0, 10.0], atol=1e-12)

        # 3. Cosine Similarity S_{1, 3}
        cos_sim = sim_res.cosine_similarity_matrix[0, 2]
        assert cos_sim == pytest.approx(1.0, abs=1e-12)

        # Also symmetric entry S_{3, 1}
        assert sim_res.cosine_similarity_matrix[2, 0] == pytest.approx(1.0, abs=1e-12)

        # VerificationEngine report check
        report = verifier.verify_benchmark_properties()
        assert report.inner_product_1_3 == pytest.approx(50.0, abs=1e-12)
        assert report.cosine_similarity_1_3 == pytest.approx(1.0, abs=1e-12)
        assert report.all_passed


class TestTheoreticalIdentitiesChecklist:
    """Automated testing checklists to verify fundamental theoretical identities."""

    @pytest.fixture
    def verifier(self) -> VerificationEngine:
        return VerificationEngine(tolerance=1e-6)

    def test_trace_and_determinant_identities(self, verifier):
        """Verifies:
        1. Trace(C) = sum_{i=1}^n lambda_i
        2. det(C) = prod_{i=1}^n lambda_i
        Tested on multiple matrices (benchmark, synthetic full rank, and low rank).
        """
        # Benchmark matrix (rank 2, det = 0)
        A_bench = VerificationEngine.get_benchmark_matrix()
        res_bench = verifier.verify_theoretical_identities(A_bench)
        assert res_bench.trace_identity_passed
        assert res_bench.trace_difference < 1e-6
        assert res_bench.det_identity_passed
        assert res_bench.det_difference < 1e-6

        # Synthetic dataset (rank 3, n = 6)
        generator = SyntheticCustomerDataGenerator(random_state=42)
        synth = generator.generate(n_customers=50, n_categories=6, rank=3)
        res_synth = verifier.verify_theoretical_identities(synth.matrix)
        assert res_synth.trace_identity_passed
        assert res_synth.det_identity_passed

    def test_eigenvector_orthogonality(self, verifier):
        """Verifies eigenvector orthogonality:
        v_i . v_j = 0 for all i != j (and V^T * V = I).
        """
        generator = SyntheticCustomerDataGenerator(random_state=77)
        synth = generator.generate(n_customers=40, n_categories=5, rank=3)

        res = verifier.verify_theoretical_identities(synth.matrix)
        assert res.orthogonality_passed
        assert res.max_orthogonality_error < 1e-6

    def test_eigenvalue_equation(self, verifier):
        """Verifies the eigenvalue equation holds true:
        C * v_k = lambda_k * v_k for all k = 1..n.
        """
        A_bench = VerificationEngine.get_benchmark_matrix()
        res = verifier.verify_theoretical_identities(A_bench)
        assert res.eigenvalue_equation_passed
        assert res.max_eigenvalue_residual < 1e-6

        # Also on synthetic matrix
        generator = SyntheticCustomerDataGenerator(random_state=123)
        synth = generator.generate(n_customers=60, n_categories=8, rank=3)
        res_synth = verifier.verify_theoretical_identities(synth.matrix)
        assert res_synth.eigenvalue_equation_passed
        assert res_synth.max_eigenvalue_residual < 1e-6

    def test_projection_operator_properties(self, verifier):
        """Verifies projection idempotence P^2 == P and symmetry P^T == P."""
        A_bench = VerificationEngine.get_benchmark_matrix()
        res = verifier.verify_theoretical_identities(A_bench)
        assert res.projection_idempotence_passed
        assert res.projection_symmetry_passed
        assert res.all_identities_passed


class TestScipyComparison:
    """Compares custom mathematical algorithm outputs against reference scipy.linalg routines."""

    @pytest.fixture
    def verifier(self) -> VerificationEngine:
        return VerificationEngine(tolerance=1e-5)

    def test_scipy_comparison_benchmark_matrix(self, verifier):
        """Compares custom outputs on A_test against scipy.linalg."""
        A_bench = VerificationEngine.get_benchmark_matrix()
        cmp_res = verifier.compare_against_scipy(A_bench)

        assert cmp_res.rank_matches, f"Custom rank {cmp_res.custom_rank} != scipy rank {cmp_res.scipy_rank}"
        assert cmp_res.nullity_matches
        assert cmp_res.eigenvalue_comparison_passed
        assert cmp_res.eigenvalues_max_diff < 1e-5
        assert cmp_res.subspace_comparison_passed
        assert cmp_res.projection_subspace_diff < 1e-5
        assert cmp_res.cosine_comparison_passed
        assert cmp_res.cosine_similarity_max_diff < 1e-10
        assert cmp_res.all_passed

    def test_scipy_comparison_synthetic_dataset(self, verifier):
        """Compares custom outputs on synthetic customer preference matrix against scipy.linalg."""
        generator = SyntheticCustomerDataGenerator(random_state=42)
        synth = generator.generate(n_customers=80, n_categories=10, rank=4, noise_level=0.0)

        cmp_res = verifier.compare_against_scipy(synth.matrix)
        assert cmp_res.rank_matches
        assert cmp_res.nullity_matches
        assert cmp_res.eigenvalue_comparison_passed
        assert cmp_res.eigenvalues_max_diff < 1e-5
        assert cmp_res.subspace_comparison_passed
        assert cmp_res.cosine_comparison_passed
        assert cmp_res.all_passed
