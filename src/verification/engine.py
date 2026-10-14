"""Verification Engine.

Cross-validates numerical computations using analytic linear algebra properties,
hand-calculated benchmark test matrices, and reference standard libraries (scipy.linalg).

Key Capabilities:
1. Hardcoded Benchmark Test Matrix A_test (Masterplan Section 4).
2. Verification of Benchmark properties:
   - Row 3 == 2 * Row 1
   - Rank r = 2, Pivot Columns = [0, 1] (Column 1, Column 2)
   - RREF == [[1, 0, 0.75], [0, 1, 0], [0, 0, 0]]
   - Inner Product <a_1, a_3> == 50.0
   - Cosine Similarity S_{1, 3} == 1.0
3. Automated Theoretical Identity Checklists:
   - Trace Identity: Trace(C) == sum(lambda_i)
   - Determinant Identity: det(C) == prod(lambda_i)
   - Orthogonality: v_i^T * v_j == 0 for all i != j (and V^T * V == I)
   - Eigenvalue Equation: C * v_k == lambda_k * v_k
   - Projection Idempotence: P^2 == P and Symmetry: P^T == P
4. Direct comparison against reference scipy.linalg and scipy.spatial routines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import scipy.linalg
import scipy.spatial.distance

from src.linear_algebra.engine import LinearAlgebraEngine, SpectralResult, ProjectionResult


# Masterplan Benchmark Matrix A_test (m=3, n=3)
BENCHMARK_MATRIX_A_TEST: np.ndarray = np.array([
    [4.0, 0.0, 3.0],
    [0.0, 5.0, 0.0],
    [8.0, 0.0, 6.0],
], dtype=np.float64)

BENCHMARK_EXPECTED_RREF: np.ndarray = np.array([
    [1.0, 0.0, 0.75],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
], dtype=np.float64)


@dataclass
class BenchmarkReport:
    """Detailed validation report for the Masterplan benchmark matrix A_test.

    Attributes:
        row3_is_2x_row1: Boolean flag indicating Row 3 == 2 * Row 1.
        row3_difference_norm: Maximum element difference ||Row 3 - 2 * Row 1||_inf.
        rank: Calculated rank (expected 2).
        pivot_columns: Pivot columns (expected [0, 1]).
        free_columns: Free columns (expected [2]).
        rref_matches: Boolean flag indicating calculated RREF == expected RREF.
        rref_max_diff: Maximum difference between calculated and expected RREF.
        inner_product_1_3: Dot product <a_1, a_3> (expected 50.0).
        cosine_similarity_1_3: Cosine similarity S_{1, 3} (expected 1.0).
        row_norms: Array of Euclidean norms [||a_1||_2, ||a_2||_2, ||a_3||_2] (expected [5, 5, 10]).
        all_passed: True if all benchmark tests pass within tolerance.
    """
    row3_is_2x_row1: bool
    row3_difference_norm: float
    rank: int
    pivot_columns: List[int]
    free_columns: List[int]
    rref_matches: bool
    rref_max_diff: float
    inner_product_1_3: float
    cosine_similarity_1_3: float
    row_norms: np.ndarray
    all_passed: bool


@dataclass
class IdentityCheckResult:
    """Validation report for mathematical linear algebra identities on covariance & PCA.

    Attributes:
        trace_identity_passed: Trace(C) == sum(lambda_i).
        trace_difference: |Trace(C) - sum(lambda_i)|.
        det_identity_passed: det(C) == prod(lambda_i).
        det_difference: |det(C) - prod(lambda_i)|.
        orthogonality_passed: v_i^T * v_j == 0 for all i != j and ||v_i|| == 1.
        max_orthogonality_error: max |V^T * V - I|.
        eigenvalue_equation_passed: C * v_k == lambda_k * v_k for all k.
        max_eigenvalue_residual: max ||C * v_k - lambda_k * v_k||_2.
        projection_idempotence_passed: P^2 == P.
        projection_idempotence_error: max |P^2 - P|.
        projection_symmetry_passed: P^T == P.
        all_identities_passed: True if all theoretical identities hold.
    """
    trace_identity_passed: bool
    trace_difference: float
    det_identity_passed: bool
    det_difference: float
    orthogonality_passed: bool
    max_orthogonality_error: float
    eigenvalue_equation_passed: bool
    max_eigenvalue_residual: float
    projection_idempotence_passed: bool
    projection_idempotence_error: float
    projection_symmetry_passed: bool
    all_identities_passed: bool


@dataclass
class ScipyComparisonResult:
    """Benchmark comparing custom numerical algorithms against scipy.linalg reference routines.

    Attributes:
        rank_matches: Custom rank matches np.linalg.matrix_rank.
        custom_rank: Rank computed by Gaussian elimination.
        scipy_rank: Rank computed by scipy/numpy SVD.
        eigenvalues_max_diff: Maximum absolute difference with scipy.linalg.eigh.
        eigenvalue_comparison_passed: Eigenvalues match within tolerance.
        projection_subspace_diff: Difference between projection operators ||P_custom - P_scipy||_F.
        subspace_comparison_passed: Subspace projection operator matches reference.
        cosine_similarity_max_diff: Maximum difference with scipy.spatial.distance.cosine.
        cosine_comparison_passed: Cosine similarity matches reference.
        nullity_matches: Dimension of null space matches scipy.linalg.null_space.
        all_passed: True if all comparisons match within numerical tolerance.
    """
    rank_matches: bool
    custom_rank: int
    scipy_rank: int
    eigenvalues_max_diff: float
    eigenvalue_comparison_passed: bool
    projection_subspace_diff: float
    subspace_comparison_passed: bool
    cosine_similarity_max_diff: float
    cosine_comparison_passed: bool
    nullity_matches: bool
    all_passed: bool


class VerificationEngine:
    """Comprehensive test and mathematical verification engine."""

    def __init__(self, tolerance: float = 1e-6) -> None:
        """Initializes the VerificationEngine.

        Args:
            tolerance: Numerical comparison tolerance.
        """
        self.tolerance = tolerance
        self.la_engine = LinearAlgebraEngine(tolerance=1e-10)

    @staticmethod
    def get_benchmark_matrix() -> np.ndarray:
        """Returns a fresh copy of the Masterplan benchmark matrix A_test."""
        return BENCHMARK_MATRIX_A_TEST.copy()

    def verify_benchmark_properties(self, tol: Optional[float] = None) -> BenchmarkReport:
        """Executes full mathematical property verification on the benchmark matrix A_test.

        Verifies:
        1. Row 3 is equal to 2 * Row 1: A[2] == 2 * A[0]
        2. Rank r = 2
        3. Pivot columns are Column 1 and Column 2 (0-indexed [0, 1])
        4. RREF matches [[1, 0, 0.75], [0, 1, 0], [0, 0, 0]]
        5. Inner Product <a_1, a_3> is exactly 50.0
        6. Cosine Similarity S_{1, 3} is exactly 1.0
        7. Norms are [5.0, 5.0, 10.0]

        Returns:
            BenchmarkReport dataclass.
        """
        eps = tol if tol is not None else self.tolerance
        A = self.get_benchmark_matrix()

        # 1. Row 3 == 2 * Row 1
        row1 = A[0]
        row3 = A[2]
        diff_row = np.abs(row3 - 2.0 * row1)
        max_diff_row = float(np.max(diff_row))
        row3_is_2x = max_diff_row <= eps

        # 2. RREF, Rank & Pivot Columns
        rref_res = self.la_engine.compute_rref(A)
        rank = rref_res.rank
        pivot_cols = rref_res.pivot_columns
        free_cols = rref_res.free_columns

        # Expected RREF
        diff_rref = np.abs(rref_res.rref_matrix - BENCHMARK_EXPECTED_RREF)
        rref_max_diff = float(np.max(diff_rref))
        rref_matches = rref_max_diff <= eps

        # 3. Inner Product & Similarity
        sim_res = self.la_engine.compute_similarity_engine(A)
        # Inner product between Customer 1 and Customer 3: G[0, 2]
        ip_1_3 = float(sim_res.gram_matrix[0, 2])
        cos_1_3 = float(sim_res.cosine_similarity_matrix[0, 2])
        norms = sim_res.vector_norms

        # Check conditions
        rank_correct = (rank == 2) and (pivot_cols == [0, 1]) and (free_cols == [2])
        ip_correct = abs(ip_1_3 - 50.0) <= eps
        cos_correct = abs(cos_1_3 - 1.0) <= eps
        norms_correct = np.allclose(norms, [5.0, 5.0, 10.0], atol=eps)

        all_passed = bool(
            row3_is_2x
            and rank_correct
            and rref_matches
            and ip_correct
            and cos_correct
            and norms_correct
        )

        return BenchmarkReport(
            row3_is_2x_row1=row3_is_2x,
            row3_difference_norm=max_diff_row,
            rank=rank,
            pivot_columns=pivot_cols,
            free_columns=free_cols,
            rref_matches=rref_matches,
            rref_max_diff=rref_max_diff,
            inner_product_1_3=ip_1_3,
            cosine_similarity_1_3=cos_1_3,
            row_norms=norms,
            all_passed=all_passed,
        )

    def verify_theoretical_identities(
        self,
        matrix: np.ndarray,
        tol: Optional[float] = None,
    ) -> IdentityCheckResult:
        """Executes automated verification checklist for fundamental theoretical identities:

        1. Trace Identity: Trace(C) == sum(lambda_i)
        2. Determinant Identity: det(C) == prod(lambda_i)
        3. Orthonormality / Orthogonality: v_i^T * v_j == 0 for all i != j and ||v_i|| == 1
        4. Eigenvalue Equation: C * v_k == lambda_k * v_k
        5. Projection Idempotence: P^2 == P
        6. Projection Symmetry: P^T == P

        Args:
            matrix: Input customer matrix A in R^{m x n}.
            tol: Tolerance threshold.

        Returns:
            IdentityCheckResult dataclass.
        """
        eps = tol if tol is not None else self.tolerance
        m, n = matrix.shape

        spec = self.la_engine.compute_spectral_decomposition(matrix, method="qr")
        C = spec.covariance_matrix
        eigenvalues = spec.eigenvalues
        eigenvectors = spec.eigenvectors

        # 1. Trace Identity: Trace(C) == sum(lambda_i)
        tr_C = float(np.trace(C))
        sum_l = float(np.sum(eigenvalues))
        tr_diff = abs(tr_C - sum_l)
        tr_passed = tr_diff <= eps

        # 2. Determinant Identity: det(C) == prod(lambda_i)
        det_C = float(np.linalg.det(C))
        prod_l = float(np.prod(eigenvalues))
        det_diff = abs(det_C - prod_l)
        # Relative or absolute check
        det_passed = det_diff <= max(eps, eps * max(abs(det_C), abs(prod_l)))

        # 3. Orthogonality & Orthonormality: V^T * V == I_n
        Vt_V = np.matmul(eigenvectors.T, eigenvectors)
        ortho_err = float(np.max(np.abs(Vt_V - np.eye(n))))
        ortho_passed = ortho_err <= eps

        # 4. Eigenvalue Equation: C * v_k == lambda_k * v_k
        residuals = []
        for k in range(n):
            v_k = eigenvectors[:, k]
            lam_k = eigenvalues[k]
            res_vec = np.matmul(C, v_k) - lam_k * v_k
            residuals.append(float(np.linalg.norm(res_vec)))
        max_eigen_residual = max(residuals)
        eigen_passed = max_eigen_residual <= eps

        # 5. Projection Operator Identities (for k = min(3, n))
        k_val = min(3, n)
        proj = self.la_engine.compute_orthogonal_projection(spec, k=k_val)
        P = proj.projection_matrix

        # Idempotence: P^2 == P
        P_sq = np.matmul(P, P)
        idem_err = float(np.max(np.abs(P_sq - P)))
        idem_passed = idem_err <= eps

        # Symmetry: P^T == P
        symm_err = float(np.max(np.abs(P - P.T)))
        symm_passed = symm_err <= eps

        all_identities_passed = bool(
            tr_passed
            and det_passed
            and ortho_passed
            and eigen_passed
            and idem_passed
            and symm_passed
        )

        return IdentityCheckResult(
            trace_identity_passed=tr_passed,
            trace_difference=tr_diff,
            det_identity_passed=det_passed,
            det_difference=det_diff,
            orthogonality_passed=ortho_passed,
            max_orthogonality_error=ortho_err,
            eigenvalue_equation_passed=eigen_passed,
            max_eigenvalue_residual=max_eigen_residual,
            projection_idempotence_passed=idem_passed,
            projection_idempotence_error=idem_err,
            projection_symmetry_passed=symm_passed,
            all_identities_passed=all_identities_passed,
        )

    def compare_against_scipy(
        self,
        matrix: np.ndarray,
        tol: Optional[float] = None,
    ) -> ScipyComparisonResult:
        """Compares custom numerical linear algebra routines directly against scipy.linalg references:

        1. Matrix Rank: Custom Gaussian Elimination rank vs np.linalg.matrix_rank / SVD rank.
        2. Eigenvalues: Custom eigenvalues vs scipy.linalg.eigh(C).
        3. Subspace Projection: Custom P = V_k * V_k^T vs Scipy P_scipy = V_{scipy} * V_{scipy}^T.
        4. Cosine Similarity: Custom S_{ij} vs (1 - scipy.spatial.distance.cosine(a_i, a_j)).
        5. Null Space Dimension: Custom nullity vs scipy.linalg.null_space(A).shape[1].

        Args:
            matrix: Input matrix A in R^{m x n}.
            tol: Tolerance.

        Returns:
            ScipyComparisonResult dataclass.
        """
        eps = tol if tol is not None else self.tolerance
        m, n = matrix.shape

        # 1. Rank Comparison
        custom_rank, custom_nullity = self.la_engine.determine_rank_and_nullity(matrix)
        scipy_rank = int(np.linalg.matrix_rank(matrix))
        rank_matches = (custom_rank == scipy_rank)

        # Nullity comparison
        scipy_null_basis = scipy.linalg.null_space(matrix)
        scipy_nullity = scipy_null_basis.shape[1]
        nullity_matches = (custom_nullity == scipy_nullity)

        # 2. Covariance & Eigenvalues comparison
        spec = self.la_engine.compute_spectral_decomposition(matrix, method="qr")
        C = spec.covariance_matrix
        custom_evals = spec.eigenvalues
        custom_evecs = spec.eigenvectors

        # Reference scipy.linalg.eigh (returns in ascending order, reverse for descending)
        scipy_evals_asc, scipy_evecs_asc = scipy.linalg.eigh(C)
        scipy_evals = scipy_evals_asc[::-1].copy()
        scipy_evecs = scipy_evecs_asc[:, ::-1].copy()

        # Compare eigenvalues
        evals_diff = float(np.max(np.abs(custom_evals - scipy_evals)))
        evals_passed = evals_diff <= 1e-5

        # 3. Subspace Projection Operator Comparison
        # To avoid sign ambiguity (+v vs -v), compare the projection matrices P = V_k * V_k^T
        k_val = min(3, custom_rank if custom_rank > 0 else 1)
        P_custom = np.matmul(custom_evecs[:, :k_val], custom_evecs[:, :k_val].T)
        P_scipy = np.matmul(scipy_evecs[:, :k_val], scipy_evecs[:, :k_val].T)
        proj_diff = float(np.linalg.norm(P_custom - P_scipy, ord="fro"))
        subspace_passed = proj_diff <= 1e-5

        # 4. Cosine Similarity Comparison
        sim_res = self.la_engine.compute_similarity_engine(matrix)
        S_custom = sim_res.cosine_similarity_matrix

        # Build reference cosine similarity from scipy.spatial.distance.cosine
        S_scipy = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            norm_i = np.linalg.norm(matrix[i])
            if norm_i > 1e-12:
                S_scipy[i, i] = 1.0
            for j in range(i + 1, m):
                norm_j = np.linalg.norm(matrix[j])
                if norm_i > 1e-12 and norm_j > 1e-12:
                    cos_dist = scipy.spatial.distance.cosine(matrix[i], matrix[j])
                    cos_sim = 1.0 - cos_dist
                    S_scipy[i, j] = cos_sim
                    S_scipy[j, i] = cos_sim

        cos_diff = float(np.max(np.abs(S_custom - S_scipy)))
        cos_passed = cos_diff <= eps

        all_passed = bool(
            rank_matches
            and nullity_matches
            and evals_passed
            and subspace_passed
            and cos_passed
        )

        return ScipyComparisonResult(
            rank_matches=rank_matches,
            custom_rank=custom_rank,
            scipy_rank=scipy_rank,
            eigenvalues_max_diff=evals_diff,
            eigenvalue_comparison_passed=evals_passed,
            projection_subspace_diff=proj_diff,
            subspace_comparison_passed=subspace_passed,
            cosine_similarity_max_diff=cos_diff,
            cosine_comparison_passed=cos_passed,
            nullity_matches=nullity_matches,
            all_passed=all_passed,
        )
