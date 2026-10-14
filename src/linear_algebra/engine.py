"""Linear Algebra Engine.

Implements core numerical linear algebra operations from mathematical first principles
without relying on high-level black-box library abstractions (e.g., scipy.linalg.null_space,
numpy.linalg.matrix_rank, scipy.linalg.eig, sklearn.decomposition.PCA).

Key Capabilities:
1. Gaussian Elimination with Partial Pivoting to Reduced Row Echelon Form (RREF).
2. Matrix Rank determination r <= min(m, n) and Nullity calculation via the Rank-Nullity Theorem (n - r).
3. Fundamental Subspace Basis Extraction:
   - Row Space R(A): non-zero row vectors of RREF.
   - Column Space C(A): pivot column vectors of the original matrix A.
   - Null Space N(A): basis derived from exact free variable parameterization.
4. Vector Space Inner Products:
   - Pairwise dot products: <a_i, a_j> = sum_{k=1}^n A_{ik} * A_{jk}.
   - Gram Matrix: G = A * A^T in R^{m x m}.
   - Diagonal norm matrix: D in R^{m x m}.
   - Cosine Similarity Matrix: S = D^-1 * G * D^-1.
5. Principal Component Analysis & Spectral Decomposition:
   - Column mean vector mu = (1/m) * sum a_i.
   - Mean-centered matrix X = A - 1 * mu^T.
   - Sample Covariance Matrix C = (1/(m-1)) * X^T * X.
   - Eigenvalue & Eigenvector computation (Householder QR algorithm & Jacobi eigenvalue solver).
   - Sorted eigenpairs: lambda_1 >= lambda_2 >= ... >= lambda_n >= 0.
   - Cumulative variance ratio R(k) = (sum_{i=1}^k lambda_i) / (sum_{j=1}^n lambda_j).
   - Orthogonal projection operator P = V_k * V_k^T.
   - Low-dimensional customer coordinates Y = X * V_k in R^{m x k}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np


@dataclass
class RREFResult:
    """Encapsulates the output of Gaussian Elimination to Reduced Row Echelon Form.

    Attributes:
        rref_matrix: The transformed matrix R in R^{m x n} in Reduced Row Echelon Form.
        rank: Matrix rank r, representing the number of non-zero pivot rows.
        nullity: Dimension of the null space, n - r (Rank-Nullity Theorem).
        pivot_columns: 0-indexed column positions containing leading 1s (length r).
        free_columns: 0-indexed column positions corresponding to free variables (length n - r).
        row_swaps: Count of row permutations performed during partial pivoting.
    """
    rref_matrix: np.ndarray
    rank: int
    nullity: int
    pivot_columns: List[int]
    free_columns: List[int]
    row_swaps: int


@dataclass
class SubspaceBases:
    """Represents the extracted bases for the fundamental subspaces of matrix A.

    Attributes:
        row_space_basis: Orthonormal/echelon basis vectors for R(A), shape (r, n).
        column_space_basis: Original column vectors spanning C(A), shape (m, r).
        null_space_basis: Basis vectors spanning N(A) such that A * v = 0, shape (n, n - r).
        rank: Dimension of Row Space and Column Space, dim(R(A)) = dim(C(A)) = r.
        nullity: Dimension of Null Space, dim(N(A)) = n - r.
    """
    row_space_basis: np.ndarray
    column_space_basis: np.ndarray
    null_space_basis: np.ndarray
    rank: int
    nullity: int


@dataclass
class SimilarityResult:
    """Contains the pairwise vector inner products, norms, and cosine similarities.

    Attributes:
        gram_matrix: Unnormalized inner product matrix G = A * A^T in R^{m x m}.
        norm_matrix_d: Diagonal matrix of Euclidean norms D = diag(||a_1||_2, ..., ||a_m||_2).
        inv_norm_matrix_d: Inverse diagonal matrix D^-1.
        cosine_similarity_matrix: S = D^-1 * G * D^-1 in R^{m x m}.
        vector_norms: 1D array of customer vector norms, ||a_i||_2 for i=1..m.
    """
    gram_matrix: np.ndarray
    norm_matrix_d: np.ndarray
    inv_norm_matrix_d: np.ndarray
    cosine_similarity_matrix: np.ndarray
    vector_norms: np.ndarray


@dataclass
class SpectralResult:
    """Encapsulates the spectral decomposition and PCA of the covariance matrix.

    Attributes:
        eigenvalues: 1D array of eigenvalues sorted in descending order: lambda_1 >= ... >= lambda_n >= 0.
        eigenvectors: 2D array V in R^{n x n} where column i is eigenvector v_i.
        variance_ratios: 1D array of individual variance ratios lambda_i / sum(lambda).
        cumulative_variance_ratios: 1D array of cumulative variance explained R(k).
        covariance_matrix: Sample covariance matrix C = (1/(m-1)) * X^T * X in R^{n x n}.
        mean_vector: Column mean vector mu in R^n.
        mean_centered_matrix: Mean-centered matrix X = A - 1 * mu^T in R^{m x n}.
        trace_covariance: Trace of covariance matrix sum(C_ii) = sum(lambda_i).
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    variance_ratios: np.ndarray
    cumulative_variance_ratios: np.ndarray
    covariance_matrix: np.ndarray
    mean_vector: np.ndarray
    mean_centered_matrix: np.ndarray
    trace_covariance: float


@dataclass
class ProjectionResult:
    """Encapsulates low-dimensional orthogonal subspace projection.

    Attributes:
        projected_coordinates: Y = X * V_k in R^{m x k}, customer coordinates in reduced subspace.
        projection_matrix: P = V_k * V_k^T in R^{n x n}, orthogonal projection operator.
        top_eigenvectors: V_k in R^{n x k}, matrix of top k principal directions.
        top_eigenvalues: Array of top k eigenvalues of length k.
        k: Number of subspace dimensions retained.
        cumulative_variance_explained: Cumulative variance ratio R(k).
        reconstructed_centered: Reconstructed centered matrix X_hat = Y * V_k^T = X * P in R^{m x n}.
        reconstructed_original: Reconstructed original matrix A_hat = X_hat + 1 * mu^T in R^{m x n}.
        reconstruction_error: Relative Frobenius reconstruction error ||X - X_hat||_F^2 / ||X||_F^2.
    """
    projected_coordinates: np.ndarray
    projection_matrix: np.ndarray
    top_eigenvectors: np.ndarray
    top_eigenvalues: np.ndarray
    k: int
    cumulative_variance_explained: float
    reconstructed_centered: np.ndarray
    reconstructed_original: np.ndarray
    reconstruction_error: float


class LinearAlgebraEngine:
    """Custom numerical linear algebra solver implementing core operations from first principles."""

    def __init__(self, tolerance: float = 1e-10) -> None:
        """Initializes the linear algebra engine.

        Args:
            tolerance: Numerical zero threshold for floating-point comparisons.
        """
        self.tolerance = tolerance

    # =========================================================================
    # Stage 2: Gaussian Elimination, RREF, Rank & Subspaces
    # =========================================================================

    def compute_rref(
        self,
        matrix: np.ndarray,
        tol: Optional[float] = None,
    ) -> RREFResult:
        """Transforms a matrix A in R^{m x n} to Reduced Row Echelon Form (RREF)

        using Gaussian Elimination with Partial Pivoting.

        Mathematical properties enforced:
        1. All zero rows are at the bottom of the matrix.
        2. The leading entry (pivot) of each non-zero row is strictly 1.0.
        3. Each leading 1 is to the right of the leading 1 in the row above it.
        4. In any column containing a leading 1, all other entries are 0.0.

        Args:
            matrix: Input dense matrix A of shape (m, n).
            tol: Optional numerical threshold. If None, uses engine tolerance.

        Returns:
            RREFResult containing the RREF matrix, rank, nullity, and pivot indices.
        """
        eps = tol if tol is not None else self.tolerance
        m, n = matrix.shape

        R = matrix.astype(np.float64, copy=True)

        pivot_row = 0
        pivot_columns: List[int] = []
        row_swaps = 0

        for col in range(n):
            if pivot_row >= m:
                break

            # 1. Partial Pivoting: Select row with maximum absolute entry in column `col`
            max_val = 0.0
            max_row_idx = pivot_row
            for r in range(pivot_row, m):
                abs_val = abs(R[r, col])
                if abs_val > max_val:
                    max_val = abs_val
                    max_row_idx = r

            if max_val <= eps:
                continue

            # 2. Row Permutation
            if max_row_idx != pivot_row:
                R[[pivot_row, max_row_idx]] = R[[max_row_idx, pivot_row]]
                row_swaps += 1

            # 3. Pivot Normalization
            pivot_val = R[pivot_row, col]
            R[pivot_row, :] = R[pivot_row, :] / pivot_val
            R[pivot_row, col] = 1.0

            # 4. Elimination above and below pivot
            for r in range(m):
                if r != pivot_row:
                    factor = R[r, col]
                    if abs(factor) > eps:
                        R[r, :] -= factor * R[pivot_row, :]
                        R[r, col] = 0.0

            pivot_columns.append(col)
            pivot_row += 1

        R[np.abs(R) < eps] = 0.0

        rank = len(pivot_columns)
        nullity = n - rank
        free_columns = [c for c in range(n) if c not in pivot_columns]

        return RREFResult(
            rref_matrix=R,
            rank=rank,
            nullity=nullity,
            pivot_columns=pivot_columns,
            free_columns=free_columns,
            row_swaps=row_swaps,
        )

    def determine_rank_and_nullity(
        self,
        matrix: np.ndarray,
        tol: Optional[float] = None,
    ) -> Tuple[int, int]:
        """Calculates the matrix rank r <= min(m, n) and nullity (n - r).

        Args:
            matrix: Input matrix A in R^{m x n}.
            tol: Numerical threshold for pivot identification.

        Returns:
            Tuple of (rank, nullity) satisfying rank + nullity = n.
        """
        rref_res = self.compute_rref(matrix, tol=tol)
        return rref_res.rank, rref_res.nullity

    def extract_subspace_bases(
        self,
        matrix: np.ndarray,
        tol: Optional[float] = None,
    ) -> SubspaceBases:
        """Extracts exact bases for the fundamental subspaces of matrix A in R^{m x n}:

        1. Row Space R(A): The r non-zero rows of RREF(A) in R^n.
        2. Column Space C(A): The r pivot columns of original matrix A in R^m.
        3. Null Space N(A): The (n - r) vectors parameterizing the free variables
           such that A * v = 0 for all v in N(A).

        Args:
            matrix: Input matrix A in R^{m x n}.
            tol: Numerical threshold.

        Returns:
            SubspaceBases dataclass containing the fundamental subspace basis matrices.
        """
        eps = tol if tol is not None else self.tolerance
        m, n = matrix.shape

        rref_res = self.compute_rref(matrix, tol=eps)
        R = rref_res.rref_matrix
        rank = rref_res.rank
        nullity = rref_res.nullity
        pivot_cols = rref_res.pivot_columns
        free_cols = rref_res.free_columns

        # Row Space Basis R(A)
        if rank > 0:
            row_space_basis = R[:rank, :].copy()
        else:
            row_space_basis = np.empty((0, n), dtype=np.float64)

        # Column Space Basis C(A)
        if rank > 0:
            column_space_basis = matrix[:, pivot_cols].astype(np.float64, copy=True)
        else:
            column_space_basis = np.empty((m, 0), dtype=np.float64)

        # Null Space Basis N(A)
        if nullity > 0:
            null_space_basis = np.zeros((n, nullity), dtype=np.float64)

            for col_idx, free_col in enumerate(free_cols):
                null_space_basis[free_col, col_idx] = 1.0
                for row_idx, pivot_col in enumerate(pivot_cols):
                    null_space_basis[pivot_col, col_idx] = -R[row_idx, free_col]

            null_space_basis[np.abs(null_space_basis) < eps] = 0.0
        else:
            null_space_basis = np.empty((n, 0), dtype=np.float64)

        return SubspaceBases(
            row_space_basis=row_space_basis,
            column_space_basis=column_space_basis,
            null_space_basis=null_space_basis,
            rank=rank,
            nullity=nullity,
        )

    # =========================================================================
    # Stage 3: Vector Inner Product & Cosine Similarity Engine
    # =========================================================================

    @staticmethod
    def vector_dot_product(u: np.ndarray, v: np.ndarray) -> float:
        """Computes the Euclidean inner (dot) product: <u, v> = sum_{k=1}^n u_k * v_k."""
        if u.shape != v.shape:
            raise ValueError(f"Vector dimensions must match. Got {u.shape} and {v.shape}")

        acc = 0.0
        for i in range(len(u)):
            acc += float(u[i]) * float(v[i])
        return acc

    def compute_gram_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Computes the m x m Gram Matrix G = A * A^T from pairwise customer inner products:

            G_{ij} = <a_i, a_j> = sum_{k=1}^n A_{ik} * A_{jk}
        """
        m, n = matrix.shape
        G = np.zeros((m, m), dtype=np.float64)

        for i in range(m):
            a_i = matrix[i]
            G[i, i] = self.vector_dot_product(a_i, a_i)

            for j in range(i + 1, m):
                val = self.vector_dot_product(a_i, matrix[j])
                G[i, j] = val
                G[j, i] = val

        return G

    def compute_similarity_engine(
        self,
        matrix: np.ndarray,
        eps: Optional[float] = None,
    ) -> SimilarityResult:
        """Computes Gram Matrix G, Norm Matrix D, Inverse D^-1, and Cosine Similarity Matrix S."""
        threshold = eps if eps is not None else self.tolerance
        m, _ = matrix.shape

        G = self.compute_gram_matrix(matrix)
        diag_elements = np.diag(G)
        vector_norms = np.sqrt(np.maximum(0.0, diag_elements))

        D = np.diag(vector_norms)

        inv_diag = np.zeros(m, dtype=np.float64)
        for i in range(m):
            if vector_norms[i] > threshold:
                inv_diag[i] = 1.0 / vector_norms[i]
            else:
                inv_diag[i] = 0.0

        D_inv = np.diag(inv_diag)

        S = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            if vector_norms[i] <= threshold:
                continue
            S[i, i] = 1.0

            for j in range(i + 1, m):
                if vector_norms[j] <= threshold:
                    continue
                cos_sim = G[i, j] * inv_diag[i] * inv_diag[j]
                cos_sim = max(-1.0, min(1.0, cos_sim))
                S[i, j] = cos_sim
                S[j, i] = cos_sim

        return SimilarityResult(
            gram_matrix=G,
            norm_matrix_d=D,
            inv_norm_matrix_d=D_inv,
            cosine_similarity_matrix=S,
            vector_norms=vector_norms,
        )

    # =========================================================================
    # Stage 4: Spectral Decomposition, PCA & Covariance Matrix
    # =========================================================================

    @staticmethod
    def compute_mean_vector(matrix: np.ndarray) -> np.ndarray:
        """Computes column mean vector mu = (1/m) * sum_{i=1}^m a_i in R^n."""
        m, n = matrix.shape
        mu = np.zeros(n, dtype=np.float64)
        for j in range(n):
            mu[j] = np.sum(matrix[:, j]) / float(m)
        return mu

    @staticmethod
    def compute_mean_centered_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Constructs mean-centered matrix X = A - 1 * mu^T where X_{ij} = A_{ij} - mu_j.

        Returns:
            Tuple of (mean_centered_matrix X, mean_vector mu).
        """
        mu = LinearAlgebraEngine.compute_mean_vector(matrix)
        X = matrix.astype(np.float64, copy=True) - mu
        return X, mu

    @staticmethod
    def compute_covariance_matrix(mean_centered_matrix: np.ndarray) -> np.ndarray:
        """Computes the Sample Covariance Matrix C = (1/(m-1)) * X^T * X in R^{n x n}.

        Implemented from first principles:
            C_{jk} = (1/(m-1)) * sum_{i=1}^m X_{ij} * X_{ik}
        """
        m, n = mean_centered_matrix.shape
        if m < 2:
            raise ValueError(f"Sample covariance requires at least 2 customer rows. Received m={m}")

        C = np.zeros((n, n), dtype=np.float64)
        scale = 1.0 / float(m - 1)

        for j in range(n):
            col_j = mean_centered_matrix[:, j]
            C[j, j] = scale * np.sum(col_j * col_j)

            for k in range(j + 1, n):
                val = scale * np.sum(col_j * mean_centered_matrix[:, k])
                C[j, k] = val
                C[k, j] = val

        return C

    @staticmethod
    def householder_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Householder QR Decomposition: Decomposes square matrix A into Q * R

        where Q is orthogonal (Q^T Q = I) and R is upper triangular.
        Implemented from numerical first principles.
        """
        m, n = A.shape
        Q = np.eye(m, dtype=np.float64)
        R = A.astype(np.float64, copy=True)

        for k in range(min(m - 1, n)):
            x = R[k:, k]
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-15:
                continue

            sign = 1.0 if x[0] >= 0 else -1.0
            alpha = sign * norm_x
            u = x.copy()
            u[0] += alpha
            norm_u = np.linalg.norm(u)
            if norm_u < 1e-15:
                continue

            v = u / norm_u

            # R[k:, k:] -= 2.0 * v * (v^T * R[k:, k:])
            v_dot_R = np.dot(v, R[k:, k:])
            R[k:, k:] -= 2.0 * np.outer(v, v_dot_R)

            # Q[:, k:] -= 2.0 * (Q[:, k:] * v) * v^T
            Q_dot_v = np.dot(Q[:, k:], v)
            Q[:, k:] -= 2.0 * np.outer(Q_dot_v, v)

        return Q, R

    @staticmethod
    def solve_characteristic_polynomial_2x2(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Analytic solver for 2x2 symmetric matrix characteristic polynomial:

            det(C - lambda * I) = lambda^2 - Tr(C) * lambda + det(C) = 0
        """
        tr = float(C[0, 0] + C[1, 1])
        det = float(C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0])
        discriminant = max(0.0, tr * tr - 4.0 * det)
        sqrt_disc = np.sqrt(discriminant)

        l1 = (tr + sqrt_disc) / 2.0
        l2 = (tr - sqrt_disc) / 2.0
        eigenvals = np.array([l1, l2], dtype=np.float64)

        # Eigenvectors via (C - lambda * I) * v = 0
        V = np.zeros((2, 2), dtype=np.float64)
        for idx, lam in enumerate([l1, l2]):
            # (C_00 - lam)*x + C_01*y = 0 => [C_01, lam - C_00]
            if abs(C[0, 1]) > 1e-12:
                v = np.array([C[0, 1], lam - C[0, 0]], dtype=np.float64)
            elif abs(C[1, 0]) > 1e-12:
                v = np.array([lam - C[1, 1], C[1, 0]], dtype=np.float64)
            else:
                v = np.array([1.0, 0.0] if idx == 0 else [0.0, 1.0], dtype=np.float64)
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-15:
                v = v / norm_v
            V[:, idx] = v

        return eigenvals, V

    @staticmethod
    def solve_characteristic_polynomial_3x3(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Analytic solver for 3x3 symmetric matrix characteristic polynomial using Cardano's

        trigonometric formulation for real symmetric matrices:
            det(C - lambda * I) = -lambda^3 + c2 * lambda^2 - c1 * lambda + c0 = 0
        """
        # Invariants of 3x3 symmetric matrix
        c2 = float(C[0, 0] + C[1, 1] + C[2, 2])
        c1 = float(
            (C[0, 0] * C[1, 1] - C[0, 1] ** 2)
            + (C[1, 1] * C[2, 2] - C[1, 2] ** 2)
            + (C[0, 0] * C[2, 2] - C[0, 2] ** 2)
        )
        c0 = float(np.linalg.det(C))

        p1 = C[0, 1] ** 2 + C[0, 2] ** 2 + C[1, 2] ** 2
        if p1 < 1e-15:
            # Diagonal matrix
            vals = np.array([C[0, 0], C[1, 1], C[2, 2]], dtype=np.float64)
            V = np.eye(3, dtype=np.float64)
            return vals, V

        q = c2 / 3.0
        p2 = (C[0, 0] - q) ** 2 + (C[1, 1] - q) ** 2 + (C[2, 2] - q) ** 2 + 2.0 * p1
        p = np.sqrt(p2 / 6.0)

        # B = (1/p) * (C - q*I)
        B = (C - q * np.eye(3)) / p
        r = float(np.linalg.det(B)) / 2.0
        r = max(-1.0, min(1.0, r))

        phi = np.arccos(r) / 3.0
        l1 = q + 2.0 * p * np.cos(phi)
        l3 = q + 2.0 * p * np.cos(phi + (2.0 * np.pi / 3.0))
        l2 = 3.0 * q - l1 - l3  # Since l1 + l2 + l3 = c2 = 3*q

        raw_eigenvalues = np.array([l1, l2, l3], dtype=np.float64)

        # Eigenvectors via cross product of rows of (C - lam * I)
        V = np.zeros((3, 3), dtype=np.float64)
        refined_eigenvalues = np.zeros(3, dtype=np.float64)

        for idx in range(3):
            lam = raw_eigenvalues[idx]
            shifted = C - lam * np.eye(3)

            r1, r2, r3 = shifted[0], shifted[1], shifted[2]
            cp1 = np.cross(r1, r2)
            cp2 = np.cross(r1, r3)
            cp3 = np.cross(r2, r3)

            norm1 = float(np.linalg.norm(cp1))
            norm2 = float(np.linalg.norm(cp2))
            norm3 = float(np.linalg.norm(cp3))

            max_norm = max(norm1, norm2, norm3)
            if max_norm > 1e-12:
                if norm1 == max_norm:
                    v = cp1 / norm1
                elif norm2 == max_norm:
                    v = cp2 / norm2
                else:
                    v = cp3 / norm3
            else:
                # Multiple eigenvalue fallback: use null space from RREF
                engine = LinearAlgebraEngine(tolerance=1e-6)
                bases = engine.extract_subspace_bases(shifted, tol=1e-6)
                if bases.null_space_basis.shape[1] > 0:
                    v = bases.null_space_basis[:, 0]
                    norm_v = np.linalg.norm(v)
                    v = v / norm_v if norm_v > 1e-15 else np.array([1.0, 0.0, 0.0])
                else:
                    v = np.zeros(3)
                    v[idx] = 1.0

            # Refine eigenvalue using Rayleigh quotient lambda = v^T * C * v
            v = v / np.linalg.norm(v)
            refined_lam = float(np.dot(v, np.dot(C, v)))
            V[:, idx] = v
            refined_eigenvalues[idx] = refined_lam

        return refined_eigenvalues, V

    @staticmethod
    def jacobi_eigen_decomposition(
        C: np.ndarray,
        tol: float = 1e-12,
        max_iter: int = 300,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Jacobi Eigenvalue Algorithm for real symmetric matrices.

        Iteratively applies orthogonal Givens similarity rotations J^T * A * J to systematically
        annihilate off-diagonal elements until quadratic convergence.
        Guarantees exact orthogonality V^T * V = I with no numerical drift.

        Args:
            C: Real symmetric matrix in R^{n x n}.
            tol: Convergence tolerance for maximum off-diagonal magnitude.
            max_iter: Maximum sweeps.

        Returns:
            Tuple of (eigenvalues, eigenvectors).
        """
        n = C.shape[0]
        A = C.astype(np.float64, copy=True)
        V = np.eye(n, dtype=np.float64)

        for _ in range(max_iter):
            # 1. Find the largest off-diagonal element |A[p, q]|
            p, q = 0, 1
            max_off = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    abs_val = abs(A[i, j])
                    if abs_val > max_off:
                        max_off = abs_val
                        p, q = i, j

            if max_off < tol:
                break

            # 2. Compute Jacobi rotation parameters (cos theta, sin theta)
            diff = A[q, q] - A[p, p]
            if abs(A[p, q]) < 1e-15:
                t = 0.0
            else:
                tau = diff / (2.0 * A[p, q])
                if tau >= 0.0:
                    t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + np.sqrt(1.0 + tau * tau))

            c = 1.0 / np.sqrt(1.0 + t * t)
            s = t * c

            # 3. Apply rotation to matrix A = J^T * A * J
            # Update rows and columns p and q
            A_pp = A[p, p]
            A_qq = A[q, q]
            A_pq = A[p, q]

            A[p, p] = c * c * A_pp - 2.0 * s * c * A_pq + s * s * A_qq
            A[q, q] = s * s * A_pp + 2.0 * s * c * A_pq + c * c * A_qq
            A[p, q] = 0.0
            A[q, p] = 0.0

            for i in range(n):
                if i != p and i != q:
                    A_ip = A[i, p]
                    A_iq = A[i, q]
                    A[i, p] = c * A_ip - s * A_iq
                    A[p, i] = A[i, p]
                    A[i, q] = s * A_ip + c * A_iq
                    A[q, i] = A[i, q]

            # 4. Accumulate rotation into eigenvector matrix V = V * J
            for i in range(n):
                V_ip = V[i, p]
                V_iq = V[i, q]
                V[i, p] = c * V_ip - s * V_iq
                V[i, q] = s * V_ip + c * V_iq

        eigenvalues = np.diag(A).copy()
        # Ensure tiny negative roundoff values in covariance are clamped to 0
        eigenvalues[eigenvalues < 0.0] = 0.0

        return eigenvalues, V

    @staticmethod
    def qr_eigen_decomposition(
        C: np.ndarray,
        tol: float = 1e-12,
        max_iter: int = 300,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Symmetric QR Algorithm with Householder Tridiagonalization, Wilkinson Shifts,

        and Submatrix Deflation for general n.
        Decomposes real symmetric matrix into C = V * Lambda * V^T with machine precision.
        """
        n = C.shape[0]
        A = C.astype(np.float64, copy=True)
        V = np.eye(n, dtype=np.float64)

        # 1. Householder similarity tridiagonalization: A -> Q * T * Q^T
        for k in range(n - 2):
            x = A[k + 1 :, k]
            norm_x = np.linalg.norm(x)
            if norm_x > 1e-14:
                sign = 1.0 if x[0] >= 0 else -1.0
                u = x.copy()
                u[0] += sign * norm_x
                norm_u = np.linalg.norm(u)
                if norm_u > 1e-15:
                    v = u / norm_u
                    P = np.eye(n - 1 - k) - 2.0 * np.outer(v, v)
                    A[k + 1 :, k:] = np.dot(P, A[k + 1 :, k:])
                    A[:, k + 1 :] = np.dot(A[:, k + 1 :], P)
                    V[:, k + 1 :] = np.dot(V[:, k + 1 :], P)

        # 2. Shifted QR iteration with Wilkinson shift and active submatrix deflation
        m = n
        while m > 1:
            converged = False
            for _ in range(max_iter):
                # Subdiagonal convergence check
                threshold = tol * (abs(A[m - 2, m - 2]) + abs(A[m - 1, m - 1]) + 1e-15)
                if abs(A[m - 1, m - 2]) <= threshold:
                    A[m - 1, m - 2] = 0.0
                    A[m - 2, m - 1] = 0.0
                    m -= 1
                    converged = True
                    break

                # Wilkinson shift from trailing 2x2 principal submatrix
                d = (A[m - 2, m - 2] - A[m - 1, m - 1]) / 2.0
                b = A[m - 1, m - 2]
                sign_d = 1.0 if d >= 0 else -1.0
                shift = A[m - 1, m - 1] - (sign_d * b * b) / (abs(d) + np.sqrt(d * d + b * b) + 1e-15)

                # Custom Householder QR step on active submatrix A[:m, :m]
                sub_A = A[:m, :m] - shift * np.eye(m)
                Q_sub, R_sub = LinearAlgebraEngine.householder_qr(sub_A)
                A[:m, :m] = np.dot(R_sub, Q_sub) + shift * np.eye(m)
                V[:, :m] = np.dot(V[:, :m], Q_sub)

            if not converged:
                m -= 1

        eigenvalues = np.diag(A).copy()
        eigenvalues[eigenvalues < 0.0] = 0.0
        return eigenvalues, V

    def compute_spectral_decomposition(
        self,
        matrix_or_cov: np.ndarray,
        is_covariance: bool = False,
        method: str = "auto",
    ) -> SpectralResult:
        """Executes full spectral decomposition and PCA on the covariance matrix.

        Pipeline:
        1. Mean centering: X = A - 1 * mu^T.
        2. Covariance matrix formulation: C = (1/(m-1)) * X^T * X.
        3. Custom numerical eigenvalue & eigenvector extraction.
        4. Sorting eigenpairs in descending order: lambda_1 >= lambda_2 >= ... >= lambda_n >= 0.
        5. Proportion of variance explained and cumulative variance ratio R(k).

        Args:
            matrix_or_cov: Either customer matrix A in R^{m x n} or covariance matrix C in R^{n x n}.
            is_covariance: If True, matrix_or_cov is already the covariance matrix C.
            method: Solver selection ('auto', 'jacobi', 'qr', 'characteristic').

        Returns:
            SpectralResult containing sorted eigenvalues, orthonormal eigenvectors, and variance metrics.
        """
        if is_covariance:
            C = matrix_or_cov.astype(np.float64, copy=True)
            n = C.shape[0]
            X = np.empty((0, n), dtype=np.float64)
            mu = np.zeros(n, dtype=np.float64)
        else:
            X, mu = self.compute_mean_centered_matrix(matrix_or_cov)
            C = self.compute_covariance_matrix(X)
            n = C.shape[0]

        # 1. Select and execute eigenvalue algorithm
        chosen_method = method.lower()
        if chosen_method == "auto":
            if n <= 3:
                chosen_method = "characteristic"
            else:
                chosen_method = "jacobi"

        if chosen_method == "characteristic" and n == 2:
            raw_vals, raw_vecs = self.solve_characteristic_polynomial_2x2(C)
        elif chosen_method == "characteristic" and n == 3:
            raw_vals, raw_vecs = self.solve_characteristic_polynomial_3x3(C)
        elif chosen_method in ["qr"]:
            raw_vals, raw_vecs = self.qr_eigen_decomposition(C)
        else:
            raw_vals, raw_vecs = self.jacobi_eigen_decomposition(C)

        # 2. Sort eigenpairs in descending order: lambda_1 >= lambda_2 >= ...
        sort_order = np.argsort(raw_vals)[::-1]
        sorted_eigenvalues = raw_vals[sort_order].copy()
        sorted_eigenvectors = raw_vecs[:, sort_order].copy()

        # 3. Canonical normalization and deterministic sign convention
        for col in range(n):
            v = sorted_eigenvectors[:, col]
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-15:
                v = v / norm_v
            # Force largest absolute element to be positive
            max_idx = int(np.argmax(np.abs(v)))
            if v[max_idx] < 0:
                v = -v
            sorted_eigenvectors[:, col] = v

        # 4. Variance analysis
        total_variance = float(np.sum(sorted_eigenvalues))
        trace_cov = float(np.trace(C))

        if total_variance > 1e-15:
            variance_ratios = sorted_eigenvalues / total_variance
            cumulative_variance_ratios = np.cumsum(variance_ratios)
        else:
            variance_ratios = np.zeros(n, dtype=np.float64)
            cumulative_variance_ratios = np.zeros(n, dtype=np.float64)

        return SpectralResult(
            eigenvalues=sorted_eigenvalues,
            eigenvectors=sorted_eigenvectors,
            variance_ratios=variance_ratios,
            cumulative_variance_ratios=cumulative_variance_ratios,
            covariance_matrix=C,
            mean_vector=mu,
            mean_centered_matrix=X,
            trace_covariance=trace_cov,
        )

    # =========================================================================
    # Stage 5: Subspace Orthogonal Projection
    # =========================================================================

    def compute_orthogonal_projection(
        self,
        matrix_or_spectral: Union[np.ndarray, SpectralResult],
        k: Optional[int] = None,
        variance_threshold: Optional[float] = None,
    ) -> ProjectionResult:
        """Projects customer vectors into a k-dimensional subspace via orthogonal projection.

        Mathematical Operations:
        1. Select top k orthonormal eigenvectors V_k = [v_1, ..., v_k] in R^{n x k}.
        2. Construct Orthogonal Projection Matrix:
               P = V_k * V_k^T in R^{n x n}
           satisfying P^T = P and P^2 = P.
        3. Project mean-centered customer vectors:
               Y = X * V_k in R^{m x k}
        4. Reconstruct back to category space:
               X_hat = Y * V_k^T = X * P in R^{m x n}
               A_hat = X_hat + 1 * mu^T in R^{m x n}
        5. Quantify information loss (reconstruction error):
               L(k) = ||X - X_hat||_F^2 / ||X||_F^2

        Args:
            matrix_or_spectral: Original matrix A in R^{m x n} or precomputed SpectralResult.
            k: Subspace dimension to retain (1 <= k <= n).
            variance_threshold: Target cumulative variance ratio (e.g., 0.90 for 90%).

        Returns:
            ProjectionResult dataclass containing Y, P, V_k, and reconstruction statistics.
        """
        if isinstance(matrix_or_spectral, SpectralResult):
            spec_res = matrix_or_spectral
        else:
            spec_res = self.compute_spectral_decomposition(matrix_or_spectral)

        X = spec_res.mean_centered_matrix
        mu = spec_res.mean_vector
        m, n = X.shape
        cum_var = spec_res.cumulative_variance_ratios

        # Resolve number of principal components k
        if k is not None:
            resolved_k = max(1, min(n, int(k)))
        elif variance_threshold is not None:
            # Find minimum k achieving desired variance
            target = float(variance_threshold)
            resolved_k = int(np.searchsorted(cum_var, target) + 1)
            resolved_k = max(1, min(n, resolved_k))
        else:
            # Default to top 2 components for 2D visualization
            resolved_k = min(2, n)

        # 1. Select top k eigenvectors V_k in R^{n x k}
        V_k = spec_res.eigenvectors[:, :resolved_k]
        top_eigenvalues = spec_res.eigenvalues[:resolved_k]
        var_explained = float(cum_var[resolved_k - 1]) if len(cum_var) >= resolved_k else 1.0

        # 2. Orthogonal Projection Operator P = V_k * V_k^T in R^{n x n}
        P = np.matmul(V_k, V_k.T)

        # 3. Subspace Customer Coordinates Y = X * V_k in R^{m x k}
        Y = np.matmul(X, V_k)

        # 4. Reconstruction
        X_hat = np.matmul(Y, V_k.T)
        A_hat = X_hat + mu

        # 5. Reconstruction Error (Relative Frobenius Norm Squared)
        norm_X_sq = float(np.sum(X * X))
        norm_diff_sq = float(np.sum((X - X_hat) ** 2))
        recon_error = (norm_diff_sq / norm_X_sq) if norm_X_sq > 1e-15 else 0.0

        return ProjectionResult(
            projected_coordinates=Y,
            projection_matrix=P,
            top_eigenvectors=V_k,
            top_eigenvalues=top_eigenvalues,
            k=resolved_k,
            cumulative_variance_explained=var_explained,
            reconstructed_centered=X_hat,
            reconstructed_original=A_hat,
            reconstruction_error=recon_error,
        )

    # =========================================================================
    # Validation & Verification Utilities
    # =========================================================================

    def verify_null_space(
        self,
        matrix: np.ndarray,
        null_space_basis: np.ndarray,
        tol: Optional[float] = None,
    ) -> bool:
        """Verifies that every basis vector v in N(A) satisfies A * v = 0 within tolerance."""
        eps = tol if tol is not None else self.tolerance
        if null_space_basis.shape[1] == 0:
            return True

        residuals = np.matmul(matrix, null_space_basis)
        max_residual = float(np.max(np.abs(residuals)))
        return max_residual <= eps

    def verify_spectral_identities(
        self,
        covariance_matrix: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        tol: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Validates fundamental theoretical identities for spectral decomposition:

        1. Trace Identity: Trace(C) == sum(lambda_i)
        2. Determinant Identity: det(C) == prod(lambda_i)
        3. Orthonormality: V^T * V == I
        4. Eigenvalue Equation: C * v_i == lambda_i * v_i

        Args:
            covariance_matrix: Covariance matrix C in R^{n x n}.
            eigenvalues: Computed eigenvalues array.
            eigenvectors: Computed eigenvector matrix V in R^{n x n}.
            tol: Tolerance.

        Returns:
            Dictionary of boolean verification flags and numerical errors.
        """
        eps = tol if tol is not None else self.tolerance
        n = covariance_matrix.shape[0]

        # 1. Trace Identity
        trace_C = float(np.trace(covariance_matrix))
        sum_lambdas = float(np.sum(eigenvalues))
        trace_error = abs(trace_C - sum_lambdas)

        # 2. Determinant Identity
        det_C = float(np.linalg.det(covariance_matrix))
        prod_lambdas = float(np.prod(eigenvalues))
        det_error = abs(det_C - prod_lambdas)

        # 3. Orthonormality: V^T * V - I == 0
        V_t_V = np.matmul(eigenvectors.T, eigenvectors)
        orthonormality_error = float(np.max(np.abs(V_t_V - np.eye(n))))

        # 4. Eigenvalue Equation: C * v_i - lambda_i * v_i == 0
        eigen_eqn_errors = []
        for i in range(n):
            v_i = eigenvectors[:, i]
            lam_i = eigenvalues[i]
            res = np.matmul(covariance_matrix, v_i) - lam_i * v_i
            eigen_eqn_errors.append(float(np.linalg.norm(res)))
        max_eigen_eqn_error = max(eigen_eqn_errors)

        return {
            "trace_match": trace_error <= 1e-6,
            "trace_error": trace_error,
            "det_match": det_error <= 1e-4,
            "det_error": det_error,
            "orthonormal": orthonormality_error <= eps,
            "orthonormality_error": orthonormality_error,
            "eigenpair_valid": max_eigen_eqn_error <= eps,
            "max_eigen_error": max_eigen_eqn_error,
        }
