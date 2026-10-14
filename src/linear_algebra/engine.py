"""Linear Algebra Engine.

Implements core numerical linear algebra operations from mathematical first principles
without relying on high-level black-box library abstractions (e.g., scipy.linalg.null_space,
numpy.linalg.matrix_rank, sympy.rref).

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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
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

        # Create working copy in float64 precision
        R = matrix.astype(np.float64, copy=True)

        pivot_row = 0
        pivot_columns: List[int] = []
        row_swaps = 0

        for col in range(n):
            if pivot_row >= m:
                break

            # 1. Partial Pivoting: Select row with maximum absolute entry in column `col`
            # at or below `pivot_row` to guarantee numerical stability.
            max_val = 0.0
            max_row_idx = pivot_row
            for r in range(pivot_row, m):
                abs_val = abs(R[r, col])
                if abs_val > max_val:
                    max_val = abs_val
                    max_row_idx = r

            # If all candidate entries in this column are effectively zero,
            # this column has no pivot; it corresponds to a free variable.
            if max_val <= eps:
                continue

            # 2. Row Permutation: Swap current pivot_row with max_row_idx if needed
            if max_row_idx != pivot_row:
                R[[pivot_row, max_row_idx]] = R[[max_row_idx, pivot_row]]
                row_swaps += 1

            # 3. Pivot Normalization: Scale pivot row so the leading entry becomes 1.0
            pivot_val = R[pivot_row, col]
            R[pivot_row, :] = R[pivot_row, :] / pivot_val
            R[pivot_row, col] = 1.0  # Eliminate round-off noise

            # 4. Elimination: Eliminate all entries in column `col` above and below the pivot
            for r in range(m):
                if r != pivot_row:
                    factor = R[r, col]
                    if abs(factor) > eps:
                        R[r, :] -= factor * R[pivot_row, :]
                        R[r, col] = 0.0  # Force exact zero

            pivot_columns.append(col)
            pivot_row += 1

        # Clean up any remaining small residual values below tolerance
        R[np.abs(R) < eps] = 0.0

        rank = len(pivot_columns)
        # By the Rank-Nullity Theorem: dim(C(A)) + dim(N(A)) = n => Nullity = n - rank
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

        Uses Gaussian Elimination with partial pivoting without external library calls.

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
            SubspaceBases dataclass containing the three subspace basis matrices.
        """
        eps = tol if tol is not None else self.tolerance
        m, n = matrix.shape

        rref_res = self.compute_rref(matrix, tol=eps)
        R = rref_res.rref_matrix
        rank = rref_res.rank
        nullity = rref_res.nullity
        pivot_cols = rref_res.pivot_columns
        free_cols = rref_res.free_columns

        # ---------------------------------------------------------------------
        # 1. Row Space Basis R(A)
        # ---------------------------------------------------------------------
        # The non-zero row vectors of RREF(A) form a linearly independent basis for R(A).
        if rank > 0:
            row_space_basis = R[:rank, :].copy()
        else:
            row_space_basis = np.empty((0, n), dtype=np.float64)

        # ---------------------------------------------------------------------
        # 2. Column Space Basis C(A)
        # ---------------------------------------------------------------------
        # The pivot columns of the ORIGINAL matrix A form a basis for C(A).
        if rank > 0:
            column_space_basis = matrix[:, pivot_cols].astype(np.float64, copy=True)
        else:
            column_space_basis = np.empty((m, 0), dtype=np.float64)

        # ---------------------------------------------------------------------
        # 3. Null Space Basis N(A)
        # ---------------------------------------------------------------------
        # Solve R * x = 0 by parameterizing free variables.
        # For each free variable j in free_cols:
        #   Set x_j = 1, and all other free variables to 0.
        #   For each pivot variable p_i in pivot_cols (corresponding to row i in R):
        #     x_{p_i} + sum_{k in free_cols} R[i, k] * x_k = 0
        #     => x_{p_i} = -R[i, j]
        if nullity > 0:
            null_space_basis = np.zeros((n, nullity), dtype=np.float64)

            for col_idx, free_col in enumerate(free_cols):
                # Free variable component set to 1.0
                null_space_basis[free_col, col_idx] = 1.0

                # Pivot variable components solved from RREF equations
                for row_idx, pivot_col in enumerate(pivot_cols):
                    # x_{pivot} = - R[row_idx, free_col]
                    null_space_basis[pivot_col, col_idx] = -R[row_idx, free_col]

            # Snap small residual noise
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
        """Computes the Euclidean inner (dot) product between two vectors:

            <u, v> = sum_{k=1}^n u_k * v_k

        Implemented from definition without library dot wrappers.

        Args:
            u: First 1D vector in R^n.
            v: Second 1D vector in R^n.

        Returns:
            Scalar inner product value.
        """
        if u.shape != v.shape:
            raise ValueError(f"Vector dimensions must match. Got {u.shape} and {v.shape}")

        acc = 0.0
        for i in range(len(u)):
            acc += float(u[i]) * float(v[i])
        return acc

    def compute_gram_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Computes the m x m Gram Matrix G = A * A^T from pairwise customer inner products:

            G_{ij} = <a_i, a_j> = sum_{k=1}^n A_{ik} * A_{jk}

        Args:
            matrix: Customer transaction matrix A in R^{m x n}.

        Returns:
            Symmetric Gram Matrix G in R^{m x m}.
        """
        m, n = matrix.shape
        G = np.zeros((m, m), dtype=np.float64)

        # Exploit symmetry: G_{ji} = G_{ij}
        for i in range(m):
            a_i = matrix[i]
            # Diagonal: <a_i, a_i> = ||a_i||_2^2
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
        """Computes the full vector similarity pipeline:

        1. Gram Matrix: G = A * A^T in R^{m x m}.
        2. Diagonal Norm Matrix: D = diag(||a_1||_2, ..., ||a_m||_2) in R^{m x m}.
        3. Inverse Diagonal Norm Matrix: D^-1 where (D^-1)_{ii} = 1 / ||a_i||_2 (0 if norm == 0).
        4. Cosine Similarity Matrix: S = D^-1 * G * D^-1 in R^{m x m}.

        Entry S_{ij} is given by:
            S_{ij} = <a_i, a_j> / (||a_i||_2 * ||a_j||_2) = cos(theta_{ij})

        Args:
            matrix: Customer purchase matrix A in R^{m x n}.
            eps: Epsilon threshold for zero-norm protection.

        Returns:
            SimilarityResult dataclass containing G, D, D^-1, S, and vector norms.
        """
        threshold = eps if eps is not None else self.tolerance
        m, _ = matrix.shape

        # 1. Compute Gram Matrix G = A * A^T
        G = self.compute_gram_matrix(matrix)

        # 2. Extract vector norms from diagonal: ||a_i||_2 = sqrt(G_{ii})
        diag_elements = np.diag(G)
        vector_norms = np.sqrt(np.maximum(0.0, diag_elements))

        # 3. Construct Diagonal Matrix D and Inverse D^-1
        D = np.diag(vector_norms)

        inv_diag = np.zeros(m, dtype=np.float64)
        for i in range(m):
            if vector_norms[i] > threshold:
                inv_diag[i] = 1.0 / vector_norms[i]
            else:
                inv_diag[i] = 0.0

        D_inv = np.diag(inv_diag)

        # 4. Derive Cosine Similarity Matrix: S = D^-1 * G * D^-1
        # In component form: S_{ij} = inv_diag[i] * G_{ij} * inv_diag[j]
        # Directly compute to preserve exact numerical symmetry and avoid triple matrix matmul:
        S = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            if vector_norms[i] <= threshold:
                continue
            S[i, i] = 1.0  # A non-zero vector has cosine similarity 1.0 with itself

            for j in range(i + 1, m):
                if vector_norms[j] <= threshold:
                    continue
                cos_sim = G[i, j] * inv_diag[i] * inv_diag[j]
                # Clamp within [-1.0, 1.0] for exact Cauchy-Schwarz preservation
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
    # Validation & Verification Utilities
    # =========================================================================

    def verify_null_space(
        self,
        matrix: np.ndarray,
        null_space_basis: np.ndarray,
        tol: Optional[float] = None,
    ) -> bool:
        """Verifies that every basis vector v in N(A) satisfies A * v = 0 within tolerance.

        Args:
            matrix: Original matrix A in R^{m x n}.
            null_space_basis: Basis matrix of shape (n, nullity).
            tol: Numerical tolerance.

        Returns:
            True if all vectors are valid null space vectors, False otherwise.
        """
        eps = tol if tol is not None else self.tolerance
        if null_space_basis.shape[1] == 0:
            return True

        residuals = np.matmul(matrix, null_space_basis)
        max_residual = float(np.max(np.abs(residuals)))
        return max_residual <= eps
