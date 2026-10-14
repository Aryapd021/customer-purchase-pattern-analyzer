"""Unit and Mathematical Verification Tests for LinearAlgebraEngine.

Tests Gaussian Elimination (RREF), Rank-Nullity Theorem, Fundamental Subspace Bases,
Gram Matrix, and Cosine Similarity Matrix against analytical and hand-calculated benchmarks.
"""

import numpy as np
import pytest

from src.linear_algebra.engine import (
    LinearAlgebraEngine,
    RREFResult,
    SubspaceBases,
    SimilarityResult,
)
from src.ingestion.synthetic_generator import SyntheticCustomerDataGenerator


class TestLinearAlgebraEngineCore:
    """Mathematical verification of Gaussian Elimination, RREF, and Subspace Bases."""

    @pytest.fixture
    def engine(self) -> LinearAlgebraEngine:
        return LinearAlgebraEngine(tolerance=1e-10)

    @pytest.fixture
    def benchmark_matrix(self) -> np.ndarray:
        """Hand-calculated benchmark matrix from Masterplan §4:
        A_test = [[4, 0, 3],
                  [0, 5, 0],
                  [8, 0, 6]]
        Row 3 = 2 * Row 1
        Rank = 2, Nullity = 1
        """
        return np.array([
            [4.0, 0.0, 3.0],
            [0.0, 5.0, 0.0],
            [8.0, 0.0, 6.0],
        ], dtype=np.float64)

    def test_rref_benchmark_matrix(self, engine, benchmark_matrix):
        """Verifies RREF of the benchmark matrix A_test."""
        res = engine.compute_rref(benchmark_matrix)

        expected_rref = np.array([
            [1.0, 0.0, 0.75],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

        np.testing.assert_allclose(res.rref_matrix, expected_rref, atol=1e-10)
        assert res.rank == 2
        assert res.nullity == 1
        assert res.pivot_columns == [0, 1]
        assert res.free_columns == [2]

    def test_rank_nullity_theorem(self, engine, benchmark_matrix):
        """Validates that Rank + Nullity == n for various matrix topologies."""
        matrices = [
            benchmark_matrix,  # 3x3, rank 2, nullity 1
            np.eye(4),         # 4x4, rank 4, nullity 0
            np.zeros((3, 5)),  # 3x5, rank 0, nullity 5
            np.ones((4, 4)),   # 4x4, rank 1, nullity 3
            np.array([         # 2x4, rank 2, nullity 2
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 6.0, 9.0],
            ]),
        ]

        for mat in matrices:
            m, n = mat.shape
            rank, nullity = engine.determine_rank_and_nullity(mat)
            assert rank + nullity == n, f"Rank-Nullity theorem failed for shape {mat.shape}"
            assert 0 <= rank <= min(m, n)
            assert 0 <= nullity <= n

    def test_subspace_bases_benchmark(self, engine, benchmark_matrix):
        """Validates Row Space, Column Space, and Null Space bases for A_test."""
        bases = engine.extract_subspace_bases(benchmark_matrix)

        # 1. Rank & Dimensions
        assert bases.rank == 2
        assert bases.nullity == 1

        # 2. Row Space Basis (non-zero rows of RREF)
        expected_row_space = np.array([
            [1.0, 0.0, 0.75],
            [0.0, 1.0, 0.0],
        ])
        np.testing.assert_allclose(bases.row_space_basis, expected_row_space, atol=1e-10)

        # 3. Column Space Basis (pivot columns of original matrix)
        expected_col_space = np.array([
            [4.0, 0.0],
            [0.0, 5.0],
            [8.0, 0.0],
        ])
        np.testing.assert_allclose(bases.column_space_basis, expected_col_space, atol=1e-10)

        # 4. Null Space Basis
        # Parameterizing free variable x3 = 1 => x1 = -0.75, x2 = 0
        expected_null_vector = np.array([[-0.75], [0.0], [1.0]])
        np.testing.assert_allclose(bases.null_space_basis, expected_null_vector, atol=1e-10)

        # Verify A * v = 0
        assert engine.verify_null_space(benchmark_matrix, bases.null_space_basis)
        residual = np.matmul(benchmark_matrix, bases.null_space_basis)
        np.testing.assert_allclose(residual, np.zeros((3, 1)), atol=1e-10)

    def test_partial_pivoting_numerical_stability(self, engine):
        """Ensures row swaps occur when pivot element is small or zero."""
        # Row 0 has leading 0, requiring row permutation
        matrix = np.array([
            [0.0, 2.0, 1.0],
            [3.0, 1.0, 4.0],
            [1.0, 5.0, 2.0],
        ])
        res = engine.compute_rref(matrix)

        assert res.rank == 3
        assert res.nullity == 0
        assert res.row_swaps >= 1
        np.testing.assert_allclose(res.rref_matrix, np.eye(3), atol=1e-10)

    def test_null_space_multiple_free_variables(self, engine):
        """Verifies null space derivation with multiple free variables."""
        # Matrix with 2 rows and 4 columns, rank 1, nullity 3
        A = np.array([
            [1.0, 2.0, -1.0, 3.0],
            [2.0, 4.0, -2.0, 6.0],
        ])
        bases = engine.extract_subspace_bases(A)

        assert bases.rank == 1
        assert bases.nullity == 3
        assert bases.null_space_basis.shape == (4, 3)

        # Every basis vector must satisfy A * v = 0
        assert engine.verify_null_space(A, bases.null_space_basis)

        # Null space vectors must be linearly independent
        # Free columns are [1, 2, 3], so rows 1, 2, 3 of null_space_basis must form I_3
        free_block = bases.null_space_basis[[1, 2, 3], :]
        np.testing.assert_allclose(free_block, np.eye(3), atol=1e-10)


class TestVectorSimilarityEngine:
    """Mathematical verification of Inner Products, Gram Matrix, and Cosine Similarity."""

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

    def test_vector_dot_product(self, engine):
        """Validates scalar dot product <u, v>."""
        u = np.array([1.0, 3.0, -5.0])
        v = np.array([4.0, -2.0, -1.0])
        # 1*4 + 3*(-2) + (-5)*(-1) = 4 - 6 + 5 = 3
        assert engine.vector_dot_product(u, v) == pytest.approx(3.0)

    def test_gram_matrix_benchmark(self, engine, benchmark_matrix):
        """Validates the Gram Matrix G = A * A^T against hand-calculated values:
        G = [[25,   0,  50],
             [ 0,  25,   0],
             [50,   0, 100]]
        """
        G = engine.compute_gram_matrix(benchmark_matrix)

        expected_G = np.array([
            [25.0, 0.0, 50.0],
            [0.0, 25.0, 0.0],
            [50.0, 0.0, 100.0],
        ])

        np.testing.assert_allclose(G, expected_G, atol=1e-10)
        # Check symmetry
        np.testing.assert_allclose(G, G.T, atol=1e-12)

    def test_cosine_similarity_matrix_benchmark(self, engine, benchmark_matrix):
        """Validates Cosine Similarity Matrix S = D^-1 * G * D^-1 for A_test:
        D = diag(5, 5, 10)
        D^-1 = diag(0.2, 0.2, 0.1)
        S = [[1.0, 0.0, 1.0],
             [0.0, 1.0, 0.0],
             [1.0, 0.0, 1.0]]
        Customer 1 and Customer 3 are identical in purchasing direction.
        """
        sim_res = engine.compute_similarity_engine(benchmark_matrix)

        # 1. Norms
        expected_norms = np.array([5.0, 5.0, 10.0])
        np.testing.assert_allclose(sim_res.vector_norms, expected_norms, atol=1e-10)

        # 2. Diagonal Norm Matrix D and D^-1
        np.testing.assert_allclose(sim_res.norm_matrix_d, np.diag(expected_norms), atol=1e-10)
        np.testing.assert_allclose(sim_res.inv_norm_matrix_d, np.diag([0.2, 0.2, 0.1]), atol=1e-10)

        # 3. Cosine Similarity Matrix S
        expected_S = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ])
        np.testing.assert_allclose(sim_res.cosine_similarity_matrix, expected_S, atol=1e-10)
        # Verify symmetry and diagonal
        np.testing.assert_allclose(sim_res.cosine_similarity_matrix, sim_res.cosine_similarity_matrix.T)
        np.testing.assert_allclose(np.diag(sim_res.cosine_similarity_matrix), [1.0, 1.0, 1.0])

    def test_similarity_with_zero_vectors(self, engine):
        """A customer with 0 purchases in all categories should yield 0 similarity safely."""
        matrix_with_zero = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [6.0, 8.0, 0.0],
        ])

        sim_res = engine.compute_similarity_engine(matrix_with_zero)
        S = sim_res.cosine_similarity_matrix

        # Zero row has norm 0
        assert sim_res.vector_norms[0] == 0.0
        # Similarity involving zero vector must be 0
        assert S[0, 0] == 0.0
        assert S[0, 1] == 0.0
        assert S[0, 2] == 0.0
        # Customers 1 and 2 are parallel
        assert S[1, 2] == pytest.approx(1.0)
        assert S[1, 1] == pytest.approx(1.0)
        assert S[2, 2] == pytest.approx(1.0)

    def test_integration_with_synthetic_generator(self, engine):
        """End-to-end integration: generate synthetic dataset of controlled rank r=3,
        and confirm LinearAlgebraEngine recovers exact rank r, valid null space, and valid S.
        """
        generator = SyntheticCustomerDataGenerator(random_state=99)
        m, n, r = 50, 8, 3
        synth = generator.generate(n_customers=m, n_categories=n, rank=r, noise_level=0.0)

        # 1. Rank & Nullity
        rank, nullity = engine.determine_rank_and_nullity(synth.matrix)
        assert rank == r
        assert nullity == n - r

        # 2. Subspaces
        bases = engine.extract_subspace_bases(synth.matrix)
        assert bases.row_space_basis.shape == (r, n)
        assert bases.column_space_basis.shape == (m, r)
        assert bases.null_space_basis.shape == (n, n - r)
        assert engine.verify_null_space(synth.matrix, bases.null_space_basis)

        # 3. Similarity Engine
        sim = engine.compute_similarity_engine(synth.matrix)
        assert sim.gram_matrix.shape == (m, m)
        assert sim.cosine_similarity_matrix.shape == (m, m)
        assert np.all(sim.cosine_similarity_matrix >= -1e-10)
        assert np.all(sim.cosine_similarity_matrix <= 1.0 + 1e-10)
        np.testing.assert_allclose(np.diag(sim.cosine_similarity_matrix), np.ones(m), atol=1e-10)
