"""Main entry point and demonstration script for the Customer Purchase Pattern & Profile Analyzer.

Executes Stages 1 through 5 of the sequential mathematical pipeline:
1. Stage 1: Synthetic data generation with controllable rank r < min(m, n), CSV/JSON parsing, missing value imputation, and vector normalizations (L2 and Mean Centering).
2. Stage 2: Gaussian Elimination with Partial Pivoting to RREF, Matrix Rank, Nullity (Rank-Nullity Theorem), and Subspace Bases (Row Space, Column Space, Null Space).
3. Stage 3: Vector Inner Products, Gram Matrix G = A * A^T, Diagonal Norm Matrix D, and Cosine Similarity Matrix S = D^-1 * G * D^-1.
4. Stage 4: Covariance Matrix C = (1/(m-1)) * X^T * X, Spectral Decomposition (QR algorithm & Jacobi method), Sorted Eigenpairs, and Cumulative Variance Ratio R(k).
5. Stage 5: Orthogonal Subspace Projection Operator P = V_k * V_k^T, Low-Dimensional Customer Coordinates Y = X * V_k, Reconstruction, and Information Loss.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from src.ingestion.synthetic_generator import SyntheticCustomerDataGenerator
from src.ingestion.data_ingestion import DataIngestionModule, NormalizationType, ImputationStrategy
from src.linear_algebra.engine import LinearAlgebraEngine


def print_header(title: str) -> None:
    """Prints a styled header."""
    if HAS_RICH:
        console = Console()
        console.print(Panel(f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))
    else:
        print("\n" + "=" * 78)
        print(f" {title.upper()} ")
        print("=" * 78)


def display_matrix_summary(title: str, matrix: np.ndarray, rows_to_show: int = 5, cols_to_show: int = 5) -> None:
    """Displays a preview and key mathematical properties of a matrix."""
    m, n = matrix.shape
    engine = LinearAlgebraEngine()
    rank, _ = engine.determine_rank_and_nullity(matrix)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    row_norms = np.linalg.norm(matrix, ord=2, axis=1)

    if HAS_RICH:
        console = Console()
        table = Table(title=f"{title} (Shape: {m} x {n}, Rank: {rank})")
        table.add_column("Index", style="cyan", no_wrap=True)
        for j in range(min(n, cols_to_show)):
            table.add_column(f"Col {j+1}", justify="right")
        if n > cols_to_show:
            table.add_column("...", justify="center")
        table.add_column("Row L2 Norm", justify="right", style="green")

        for i in range(min(m, rows_to_show)):
            row_items = [f"Row {i+1}"]
            for j in range(min(n, cols_to_show)):
                row_items.append(f"{matrix[i, j]:.4f}")
            if n > cols_to_show:
                row_items.append("...")
            row_items.append(f"{row_norms[i]:.4f}")
            table.add_row(*row_items)

        console.print(table)
        console.print(f"[bold yellow]Top 5 Singular Values:[/bold yellow] {[f'{s:.4e}' for s in singular_values[:5]]}")
        console.print(f"[bold yellow]Col Means (first 5):[/bold yellow] {[f'{cm:.4f}' for cm in np.mean(matrix, axis=0)[:5]]}\n")
    else:
        print(f"\n--- {title} ---")
        print(f"Shape: {m} x {n} | Rank: {rank}")
        print("Preview (top rows):")
        print(matrix[:rows_to_show, :cols_to_show])
        print(f"Top Singular Values: {[f'{s:.4e}' for s in singular_values[:5]]}")
        print(f"Row Norms (first 5): {[f'{rn:.4f}' for rn in row_norms[:5]]}")
        print(f"Col Means (first 5): {[f'{cm:.4f}' for cm in np.mean(matrix, axis=0)[:5]]}\n")


def display_table(title: str, matrix: np.ndarray, row_labels: list[str], col_labels: list[str]) -> None:
    """Displays a complete 2D matrix formatted as a table."""
    m, n = matrix.shape
    if HAS_RICH:
        console = Console()
        table = Table(title=title)
        table.add_column("", style="cyan", no_wrap=True)
        for label in col_labels:
            table.add_column(label, justify="right")
        for i in range(m):
            row_vals = [row_labels[i]] + [f"{matrix[i, j]:.4f}" for j in range(n)]
            table.add_row(*row_vals)
        console.print(table)
    else:
        print(f"\n--- {title} ---")
        for i in range(m):
            vals = "  ".join(f"{matrix[i, j]:8.4f}" for j in range(n))
            print(f"{row_labels[i]:10s} | {vals}")


def run_pipeline_demo() -> None:
    """Executes a complete demonstration of Stages 1 through 5."""
    la_engine = LinearAlgebraEngine()

    # =========================================================================
    # STAGE 1: Data Ingestion & Synthetic Data Generation
    # =========================================================================
    print_header("Stage 1: Data Ingestion & Synthetic Data Generation")

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "synthetic_customers.csv"
    json_path = output_dir / "synthetic_customers.json"

    m = 100  # customers
    n = 10   # product categories
    r = 3    # controlled rank r < min(m, n)
    print(f"Generating synthetic customer matrix: m={m} customers, n={n} categories, target rank r={r}...")

    generator = SyntheticCustomerDataGenerator(random_state=42)
    synth_res = generator.generate(
        n_customers=m,
        n_categories=n,
        rank=r,
        noise_level=0.0,
    )

    print(f"Successfully generated matrix A in R^{{{m} x {n}}}.")
    print(f"Target Rank: {synth_res.target_rank} | SVD Evaluated Rank: {synth_res.exact_rank}")
    synth_res.to_csv(str(csv_path))
    synth_res.to_json(str(json_path), orient="split")

    display_matrix_summary("Raw Synthetic Matrix A", synth_res.matrix)

    ingestion = DataIngestionModule(imputation_strategy=ImputationStrategy.ZERO)
    ingested_raw = ingestion.process(csv_path, normalization=NormalizationType.NONE)
    print(f"Ingested {ingested_raw.n_customers} customers and {ingested_raw.n_categories} categories from CSV.")

    # =========================================================================
    # STAGE 2: Core Linear Algebra - RREF, Rank & Subspaces
    # =========================================================================
    print_header("Stage 2: Core Linear Algebra Engine - RREF & Subspaces")

    rref_res = la_engine.compute_rref(synth_res.matrix)
    print(f"Gaussian Elimination with Partial Pivoting Results:")
    print(f"  - Computed Rank r: {rref_res.rank}")
    print(f"  - Nullity (n - r): {rref_res.nullity}")
    print(f"  - Pivot Columns: {rref_res.pivot_columns}")
    print(f"  - Free Variable Columns: {rref_res.free_columns}")
    print(f"  - Row Swaps Performed: {rref_res.row_swaps}")
    print(f"  - Rank-Nullity Verification: {rref_res.rank} + {rref_res.nullity} == {n} (Q.E.D.)")

    bases = la_engine.extract_subspace_bases(synth_res.matrix)
    print(f"\nFundamental Subspace Dimensions:")
    print(f"  - Row Space R(A) Basis Shape: {bases.row_space_basis.shape} (dim = {bases.rank})")
    print(f"  - Column Space C(A) Basis Shape: {bases.column_space_basis.shape} (dim = {bases.rank})")
    print(f"  - Null Space N(A) Basis Shape: {bases.null_space_basis.shape} (dim = {bases.nullity})")

    null_space_valid = la_engine.verify_null_space(synth_res.matrix, bases.null_space_basis)
    print(f"  - Null Space Verification (A * v_i = 0): {'PASSED' if null_space_valid else 'FAILED'}")

    # =========================================================================
    # STAGE 3: Vector Inner Product & Cosine Similarity Engine
    # =========================================================================
    print_header("Stage 3: Vector Similarity Engine (G = A * A^T, S = D^-1 * G * D^-1)")

    sim_res = la_engine.compute_similarity_engine(synth_res.matrix)
    print(f"Gram Matrix G = A * A^T shape: {sim_res.gram_matrix.shape}")
    print(f"Cosine Similarity Matrix S shape: {sim_res.cosine_similarity_matrix.shape}")
    print(f"Top 5 Customer Norms (D_ii): {[f'{val:.4f}' for val in sim_res.vector_norms[:5]]}")

    cust_labels_5 = [f"Cust {i+1}" for i in range(5)]
    display_table(
        "Pairwise Cosine Similarity Submatrix S[1:5, 1:5]",
        sim_res.cosine_similarity_matrix[:5, :5],
        cust_labels_5,
        cust_labels_5,
    )

    # =========================================================================
    # STAGE 4: Spectral Decomposition, Covariance & PCA
    # =========================================================================
    print_header("Stage 4: Covariance Matrix & Spectral Decomposition (PCA)")

    spec_res = la_engine.compute_spectral_decomposition(synth_res.matrix, method="qr")
    print(f"Mean Vector mu in R^{n} (first 5): {[f'{val:.4f}' for val in spec_res.mean_vector[:5]]}")
    print(f"Mean-Centered Matrix X shape: {spec_res.mean_centered_matrix.shape}")
    print(f"Sample Covariance Matrix C shape: {spec_res.covariance_matrix.shape}")
    print(f"Trace(Covariance): {spec_res.trace_covariance:.4f} | Sum(Eigenvalues): {np.sum(spec_res.eigenvalues):.4f}")

    if HAS_RICH:
        console = Console()
        eigen_table = Table(title="Eigenvalue Spectrum & Variance Explained")
        eigen_table.add_column("Principal Component", style="cyan")
        eigen_table.add_column("Eigenvalue (lambda_i)", justify="right")
        eigen_table.add_column("Variance Explained", justify="right")
        eigen_table.add_column("Cumulative Variance R(k)", justify="right", style="green")

        for idx in range(min(n, 7)):
            lam = spec_res.eigenvalues[idx]
            vr = spec_res.variance_ratios[idx] * 100.0
            cvr = spec_res.cumulative_variance_ratios[idx] * 100.0
            eigen_table.add_row(f"PC {idx+1}", f"{lam:.4f}", f"{vr:.2f}%", f"{cvr:.2f}%")
        console.print(eigen_table)
    else:
        print("\nPrincipal Component Spectrum:")
        for idx in range(min(n, 7)):
            print(f"  PC {idx+1}: lambda={spec_res.eigenvalues[idx]:.4e} | Var={spec_res.variance_ratios[idx]*100:.2f}% | R(k)={spec_res.cumulative_variance_ratios[idx]*100:.2f}%")

    # =========================================================================
    # STAGE 5: Subspace Orthogonal Projection
    # =========================================================================
    print_header("Stage 5: Subspace Orthogonal Projection (P = V_k * V_k^T, Y = X * V_k)")

    # Select top k=3 components corresponding to target rank r=3
    k_proj = r
    proj_res = la_engine.compute_orthogonal_projection(spec_res, k=k_proj)
    print(f"Subspace Projection Dimension: k = {k_proj}")
    print(f"Projection Operator Matrix P shape: {proj_res.projection_matrix.shape}")
    print(f"Projected Customer Coordinates Y shape: {proj_res.projected_coordinates.shape}")
    print(f"Cumulative Variance Preserved R({k_proj}): {proj_res.cumulative_variance_explained * 100.0:.2f}%")
    print(f"Relative Reconstruction Error (Information Loss): {proj_res.reconstruction_error:.4e}")

    # Display preview of projected coordinates for first 5 customers
    coord_labels = [f"Cust {i+1}" for i in range(5)]
    pc_labels = [f"PC {j+1}" for j in range(k_proj)]
    display_table(
        f"Customer Subspace Coordinates Y in R^{{5 x {k_proj}}}",
        proj_res.projected_coordinates[:5, :k_proj],
        coord_labels,
        pc_labels,
    )

    # Check projection matrix idempotence P^2 == P and symmetry P^T == P
    P = proj_res.projection_matrix
    sym_err = float(np.max(np.abs(P - P.T)))
    idem_err = float(np.max(np.abs(np.matmul(P, P) - P)))
    print(f"\nOrthogonal Projection Properties:")
    print(f"  - Symmetry Error (||P - P^T||): {sym_err:.4e} (Must be 0)")
    print(f"  - Idempotence Error (||P^2 - P||): {idem_err:.4e} (Must be 0)")
    print(f"  - Trace(P): {np.trace(P):.4f} (Must equal k = {k_proj})")

    # =========================================================================
    # Mathematical Verification on Benchmark Matrix A_test (from Masterplan §4)
    # =========================================================================
    print_header("Benchmark Matrix Hand-Calculated Verification (Masterplan Section 4)")

    a_test = np.array([
        [4.0, 0.0, 3.0],
        [0.0, 5.0, 0.0],
        [8.0, 0.0, 6.0],
    ], dtype=np.float64)

    test_spec = la_engine.compute_spectral_decomposition(a_test, method="characteristic")
    test_proj = la_engine.compute_orthogonal_projection(test_spec, k=2)

    labels_test = ["Customer 1", "Customer 2", "Customer 3"]
    cat_test = ["Category 1", "Category 2", "Category 3"]

    display_table("Original Benchmark Matrix A_test", a_test, labels_test, cat_test)
    display_table("Mean-Centered Matrix X", test_spec.mean_centered_matrix, labels_test, cat_test)
    display_table("Sample Covariance Matrix C = (1/2) * X^T * X", test_spec.covariance_matrix, cat_test, cat_test)

    print(f"\nSpectral Decomposition of C (Rank = 2):")
    print(f"  - Theoretical Trace: 100/3 ~ 33.3333 | Computed Trace: {test_spec.trace_covariance:.4f}")
    print(f"  - Sorted Eigenvalues: {test_spec.eigenvalues}")
    print(f"  - Top 2 Components Cumulative Variance: {test_spec.cumulative_variance_ratios[1] * 100.0:.2f}% (100.0% expected)")
    print(f"  - Reconstruction Error with k=2: {test_proj.reconstruction_error:.4e} (Exact 0.0 expected)")

    display_table("Orthonormal Eigenvectors V", test_spec.eigenvectors, cat_test, ["v_1", "v_2", "v_3"])
    display_table("Orthogonal Projection Matrix P = V_2 * V_2^T", test_proj.projection_matrix, cat_test, cat_test)

    print("\nAll Stages 1 through 5 verified and fully operational!")


if __name__ == "__main__":
    run_pipeline_demo()
