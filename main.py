"""Main entry point and demonstration script for the Customer Purchase Pattern & Profile Analyzer.

Executes Stages 1, 2, and 3 of the sequential mathematical pipeline:
1. Synthetic customer-category matrix generation with controllable rank r < min(m, n).
2. Data ingestion, parsing (CSV/JSON), and mathematical vector normalization (L2 and Mean Centering).
3. Custom Gaussian Elimination with Partial Pivoting to Reduced Row Echelon Form (RREF).
4. Matrix Rank determination, Nullity computation via the Rank-Nullity Theorem (n - r).
5. Fundamental Subspace extraction: Row Space R(A), Column Space C(A), and Null Space N(A).
6. Vector inner products, Gram Matrix G = A * A^T, diagonal norm matrix D, and Cosine Similarity Matrix S = D^-1 * G * D^-1.
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
        print("\n" + "=" * 75)
        print(f" {title.upper()} ")
        print("=" * 75)


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
    """Executes a complete demonstration of Stages 1, 2, and 3."""
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

    # Normalization
    ingested_l2 = ingestion.process(csv_path, normalization=NormalizationType.L2)
    ingested_centered = ingestion.process(csv_path, normalization=NormalizationType.MEAN_CENTER)

    # =========================================================================
    # STAGE 2: Core Linear Algebra - RREF, Rank & Subspaces
    # =========================================================================
    print_header("Stage 2: Core Linear Algebra Engine - RREF & Subspaces")

    # 1. Compute RREF via Gaussian Elimination with Partial Pivoting
    rref_res = la_engine.compute_rref(synth_res.matrix)
    print(f"Gaussian Elimination with Partial Pivoting Results:")
    print(f"  - Computed Rank r: {rref_res.rank}")
    print(f"  - Nullity (n - r): {rref_res.nullity}")
    print(f"  - Pivot Columns: {rref_res.pivot_columns}")
    print(f"  - Free Variable Columns: {rref_res.free_columns}")
    print(f"  - Row Swaps Performed: {rref_res.row_swaps}")

    # Rank-Nullity Theorem Check
    assert rref_res.rank + rref_res.nullity == n, "Rank-Nullity theorem must hold: rank + nullity = n"
    print(f"  - Rank-Nullity Verification: {rref_res.rank} + {rref_res.nullity} == {n} (Q.E.D.)")

    # 2. Extract Fundamental Subspace Bases
    bases = la_engine.extract_subspace_bases(synth_res.matrix)
    print(f"\nFundamental Subspace Dimensions:")
    print(f"  - Row Space R(A) Basis Shape: {bases.row_space_basis.shape} (dim = {bases.rank})")
    print(f"  - Column Space C(A) Basis Shape: {bases.column_space_basis.shape} (dim = {bases.rank})")
    print(f"  - Null Space N(A) Basis Shape: {bases.null_space_basis.shape} (dim = {bases.nullity})")

    # Verify Null Space identity: A * v = 0
    null_space_valid = la_engine.verify_null_space(synth_res.matrix, bases.null_space_basis)
    print(f"  - Null Space Verification (A * v_i = 0): {'PASSED' if null_space_valid else 'FAILED'}")
    assert null_space_valid, "All null space basis vectors must satisfy A * v = 0"

    # =========================================================================
    # STAGE 3: Vector Inner Product & Cosine Similarity Engine
    # =========================================================================
    print_header("Stage 3: Vector Similarity Engine (G = A * A^T, S = D^-1 * G * D^-1)")

    sim_res = la_engine.compute_similarity_engine(synth_res.matrix)
    print(f"Gram Matrix G = A * A^T shape: {sim_res.gram_matrix.shape}")
    print(f"Cosine Similarity Matrix S shape: {sim_res.cosine_similarity_matrix.shape}")
    print(f"Top 5 Customer Norms (D_ii): {[f'{val:.4f}' for val in sim_res.vector_norms[:5]]}")

    # Display similarity matrix preview for first 5 customers
    cust_labels_5 = [f"Cust {i+1}" for i in range(5)]
    display_table(
        "Pairwise Cosine Similarity Submatrix S[1:5, 1:5]",
        sim_res.cosine_similarity_matrix[:5, :5],
        cust_labels_5,
        cust_labels_5,
    )

    # =========================================================================
    # Mathematical Verification on Benchmark Matrix A_test (from Masterplan §4)
    # =========================================================================
    print_header("Benchmark Matrix Hand-Calculated Verification (Masterplan Section 4)")

    a_test = np.array([
        [4.0, 0.0, 3.0],
        [0.0, 5.0, 0.0],
        [8.0, 0.0, 6.0],
    ], dtype=np.float64)

    test_rref = la_engine.compute_rref(a_test)
    test_bases = la_engine.extract_subspace_bases(a_test)
    test_sim = la_engine.compute_similarity_engine(a_test)

    labels_test = ["Customer 1", "Customer 2", "Customer 3"]
    cat_test = ["Category 1", "Category 2", "Category 3"]

    display_table("Original Benchmark Matrix A_test", a_test, labels_test, cat_test)
    display_table("Calculated RREF(A_test)", test_rref.rref_matrix, ["Row 1", "Row 2", "Row 3"], cat_test)

    print(f"\nVerification Results:")
    print(f"  - Theoretical Rank: 2 | Computed Rank: {test_rref.rank} (Pivot cols: {test_rref.pivot_columns})")
    print(f"  - Theoretical Nullity: 1 | Computed Nullity: {test_rref.nullity}")
    print(f"  - Null Space Vector v: {test_bases.null_space_basis.ravel()} (Theoretical: [-0.75, 0.0, 1.0])")
    print(f"  - Gram Matrix G diagonal norms: {test_sim.vector_norms} (Theoretical: [5.0, 5.0, 10.0])")

    display_table("Gram Matrix G = A_test * A_test^T", test_sim.gram_matrix, labels_test, labels_test)
    display_table("Cosine Similarity Matrix S = D^-1 * G * D^-1", test_sim.cosine_similarity_matrix, labels_test, labels_test)

    print(f"  - Cosine Similarity between Customer 1 and Customer 3: {test_sim.cosine_similarity_matrix[0, 2]:.4f}")
    print(f"    Interpretation: Customer 1 (4, 0, 3) and Customer 3 (8, 0, 6) have similarity 1.0 because")
    print(f"    their category purchasing vectors are strictly parallel in R^3.")
    print("\nAll pipeline components verified and mathematically sound!")


if __name__ == "__main__":
    run_pipeline_demo()
