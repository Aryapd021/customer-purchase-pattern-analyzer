"""Main entry point and demonstration script for the Customer Purchase Pattern & Profile Analyzer.

Executes Stage 1 of the mathematical pipeline:
1. Synthetic customer-category matrix generation with controllable rank r < min(m, n).
2. Data ingestion and parsing (CSV/JSON).
3. Missing value imputation.
4. Matrix formulation into dense NumPy array A_{m x n}.
5. Mathematical vector normalization (L2 normalization and Mean Centering).
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


def print_header(title: str) -> None:
    """Prints a styled header."""
    if HAS_RICH:
        console = Console()
        console.print(Panel(f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))
    else:
        print("\n" + "=" * 70)
        print(f" {title.upper()} ")
        print("=" * 70)


def display_matrix_summary(title: str, matrix: np.ndarray, rows_to_show: int = 5, cols_to_show: int = 5) -> None:
    """Displays a preview and key mathematical properties of a matrix."""
    m, n = matrix.shape
    rank = int(np.linalg.matrix_rank(matrix))
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


def run_pipeline_demo() -> None:
    """Executes a complete demonstration of Stage 1."""
    print_header("Stage 1: Data Ingestion & Synthetic Data Generation Pipeline")

    # 1. Setup Directories
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "synthetic_customers.csv"
    json_path = output_dir / "synthetic_customers.json"

    # 2. Generate Synthetic Matrix with Controllable Rank
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
    print(f"Prescribed Rank r: {synth_res.target_rank} | Mathematically Computed Rank: {synth_res.exact_rank}")
    assert synth_res.exact_rank == r, "Computed rank must match target rank!"

    # Export to disk
    synth_res.to_csv(str(csv_path))
    synth_res.to_json(str(json_path), orient="split")
    print(f"Persisted dataset to:\n  - CSV: {csv_path}\n  - JSON: {json_path}")

    # Display Raw Matrix
    display_matrix_summary("Raw Synthetic Matrix A", synth_res.matrix)

    # 3. Data Ingestion from CSV
    print_header("Ingesting from CSV & Missing Value Imputation")
    ingestion = DataIngestionModule(imputation_strategy=ImputationStrategy.ZERO)
    ingested_raw = ingestion.process(csv_path, normalization=NormalizationType.NONE)
    print(f"Ingested {ingested_raw.n_customers} customers and {ingested_raw.n_categories} categories.")

    # 4. Mathematical Normalization Options
    # Option A: L2 Normalization (Row-wise customer directional vector normalization)
    print_header("Mathematical Normalization: L2 Normalization (a_i / ||a_i||_2)")
    ingested_l2 = ingestion.process(csv_path, normalization=NormalizationType.L2)
    display_matrix_summary("L2 Normalized Matrix", ingested_l2.matrix)

    # Verify all row norms are exactly 1.0
    l2_norms = np.linalg.norm(ingested_l2.matrix, ord=2, axis=1)
    np.testing.assert_allclose(l2_norms, np.ones(m), atol=1e-12)
    print("Verification Passed: All customer row vectors have ||a_i||_2 == 1.0 (unit hypersphere representation).")

    # Option B: Mean Centering (X = A - 1 * mu^T)
    print_header("Mathematical Normalization: Mean Centering (X = A - 1 * mu^T)")
    ingested_centered = ingestion.process(csv_path, normalization=NormalizationType.MEAN_CENTER)
    display_matrix_summary("Mean-Centered Matrix X", ingested_centered.matrix)

    # Verify all column means are 0.0
    col_means = np.mean(ingested_centered.matrix, axis=0)
    np.testing.assert_allclose(col_means, np.zeros(n), atol=1e-12)
    print("Verification Passed: All category column means are identically 0.0 (prepared for Covariance matrix C).")

    # 5. Benchmark Matrix from Masterplan
    print_header("Hand-Calculated Benchmark Matrix Verification")
    a_benchmark = np.array([
        [4.0, 0.0, 3.0],
        [0.0, 5.0, 0.0],
        [8.0, 0.0, 6.0],
    ], dtype=np.float64)

    bm_raw = ingestion.process(a_benchmark, normalization=NormalizationType.NONE)
    bm_l2 = ingestion.process(a_benchmark, normalization=NormalizationType.L2)
    bm_mc = ingestion.process(a_benchmark, normalization=NormalizationType.MEAN_CENTER)

    print(f"Benchmark Matrix A_test:\n{a_benchmark}")
    print(f"Computed Rank: {np.linalg.matrix_rank(a_benchmark)} (Theoretical Rank: 2)")
    print(f"Row Norms: {bm_raw.row_norms} (Theoretical: [5.0, 5.0, 10.0])")
    print(f"L2 Normalized:\n{bm_l2.matrix}")
    print(f"Mean-Centered:\n{bm_mc.matrix}")
    print("\nStage 1 Data Ingestion & Matrix Formulation pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline_demo()
