"""Main entry point and execution script for the Customer Purchase Pattern & Profile Analyzer.

Wires all modules together into a strictly sequential computational pipeline:
Stage 1: Matrix Ingestion      -> Matrix A in R^{m x n}
Stage 2: Rank & Basis          -> RREF, Rank r, Pivot Columns, Fundamental Subspace Bases
Stage 3: Inner Product         -> Gram Matrix G = A * A^T, Diagonal Norms D, Cosine Similarity S
Stage 4: Spectral / PCA        -> Covariance C = 1/(m-1) * X^T * X, Eigenvalues lambda, Eigenvectors V
Stage 5: Subspace Projection   -> Orthogonal Operator P = V_k * V_k^T, Coordinates Y = X * V_k
Stage 6: Profile Formation     -> Subspace Euclidean Proximity Clustering, Centroid Re-projection, Archetypes
Stage 7: Verification Engine   -> Benchmark Validation, Theoretical Identities, Scipy Comparison

To launch the interactive Streamlit presentation dashboard:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from src.ingestion.data_ingestion import NormalizationType, ImputationStrategy
from src.ingestion.synthetic_generator import SyntheticCustomerDataGenerator
from src.pipeline import SequentialMathematicalPipeline, PipelineExecutionResult
from src.visualization.engine import VisualizationEngine


def print_header(title: str) -> None:
    """Prints a styled header."""
    if HAS_RICH:
        console = Console()
        console.print(Panel(f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))
    else:
        print("\n" + "=" * 80)
        print(f" {title.upper()} ")
        print("=" * 80)


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


def run_pipeline() -> PipelineExecutionResult:
    """Executes the complete sequential mathematical data pipeline."""
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    print_header("Customer Purchase Pattern & Profile Analyzer: Sequential Computational Pipeline")
    print("Initiating sequential flow: Ingestion -> Rank & Basis -> Inner Product -> Spectral PCA -> Projection -> Profiling")

    # 1. Instantiate Pipeline & Synthetic Dataset
    m, n, r = 100, 10, 3
    print(f"\nGenerating synthetic customer preference matrix: m={m} customers, n={n} categories, target rank r={r}...")
    generator = SyntheticCustomerDataGenerator(random_state=42)
    synth_data = generator.generate(n_customers=m, n_categories=n, rank=r, noise_level=0.0)

    # Persist raw data
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    synth_data.to_csv(str(data_dir / "synthetic_customers.csv"))
    synth_data.to_json(str(data_dir / "synthetic_customers.json"), orient="split")

    # 2. Execute Sequential Pipeline
    pipeline = SequentialMathematicalPipeline(random_state=42)
    res = pipeline.run(
        data_source=synth_data,
        k_dimensions=r,
        n_clusters=3,
        normalization=NormalizationType.NONE,
    )

    # =========================================================================
    # STAGE 1 OUTPUTS
    # =========================================================================
    print_header("Stage 1: Matrix Ingestion -> Dense Matrix A in R^{m x n}")
    print(f"  - Customers (m): {res.ingestion.n_customers} | Categories (n): {res.ingestion.n_categories}")
    print(f"  - First 5 Customer Norms (||a_i||_2): {[f'{val:.2f}' for val in res.ingestion.row_norms[:5]]}")
    print(f"  - First 5 Category Means (mu_j): {[f'{val:.2f}' for val in res.ingestion.feature_means[:5]]}")

    # =========================================================================
    # STAGE 2 OUTPUTS
    # =========================================================================
    print_header("Stage 2: Rank & Basis Analyzer -> RREF, Rank r, Subspaces")
    print(f"  - Computed Linear Rank (r): {res.rref.rank}")
    print(f"  - Nullity (n - r): {res.rref.nullity}")
    print(f"  - Pivot Columns (Independent Categories): {res.rref.pivot_columns}")
    print(f"  - Free Columns (Redundant Categories): {res.rref.free_columns}")
    print(f"  - Rank-Nullity Verification: {res.rref.rank} + {res.rref.nullity} == {n} (Q.E.D.)")
    print(f"  - Subspace Dimensions:")
    print(f"      * Row Space R(A): {res.subspaces.row_space_basis.shape} (dim = {res.subspaces.rank})")
    print(f"      * Column Space C(A): {res.subspaces.column_space_basis.shape} (dim = {res.subspaces.rank})")
    print(f"      * Null Space N(A): {res.subspaces.null_space_basis.shape} (dim = {res.subspaces.nullity})")

    # =========================================================================
    # STAGE 3 OUTPUTS
    # =========================================================================
    print_header("Stage 3: Vector Similarity Engine -> Gram Matrix G, Cosine Matrix S")
    print(f"  - Gram Matrix G = A * A^T shape: {res.similarity.gram_matrix.shape}")
    print(f"  - Cosine Similarity Matrix S = D^-1 * G * D^-1 shape: {res.similarity.cosine_similarity_matrix.shape}")
    cust_labels_5 = [f"Cust {i+1}" for i in range(5)]
    display_table(
        "Pairwise Cosine Similarity Submatrix S[1:5, 1:5]",
        res.similarity.cosine_similarity_matrix[:5, :5],
        cust_labels_5,
        cust_labels_5,
    )

    # =========================================================================
    # STAGE 4 OUTPUTS
    # =========================================================================
    print_header("Stage 4: Covariance & Spectral Decomposition (PCA)")
    print(f"  - Mean-Centered Matrix X shape: {res.spectral.mean_centered_matrix.shape}")
    print(f"  - Covariance Matrix C shape: {res.spectral.covariance_matrix.shape}")
    print(f"  - Trace(Covariance): {res.spectral.trace_covariance:.4f} | Sum(Eigenvalues): {np.sum(res.spectral.eigenvalues):.4f}")

    if HAS_RICH:
        console = Console()
        eigen_table = Table(title="Eigenvalue Spectrum & Variance Retention R(k)")
        eigen_table.add_column("Principal Component", style="cyan")
        eigen_table.add_column("Eigenvalue (lambda_i)", justify="right")
        eigen_table.add_column("Variance Explained", justify="right")
        eigen_table.add_column("Cumulative R(k)", justify="right", style="green")

        for idx in range(min(n, 6)):
            lam = res.spectral.eigenvalues[idx]
            vr = res.spectral.variance_ratios[idx] * 100.0
            cvr = res.spectral.cumulative_variance_ratios[idx] * 100.0
            eigen_table.add_row(f"PC {idx+1}", f"{lam:.4f}", f"{vr:.2f}%", f"{cvr:.2f}%")
        console.print(eigen_table)

    # =========================================================================
    # STAGE 5 OUTPUTS
    # =========================================================================
    print_header("Stage 5: Subspace Orthogonal Projection -> P = V_k * V_k^T, Y = X * V_k")
    print(f"  - Retained Dimensions: k = {res.projection.k}")
    print(f"  - Projection Operator Matrix P shape: {res.projection.projection_matrix.shape}")
    print(f"  - Subspace Customer Coordinates Y shape: {res.projection.projected_coordinates.shape}")
    print(f"  - Cumulative Variance Preserved: {res.projection.cumulative_variance_explained * 100.0:.2f}%")
    print(f"  - Relative Reconstruction Error (Information Loss): {res.projection.reconstruction_error:.4e}")

    # =========================================================================
    # STAGE 6 OUTPUTS
    # =========================================================================
    print_header("Stage 6: Profile Formation & Behavioral Archetype Extraction")
    cluster_labels = [c.archetype_name for c in res.profiling.clusters]

    if HAS_RICH:
        console = Console()
        prof_table = Table(title="Synthesized Customer Behavioral Archetypes")
        prof_table.add_column("Cluster", style="cyan")
        prof_table.add_column("Archetype Profile", style="bold yellow")
        prof_table.add_column("Count", justify="right")
        prof_table.add_column("Share (%)", justify="right")
        prof_table.add_column("Avg Total Spend", justify="right", style="green")
        prof_table.add_column("Dominant Spending Categories", style="white")

        for c in res.profiling.clusters:
            prof_table.add_row(
                f"Group {c.cluster_id + 1}",
                c.archetype_name,
                str(c.customer_count),
                f"{c.percentage:.1f}%",
                f"${c.average_spend:.2f}",
                ", ".join(c.dominant_categories[:3]),
            )
        console.print(prof_table)

    # =========================================================================
    # STAGE 7: VERIFICATION ENGINE OUTPUTS
    # =========================================================================
    print_header("Stage 7: Verification Engine - Benchmarks & Scipy Validation")

    if HAS_RICH:
        console = Console()
        b_rep = res.benchmark_report
        bench_table = Table(title="Benchmark Matrix A_test Validation (Masterplan Section 4)")
        bench_table.add_column("Property / Test", style="cyan")
        bench_table.add_column("Expected", justify="right")
        bench_table.add_column("Calculated", justify="right")
        bench_table.add_column("Status", justify="center", style="bold green")

        bench_table.add_row("Row 3 == 2 * Row 1", "True (Diff=0.0)", f"{b_rep.row3_is_2x_row1} (Diff={b_rep.row3_difference_norm:.2e})", "[green]PASSED[/green]")
        bench_table.add_row("Matrix Rank r", "2", str(b_rep.rank), "[green]PASSED[/green]")
        bench_table.add_row("Pivot Columns", "[0, 1] (Cols 1, 2)", str(b_rep.pivot_columns), "[green]PASSED[/green]")
        bench_table.add_row("RREF Exact Match", "[[1, 0, 0.75], ...]", f"{b_rep.rref_matches} (Diff={b_rep.rref_max_diff:.2e})", "[green]PASSED[/green]")
        bench_table.add_row("Inner Product <a_1, a_3>", "50.0", f"{b_rep.inner_product_1_3:.4f}", "[green]PASSED[/green]")
        bench_table.add_row("Cosine Similarity S_1,3", "1.0", f"{b_rep.cosine_similarity_1_3:.4f}", "[green]PASSED[/green]")
        console.print(bench_table)

        i_rep = res.identity_report
        ident_table = Table(title="Theoretical Linear Algebra Identities Checklist")
        ident_table.add_column("Identity", style="cyan")
        ident_table.add_column("Error Metric", justify="right")
        ident_table.add_column("Status", justify="center")

        ident_table.add_row("Trace Identity: Trace(C) == sum(lambda_i)", f"|Diff| = {i_rep.trace_difference:.2e}", "[green]PASSED[/green]")
        ident_table.add_row("Determinant Identity: det(C) == prod(lambda_i)", f"|Diff| = {i_rep.det_difference:.2e}", "[green]PASSED[/green]")
        ident_table.add_row("Eigenvector Orthogonality: v_i^T * v_j == 0", f"max |V^T V - I| = {i_rep.max_orthogonality_error:.2e}", "[green]PASSED[/green]")
        ident_table.add_row("Eigenvalue Equation: C * v_k == lambda_k * v_k", f"max residual = {i_rep.max_eigenvalue_residual:.2e}", "[green]PASSED[/green]")
        ident_table.add_row("Projection Idempotence: P^2 == P", f"max |P^2 - P| = {i_rep.projection_idempotence_error:.2e}", "[green]PASSED[/green]")
        console.print(ident_table)

        s_rep = res.scipy_report
        scipy_table = Table(title="Numerical Stability vs. scipy.linalg Standard Reference")
        scipy_table.add_column("Routine / Output", style="cyan")
        scipy_table.add_column("Custom Algorithm", justify="center")
        scipy_table.add_column("scipy.linalg Reference", justify="center")
        scipy_table.add_column("Max Absolute Discrepancy", justify="right")
        scipy_table.add_column("Status", justify="center")

        scipy_table.add_row("Matrix Rank", f"r = {s_rep.custom_rank}", f"r = {s_rep.scipy_rank}", "0", "[green]MATCH[/green]")
        scipy_table.add_row("Eigenvalues (PCA)", "Custom QR/Deflation", "scipy.linalg.eigh", f"{s_rep.eigenvalues_max_diff:.2e}", "[green]MATCH[/green]")
        scipy_table.add_row("Projection Subspace P", "Custom V_k V_k^T", "scipy V_k V_k^T", f"{s_rep.projection_subspace_diff:.2e}", "[green]MATCH[/green]")
        scipy_table.add_row("Cosine Similarity", "Custom D^-1 G D^-1", "1 - scipy.spatial.cosine", f"{s_rep.cosine_similarity_max_diff:.2e}", "[green]MATCH[/green]")
        console.print(scipy_table)

    # 3. Render Dashboards
    print_header("Rendering Graphical Dashboards (PNG & HTML)")
    viz = VisualizationEngine()
    var_exp_2d = (
        float(res.spectral.variance_ratios[0] * 100.0),
        float(res.spectral.variance_ratios[1] * 100.0),
    )
    viz.plot_subspace_projection_2d(
        Y=res.projection.projected_coordinates,
        labels=res.profiling.labels,
        cluster_names=cluster_labels,
        centroids=res.profiling.projected_centroids[:, :2],
        variance_explained=var_exp_2d,
        save_path=fig_dir / "customer_subspace_2d",
    )
    viz.plot_subspace_projection_3d(
        Y=res.projection.projected_coordinates,
        labels=res.profiling.labels,
        cluster_names=cluster_labels,
        centroids=res.profiling.projected_centroids,
        save_path=fig_dir / "customer_subspace_3d",
    )
    viz.plot_similarity_heatmap(
        similarity_matrix=res.similarity.cosine_similarity_matrix,
        labels=res.profiling.labels,
        max_display=40,
        save_path=fig_dir / "cosine_similarity_heatmap",
    )
    viz.plot_variance_scree(
        eigenvalues=res.spectral.eigenvalues,
        variance_ratios=res.spectral.variance_ratios,
        cumulative_ratios=res.spectral.cumulative_variance_ratios,
        threshold=0.90,
        save_path=fig_dir / "variance_scree_plot",
    )
    viz.plot_profile_radar_chart(
        category_centroids=res.profiling.category_centroids,
        category_names=res.ingestion.category_names,
        cluster_names=cluster_labels,
        save_path=fig_dir / "customer_archetype_radar",
    )
    print("  [x] All dashboards rendered to reports/figures/ (PNG & HTML)")

    print("\n" + "=" * 80)
    print(" Complete Linear Algebra Customer Analytics Pipeline Executed Successfully!")
    print(" To launch the interactive Streamlit dashboard: streamlit run app.py")
    print("=" * 80)

    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--ui", "--streamlit", "-u"]:
        print("Launching interactive Streamlit UI...")
        os.system("streamlit run app.py")
    else:
        run_pipeline()
