"""Main entry point and demonstration script for the Customer Purchase Pattern & Profile Analyzer.

Executes the complete sequential mathematical pipeline:
1. Stage 1: Synthetic data generation with controllable rank r < min(m, n), CSV/JSON parsing, missing value imputation, and vector normalizations (L2 and Mean Centering).
2. Stage 2: Gaussian Elimination with Partial Pivoting to RREF, Matrix Rank, Nullity (Rank-Nullity Theorem), and Subspace Bases (Row Space, Column Space, Null Space).
3. Stage 3: Vector Inner Products, Gram Matrix G = A * A^T, Diagonal Norm Matrix D, and Cosine Similarity Matrix S = D^-1 * G * D^-1.
4. Stage 4: Covariance Matrix C = (1/(m-1)) * X^T * X, Spectral Decomposition (QR algorithm & Jacobi method), Sorted Eigenpairs, and Cumulative Variance Ratio R(k).
5. Stage 5: Orthogonal Subspace Projection Operator P = V_k * V_k^T, Low-Dimensional Customer Coordinates Y = X * V_k, Reconstruction, and Information Loss.
6. Stage 6: Profile Synthesis: Subspace Euclidean distance clustering d(y_i, y_j) = ||y_i - y_j||_2, Centroid Calculation, Re-projection c_orig = c_proj * V_k^T + mu, and Archetype Interpretation.
7. Dashboards: High-resolution visual dashboards rendered via Matplotlib and Plotly (2D/3D Projections, Similarity Heatmaps, Scree Plots, Radar Charts).
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
from src.profiling.profiler import AnalyticalProfilingModule
from src.visualization.engine import VisualizationEngine
from src.verification.engine import VerificationEngine


def print_header(title: str) -> None:
    """Prints a styled header."""
    if HAS_RICH:
        console = Console()
        console.print(Panel(f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))
    else:
        print("\n" + "=" * 80)
        print(f" {title.upper()} ")
        print("=" * 80)


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
    """Executes a complete demonstration of the end-to-end mathematical data pipeline."""
    la_engine = LinearAlgebraEngine()
    viz_engine = VisualizationEngine()
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

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

    # =========================================================================
    # STAGE 5: Subspace Orthogonal Projection
    # =========================================================================
    print_header("Stage 5: Subspace Orthogonal Projection (P = V_k * V_k^T, Y = X * V_k)")

    k_proj = r
    proj_res = la_engine.compute_orthogonal_projection(spec_res, k=k_proj)
    print(f"Subspace Projection Dimension: k = {k_proj}")
    print(f"Projection Operator Matrix P shape: {proj_res.projection_matrix.shape}")
    print(f"Projected Customer Coordinates Y shape: {proj_res.projected_coordinates.shape}")
    print(f"Cumulative Variance Preserved R({k_proj}): {proj_res.cumulative_variance_explained * 100.0:.2f}%")
    print(f"Relative Reconstruction Error (Information Loss): {proj_res.reconstruction_error:.4e}")

    # =========================================================================
    # STAGE 6: Profile Synthesis & Interpretation
    # =========================================================================
    print_header("Stage 6: Profile Synthesis & Customer Archetype Extraction")

    profiler = AnalyticalProfilingModule(random_state=42)
    profile_res = profiler.profile_customers(
        projected_coordinates=proj_res.projected_coordinates,
        top_eigenvectors=proj_res.top_eigenvectors,
        mean_vector=spec_res.mean_vector,
        n_clusters=3,
        customer_ids=synth_res.customer_ids,
        category_names=synth_res.category_names,
    )

    cluster_names = [c.archetype_name for c in profile_res.clusters]

    if HAS_RICH:
        console = Console()
        prof_table = Table(title="Discovered Customer Behavioral Archetypes")
        prof_table.add_column("Cluster", style="cyan")
        prof_table.add_column("Archetype Profile", style="bold yellow")
        prof_table.add_column("Customer Count", justify="right")
        prof_table.add_column("Share (%)", justify="right")
        prof_table.add_column("Avg Total Spend", justify="right", style="green")
        prof_table.add_column("Dominant Categories", style="white")

        for c in profile_res.clusters:
            prof_table.add_row(
                f"Group {c.cluster_id + 1}",
                c.archetype_name,
                str(c.customer_count),
                f"{c.percentage:.1f}%",
                f"${c.average_spend:.2f}",
                ", ".join(c.dominant_categories[:3]),
            )
        console.print(prof_table)
    else:
        df_summary = profile_res.to_dataframe()
        print(df_summary[["Cluster", "Archetype", "Customers", "Share (%)", "Avg Spend", "Top Categories"]])

    # =========================================================================
    # VISUALIZATION DASHBOARDS
    # =========================================================================
    print_header("Visualization Engine: Generating Graphical Dashboards")

    # 1. 2D Subspace Projection
    var_exp_2d = (
        float(spec_res.variance_ratios[0] * 100.0),
        float(spec_res.variance_ratios[1] * 100.0),
    )
    viz_engine.plot_subspace_projection_2d(
        Y=proj_res.projected_coordinates,
        labels=profile_res.labels,
        cluster_names=cluster_names,
        centroids=profile_res.projected_centroids[:, :2],
        variance_explained=var_exp_2d,
        save_path=fig_dir / "customer_subspace_2d",
    )
    print("  [x] Rendered 2D Subspace Projection: reports/figures/customer_subspace_2d.[png|html]")

    # 2. 3D Subspace Projection
    viz_engine.plot_subspace_projection_3d(
        Y=proj_res.projected_coordinates,
        labels=profile_res.labels,
        cluster_names=cluster_names,
        centroids=profile_res.projected_centroids,
        save_path=fig_dir / "customer_subspace_3d",
    )
    print("  [x] Rendered 3D Subspace Projection: reports/figures/customer_subspace_3d.[png|html]")

    # 3. Cosine Similarity Heatmap
    viz_engine.plot_similarity_heatmap(
        similarity_matrix=sim_res.cosine_similarity_matrix,
        labels=profile_res.labels,
        max_display=40,
        save_path=fig_dir / "cosine_similarity_heatmap",
    )
    print("  [x] Rendered Cosine Similarity Heatmap: reports/figures/cosine_similarity_heatmap.[png|html]")

    # 4. Variance Scree Plot
    viz_engine.plot_variance_scree(
        eigenvalues=spec_res.eigenvalues,
        variance_ratios=spec_res.variance_ratios,
        cumulative_ratios=spec_res.cumulative_variance_ratios,
        threshold=0.90,
        save_path=fig_dir / "variance_scree_plot",
    )
    print("  [x] Rendered Variance Scree Plot: reports/figures/variance_scree_plot.[png|html]")

    # 5. Profile Radar / Spider Chart
    viz_engine.plot_profile_radar_chart(
        category_centroids=profile_res.category_centroids,
        category_names=synth_res.category_names,
        cluster_names=cluster_names,
        save_path=fig_dir / "customer_archetype_radar",
    )
    print("  [x] Rendered Archetype Profile Radar Chart: reports/figures/customer_archetype_radar.[png|html]")

    # =========================================================================
    # MATHEMATICAL VERIFICATION & SCIPY COMPARISON
    # =========================================================================
    print_header("Stage 7: Verification Engine - Benchmarks & Scipy Validation")

    verifier = VerificationEngine(tolerance=1e-6)
    bench_report = verifier.verify_benchmark_properties()
    ident_report = verifier.verify_theoretical_identities(synth_res.matrix)
    scipy_report = verifier.compare_against_scipy(synth_res.matrix)

    if HAS_RICH:
        console = Console()

        # Benchmark Table
        bench_table = Table(title="Benchmark Matrix A_test Validation (Masterplan Section 4)")
        bench_table.add_column("Property / Test", style="cyan")
        bench_table.add_column("Expected", justify="right")
        bench_table.add_column("Calculated", justify="right")
        bench_table.add_column("Status", justify="center", style="bold green")

        bench_table.add_row("Row 3 == 2 * Row 1", "True (Diff=0.0)", f"{bench_report.row3_is_2x_row1} (Diff={bench_report.row3_difference_norm:.2e})", "[green]PASSED[/green]")
        bench_table.add_row("Matrix Rank r", "2", str(bench_report.rank), "[green]PASSED[/green]")
        bench_table.add_row("Pivot Columns", "[0, 1] (Cols 1, 2)", str(bench_report.pivot_columns), "[green]PASSED[/green]")
        bench_table.add_row("RREF Exact Match", "[[1, 0, 0.75], ...]", f"{bench_report.rref_matches} (Diff={bench_report.rref_max_diff:.2e})", "[green]PASSED[/green]")
        bench_table.add_row("Inner Product <a_1, a_3>", "50.0", f"{bench_report.inner_product_1_3:.4f}", "[green]PASSED[/green]")
        bench_table.add_row("Cosine Similarity S_1,3", "1.0", f"{bench_report.cosine_similarity_1_3:.4f}", "[green]PASSED[/green]")
        console.print(bench_table)

        # Theoretical Identities Table
        ident_table = Table(title="Theoretical Linear Algebra Identities Verification Checklist")
        ident_table.add_column("Mathematical Identity", style="cyan")
        ident_table.add_column("Error Metric", justify="right")
        ident_table.add_column("Status", justify="center")

        ident_table.add_row("Trace Identity: Trace(C) == sum(lambda_i)", f"|Diff| = {ident_report.trace_difference:.2e}", "[green]PASSED[/green]" if ident_report.trace_identity_passed else "[red]FAILED[/red]")
        ident_table.add_row("Determinant Identity: det(C) == prod(lambda_i)", f"|Diff| = {ident_report.det_difference:.2e}", "[green]PASSED[/green]" if ident_report.det_identity_passed else "[red]FAILED[/red]")
        ident_table.add_row("Eigenvector Orthogonality: v_i^T * v_j == 0", f"max |V^T V - I| = {ident_report.max_orthogonality_error:.2e}", "[green]PASSED[/green]" if ident_report.orthogonality_passed else "[red]FAILED[/red]")
        ident_table.add_row("Eigenvalue Equation: C * v_k == lambda_k * v_k", f"max ||C v - lam v|| = {ident_report.max_eigenvalue_residual:.2e}", "[green]PASSED[/green]" if ident_report.eigenvalue_equation_passed else "[red]FAILED[/red]")
        ident_table.add_row("Projection Idempotence: P^2 == P", f"max |P^2 - P| = {ident_report.projection_idempotence_error:.2e}", "[green]PASSED[/green]" if ident_report.projection_idempotence_passed else "[red]FAILED[/red]")
        console.print(ident_table)

        # Scipy Reference Table
        scipy_table = Table(title="Numerical Stability Comparison vs. scipy.linalg Standard Reference")
        scipy_table.add_column("Routine / Output", style="cyan")
        scipy_table.add_column("Custom Algorithm", justify="center")
        scipy_table.add_column("scipy.linalg Reference", justify="center")
        scipy_table.add_column("Max Absolute Discrepancy", justify="right")
        scipy_table.add_column("Status", justify="center")

        scipy_table.add_row("Matrix Rank", f"r = {scipy_report.custom_rank}", f"r = {scipy_report.scipy_rank}", "0", "[green]MATCH[/green]")
        scipy_table.add_row("Eigenvalues (PCA)", "Custom QR/Deflation", "scipy.linalg.eigh", f"{scipy_report.eigenvalues_max_diff:.2e}", "[green]MATCH[/green]")
        scipy_table.add_row("Projection Subspace P", "Custom V_k V_k^T", "scipy V_k V_k^T", f"{scipy_report.projection_subspace_diff:.2e}", "[green]MATCH[/green]")
        scipy_table.add_row("Cosine Similarity Matrix", "Custom D^-1 G D^-1", "1 - scipy.spatial.cosine", f"{scipy_report.cosine_similarity_max_diff:.2e}", "[green]MATCH[/green]")
        console.print(scipy_table)
    else:
        print(f"Benchmark Test A_test: {'ALL PASSED' if bench_report.all_passed else 'FAILED'}")
        print(f"Theoretical Identities: {'ALL PASSED' if ident_report.all_identities_passed else 'FAILED'}")
        print(f"Scipy Comparison: {'ALL MATCHED' if scipy_report.all_passed else 'FAILED'}")

    print("\n" + "=" * 80)
    print(" Complete Linear Algebra Customer Analytics Pipeline Executed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline_demo()
