"""Visualization Engine.

Renders publication-quality visualizations for the linear algebra and customer profiling pipeline:
1. 2D and 3D Subspace Projections (Matplotlib & Plotly).
2. Pairwise Cosine Similarity Heatmaps (Matplotlib/Seaborn & Plotly).
3. Variance Scree Plots with Cumulative Variance Thresholds (Matplotlib & Plotly).
4. Customer Archetype Profile Radar (Spider) Charts (Matplotlib & Plotly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for server/CLI environments
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


class VisualizationEngine:
    """Renders 2D/3D projections, heatmaps, scree plots, and radar charts."""

    def __init__(self, style: str = "seaborn-v0_8-whitegrid") -> None:
        """Initializes the visualization engine with custom styling."""
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")
        self.palette = sns.color_palette("deep", 10)

    # =========================================================================
    # 1. Subspace Projections (2D & 3D)
    # =========================================================================

    def plot_subspace_projection_2d(
        self,
        Y: np.ndarray,
        labels: Optional[np.ndarray] = None,
        cluster_names: Optional[List[str]] = None,
        centroids: Optional[np.ndarray] = None,
        variance_explained: Optional[Tuple[float, float]] = None,
        title: str = "Customer Subspace Projection (2D Principal Coordinates)",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[plt.Figure, go.Figure]:
        """Renders 2D customer projection using Matplotlib and Plotly.

        Args:
            Y: Subspace coordinates matrix in R^{m x k} (k >= 2).
            labels: Cluster assignment vector of length m.
            cluster_names: Archetype label strings for each cluster.
            centroids: Projected cluster centroids in R^{K x 2}.
            variance_explained: Variance percentage tuple for (PC1, PC2).
            title: Chart title.
            save_path: Optional output path.

        Returns:
            Tuple of (matplotlib_figure, plotly_figure).
        """
        m = Y.shape[0]
        y1, y2 = Y[:, 0], Y[:, 1]
        lbls = labels if labels is not None else np.zeros(m, dtype=int)
        unique_labels = np.unique(lbls)
        n_clusters = len(unique_labels)

        # Variance labels
        v1_str = f" ({variance_explained[0]:.1f}% var)" if variance_explained else ""
        v2_str = f" ({variance_explained[1]:.1f}% var)" if variance_explained else ""

        # --- Matplotlib Plot ---
        fig_mpl, ax = plt.subplots(figsize=(10, 7), dpi=150)
        colors = sns.color_palette("tab10", n_clusters)

        for idx, g in enumerate(unique_labels):
            mask = lbls == g
            group_name = cluster_names[g] if cluster_names and g < len(cluster_names) else f"Cluster {g+1}"
            ax.scatter(
                y1[mask],
                y2[mask],
                color=colors[idx % len(colors)],
                label=group_name,
                alpha=0.75,
                edgecolors="none",
                s=60,
            )

        if centroids is not None:
            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                color="black",
                marker="X",
                s=180,
                linewidths=2,
                edgecolors="white",
                label="Centroids (c_proj)",
                zorder=10,
            )

        ax.set_xlabel(f"Principal Component 1 (y_1){v1_str}", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"Principal Component 2 (y_2){v2_str}", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.legend(title="Customer Archetypes", loc="best", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig_mpl.savefig(p.with_suffix(".png"))

        # --- Plotly Interactive Plot ---
        fig_plotly = go.Figure()
        for idx, g in enumerate(unique_labels):
            mask = lbls == g
            group_name = cluster_names[g] if cluster_names and g < len(cluster_names) else f"Cluster {g+1}"
            fig_plotly.add_trace(
                go.Scatter(
                    x=y1[mask],
                    y=y2[mask],
                    mode="markers",
                    name=group_name,
                    marker=dict(size=8, opacity=0.8),
                    text=[f"Customer: {i+1}<br>Group: {group_name}" for i in np.where(mask)[0]],
                    hoverinfo="text+x+y",
                )
            )

        if centroids is not None:
            fig_plotly.add_trace(
                go.Scatter(
                    x=centroids[:, 0],
                    y=centroids[:, 1],
                    mode="markers+text",
                    name="Centroids",
                    marker=dict(symbol="x", size=14, color="black", line=dict(width=2, color="white")),
                    text=[f"Centroid {i+1}" for i in range(len(centroids))],
                    textposition="top center",
                )
            )

        fig_plotly.update_layout(
            title=dict(text=title, font=dict(size=18, family="Arial", color="#2c3e50")),
            xaxis_title=f"Principal Component 1{v1_str}",
            yaxis_title=f"Principal Component 2{v2_str}",
            template="plotly_white",
            legend_title_text="Archetypes",
        )

        if save_path:
            fig_plotly.write_html(str(Path(save_path).with_suffix(".html")))

        return fig_mpl, fig_plotly

    def plot_subspace_projection_3d(
        self,
        Y: np.ndarray,
        labels: Optional[np.ndarray] = None,
        cluster_names: Optional[List[str]] = None,
        centroids: Optional[np.ndarray] = None,
        title: str = "Customer Subspace Projection (3D Coordinate Space)",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[plt.Figure, go.Figure]:
        """Renders 3D customer projection using Matplotlib and Plotly."""
        if Y.shape[1] < 3:
            raise ValueError(f"3D projection requires at least 3 subspace dimensions. Provided shape: {Y.shape}")

        m = Y.shape[0]
        y1, y2, y3 = Y[:, 0], Y[:, 1], Y[:, 2]
        lbls = labels if labels is not None else np.zeros(m, dtype=int)
        unique_labels = np.unique(lbls)
        n_clusters = len(unique_labels)
        colors = sns.color_palette("tab10", n_clusters)

        # --- Matplotlib 3D ---
        fig_mpl = plt.figure(figsize=(11, 8), dpi=150)
        ax = fig_mpl.add_subplot(111, projection="3d")

        for idx, g in enumerate(unique_labels):
            mask = lbls == g
            group_name = cluster_names[g] if cluster_names and g < len(cluster_names) else f"Cluster {g+1}"
            ax.scatter(
                y1[mask],
                y2[mask],
                y3[mask],
                color=colors[idx % len(colors)],
                label=group_name,
                alpha=0.75,
                s=45,
            )

        if centroids is not None:
            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                centroids[:, 2],
                color="black",
                marker="X",
                s=200,
                label="Centroids",
            )

        ax.set_xlabel("PC 1 (y_1)", fontweight="bold")
        ax.set_ylabel("PC 2 (y_2)", fontweight="bold")
        ax.set_zlabel("PC 3 (y_3)", fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.legend(title="Archetypes", loc="best")
        plt.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig_mpl.savefig(p.with_suffix(".png"))

        # --- Plotly 3D ---
        fig_plotly = go.Figure()
        for idx, g in enumerate(unique_labels):
            mask = lbls == g
            group_name = cluster_names[g] if cluster_names and g < len(cluster_names) else f"Cluster {g+1}"
            fig_plotly.add_trace(
                go.Scatter3d(
                    x=y1[mask],
                    y=y2[mask],
                    z=y3[mask],
                    mode="markers",
                    name=group_name,
                    marker=dict(size=5, opacity=0.8),
                )
            )

        if centroids is not None:
            fig_plotly.add_trace(
                go.Scatter3d(
                    x=centroids[:, 0],
                    y=centroids[:, 1],
                    z=centroids[:, 2],
                    mode="markers+text",
                    name="Centroids",
                    marker=dict(symbol="cross", size=10, color="black"),
                    text=[f"C_{i+1}" for i in range(len(centroids))],
                )
            )

        fig_plotly.update_layout(
            title=title,
            scene=dict(
                xaxis_title="PC 1",
                yaxis_title="PC 2",
                zaxis_title="PC 3",
            ),
            template="plotly_white",
        )

        if save_path:
            fig_plotly.write_html(str(Path(save_path).with_suffix(".html")))

        return fig_mpl, fig_plotly

    # =========================================================================
    # 2. Similarity Heatmap (S = D^-1 * G * D^-1)
    # =========================================================================

    def plot_similarity_heatmap(
        self,
        similarity_matrix: np.ndarray,
        labels: Optional[np.ndarray] = None,
        max_display: int = 50,
        title: str = "Pairwise Cosine Similarity Heatmap (S = D^-1 * G * D^-1)",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[plt.Figure, go.Figure]:
        """Renders customer-by-customer cosine similarity matrix heatmap."""
        m = similarity_matrix.shape[0]
        display_m = min(m, max_display)

        # Sort customers by cluster label if provided to highlight block diagonal structure
        if labels is not None:
            sort_idx = np.argsort(labels)[:display_m]
        else:
            sort_idx = np.arange(display_m)

        S_sub = similarity_matrix[np.ix_(sort_idx, sort_idx)]
        cust_ticks = [f"C_{i+1}" for i in sort_idx]

        # --- Matplotlib / Seaborn Heatmap ---
        fig_mpl, ax = plt.subplots(figsize=(10, 8), dpi=150)
        sns.heatmap(
            S_sub,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            cbar_kws={"label": "Cosine Similarity cos(theta)"},
            xticklabels=max_display <= 30,
            yticklabels=max_display <= 30,
        )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Customer Index", fontweight="bold")
        ax.set_ylabel("Customer Index", fontweight="bold")
        plt.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig_mpl.savefig(p.with_suffix(".png"))

        # --- Plotly Heatmap ---
        fig_plotly = go.Figure(
            data=go.Heatmap(
                z=S_sub,
                x=cust_ticks,
                y=cust_ticks,
                colorscale="Viridis",
                zmin=0.0,
                zmax=1.0,
                colorbar=dict(title="Cosine Sim"),
            )
        )
        fig_plotly.update_layout(
            title=title,
            xaxis_title="Customer Index",
            yaxis_title="Customer Index",
            template="plotly_white",
        )

        if save_path:
            fig_plotly.write_html(str(Path(save_path).with_suffix(".html")))

        return fig_mpl, fig_plotly

    # =========================================================================
    # 3. Variance Scree Plot
    # =========================================================================

    def plot_variance_scree(
        self,
        eigenvalues: np.ndarray,
        variance_ratios: np.ndarray,
        cumulative_ratios: np.ndarray,
        threshold: float = 0.90,
        title: str = "Eigenvalue Spectrum & Cumulative Variance Scree Plot",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[plt.Figure, go.Figure]:
        """Renders PCA Scree plot with eigenvalue bars, cumulative variance line, and threshold."""
        n = len(eigenvalues)
        components = np.arange(1, n + 1)
        var_pcts = variance_ratios * 100.0
        cum_pcts = cumulative_ratios * 100.0

        # --- Matplotlib Plot ---
        fig_mpl, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
        ax2 = ax1.twinx()

        # Bars: Individual variance explained
        bars = ax1.bar(
            components,
            var_pcts,
            color="#3498db",
            alpha=0.75,
            edgecolor="#2980b9",
            label="Individual Variance (%)",
        )

        # Line: Cumulative variance explained
        line = ax2.plot(
            components,
            cum_pcts,
            color="#e74c3c",
            marker="o",
            linewidth=2.5,
            label="Cumulative Variance R(k)",
        )

        # Threshold line
        thresh_line = ax2.axhline(
            threshold * 100.0,
            color="#27ae60",
            linestyle="--",
            linewidth=1.8,
            label=f"Information Retention Target ({threshold*100:.0f}%)",
        )

        ax1.set_xlabel("Principal Component (k)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Variance Explained per Component (%)", color="#2980b9", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Cumulative Variance Ratio R(k) (%)", color="#c0392b", fontsize=11, fontweight="bold")
        ax1.set_xticks(components)
        ax1.set_ylim(0, max(var_pcts) * 1.25 if len(var_pcts) > 0 else 100)
        ax2.set_ylim(0, 105)

        # Add data labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 1.0:
                ax1.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True)

        ax1.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax1.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig_mpl.savefig(p.with_suffix(".png"))

        # --- Plotly Scree Plot ---
        fig_plotly = go.Figure()
        fig_plotly.add_trace(
            go.Bar(
                x=components,
                y=var_pcts,
                name="Individual Variance (%)",
                marker_color="#3498db",
                yaxis="y",
            )
        )
        fig_plotly.add_trace(
            go.Scatter(
                x=components,
                y=cum_pcts,
                name="Cumulative Variance R(k)",
                marker=dict(color="#e74c3c", size=8),
                line=dict(color="#e74c3c", width=3),
                yaxis="y2",
            )
        )
        fig_plotly.add_hline(
            y=threshold * 100.0,
            line_dash="dash",
            line_color="#27ae60",
            annotation_text=f"Target: {threshold*100:.0f}%",
            yref="y2",
        )

        fig_plotly.update_layout(
            title=title,
            xaxis=dict(title="Principal Component (k)", tickmode="linear"),
            yaxis=dict(title="Individual Variance (%)", showgrid=False),
            yaxis2=dict(
                title="Cumulative Variance Explained (%)",
                overlaying="y",
                side="right",
                range=[0, 105],
            ),
            legend=dict(x=0.6, y=0.5),
            template="plotly_white",
        )

        if save_path:
            fig_plotly.write_html(str(Path(save_path).with_suffix(".html")))

        return fig_mpl, fig_plotly

    # =========================================================================
    # 4. Profile Radar / Spider Chart
    # =========================================================================

    def plot_profile_radar_chart(
        self,
        category_centroids: np.ndarray,
        category_names: List[str],
        cluster_names: Optional[List[str]] = None,
        title: str = "Customer Behavioral Archetype Radar Profile",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[plt.Figure, go.Figure]:
        """Renders polar radar/spider chart comparing category spending profiles."""
        K, n = category_centroids.shape
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        # Close the loop
        angles += angles[:1]

        # Clean category names for display
        clean_cats = [c.replace("_", " ") for c in category_names]

        # --- Matplotlib Radar Chart ---
        fig_mpl, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True), dpi=150)
        colors = sns.color_palette("tab10", K)

        for g in range(K):
            vals = category_centroids[g].tolist()
            vals += vals[:1]
            c_name = cluster_names[g] if cluster_names and g < len(cluster_names) else f"Archetype {g+1}"
            color = colors[g % len(colors)]

            ax.plot(angles, vals, color=color, linewidth=2, label=c_name)
            ax.fill(angles, vals, color=color, alpha=0.18)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(clean_cats, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True)
        plt.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig_mpl.savefig(p.with_suffix(".png"))

        # --- Plotly Radar Chart ---
        fig_plotly = go.Figure()
        for g in range(K):
            vals = category_centroids[g].tolist()
            vals += vals[:1]
            c_name = cluster_names[g] if cluster_names and g < len(cluster_names) else f"Archetype {g+1}"

            fig_plotly.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=clean_cats + [clean_cats[0]],
                    fill="toself",
                    name=c_name,
                )
            )

        fig_plotly.update_layout(
            title=title,
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            template="plotly_white",
        )

        if save_path:
            fig_plotly.write_html(str(Path(save_path).with_suffix(".html")))

        return fig_mpl, fig_plotly
