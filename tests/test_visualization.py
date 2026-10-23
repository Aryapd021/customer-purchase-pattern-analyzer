"""Unit and Functional Tests for VisualizationEngine."""

import numpy as np
import pytest
from pathlib import Path

from src.visualization.engine import VisualizationEngine


class TestVisualizationEngine:
    """Verifies that 2D/3D projections, heatmaps, scree plots, and radar charts render and save properly."""

    @pytest.fixture
    def engine(self):
        return VisualizationEngine()

    @pytest.fixture
    def dummy_data(self):
        np.random.seed(42)
        m, n, k, K = 30, 6, 3, 3
        Y = np.random.randn(m, k)
        labels = np.random.randint(0, K, size=m)
        centroids = np.random.randn(K, k)
        cluster_names = ["Tech Enthusiasts", "Grocery Focused", "Fashion Shoppers"]
        cat_names = [f"Cat_{j+1}" for j in range(n)]
        cat_centroids = np.abs(np.random.randn(K, n) * 50.0 + 100.0)
        sim_matrix = np.eye(m)
        eigenvalues = np.array([40.0, 25.0, 15.0, 5.0, 2.0, 1.0])
        var_ratios = eigenvalues / np.sum(eigenvalues)
        cum_ratios = np.cumsum(var_ratios)

        return {
            "Y": Y,
            "labels": labels,
            "centroids": centroids,
            "cluster_names": cluster_names,
            "cat_names": cat_names,
            "cat_centroids": cat_centroids,
            "sim_matrix": sim_matrix,
            "eigenvalues": eigenvalues,
            "var_ratios": var_ratios,
            "cum_ratios": cum_ratios,
        }

    def test_plot_subspace_projection_2d(self, engine, dummy_data, tmp_path):
        save_file = tmp_path / "projection_2d"
        fig_mpl, fig_plotly = engine.plot_subspace_projection_2d(
            Y=dummy_data["Y"],
            labels=dummy_data["labels"],
            cluster_names=dummy_data["cluster_names"],
            centroids=dummy_data["centroids"][:, :2],
            save_path=save_file,
        )

        assert fig_mpl is not None
        assert fig_plotly is not None
        assert (tmp_path / "projection_2d.png").exists()
        assert (tmp_path / "projection_2d.html").exists()

    def test_plot_subspace_projection_3d(self, engine, dummy_data, tmp_path):
        save_file = tmp_path / "projection_3d"
        fig_mpl, fig_plotly = engine.plot_subspace_projection_3d(
            Y=dummy_data["Y"],
            labels=dummy_data["labels"],
            cluster_names=dummy_data["cluster_names"],
            centroids=dummy_data["centroids"],
            save_path=save_file,
        )

        assert fig_mpl is not None
        assert fig_plotly is not None
        assert (tmp_path / "projection_3d.png").exists()
        assert (tmp_path / "projection_3d.html").exists()

    def test_plot_similarity_heatmap(self, engine, dummy_data, tmp_path):
        save_file = tmp_path / "heatmap"
        m = dummy_data["sim_matrix"].shape[0]
        test_ids = [f"CUST_{i+1:03d}" for i in range(m)]

        fig_mpl, fig_plotly = engine.plot_similarity_heatmap(
            similarity_matrix=dummy_data["sim_matrix"],
            labels=dummy_data["labels"],
            customer_ids=test_ids,
            sort_by_cluster=False,
            save_path=save_file,
        )

        assert fig_mpl is not None
        assert fig_plotly is not None
        assert (tmp_path / "heatmap.png").exists()
        assert (tmp_path / "heatmap.html").exists()

        # Verify strict square aspect ratio
        assert fig_plotly.layout.yaxis.scaleanchor == "x"
        assert fig_plotly.layout.yaxis.scaleratio == 1
        assert fig_plotly.layout.width == fig_plotly.layout.height

        # Verify customer IDs are present
        assert list(fig_plotly.data[0].x) == test_ids
        assert list(fig_plotly.data[0].y) == test_ids

    def test_plot_variance_scree(self, engine, dummy_data, tmp_path):
        save_file = tmp_path / "scree"
        fig_mpl, fig_plotly = engine.plot_variance_scree(
            eigenvalues=dummy_data["eigenvalues"],
            variance_ratios=dummy_data["var_ratios"],
            cumulative_ratios=dummy_data["cum_ratios"],
            save_path=save_file,
        )

        assert fig_mpl is not None
        assert fig_plotly is not None
        assert (tmp_path / "scree.png").exists()
        assert (tmp_path / "scree.html").exists()

    def test_plot_profile_radar_chart(self, engine, dummy_data, tmp_path):
        save_file = tmp_path / "radar"
        fig_mpl, fig_plotly = engine.plot_profile_radar_chart(
            category_centroids=dummy_data["cat_centroids"],
            category_names=dummy_data["cat_names"],
            cluster_names=dummy_data["cluster_names"],
            save_path=save_file,
        )

        assert fig_mpl is not None
        assert fig_plotly is not None
        assert (tmp_path / "radar.png").exists()
        assert (tmp_path / "radar.html").exists()
