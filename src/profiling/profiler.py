"""Analytical Profiling Module.

Interprets the outputs of the linear algebra pipeline (low-dimensional projected coordinates,
spectral eigenvectors, and population mean vector) to synthesize distinct customer behavioral archetypes.

Mathematical Workflow:
1. Customer proximity evaluation via Euclidean distance in the k-dimensional subspace:
       d(y_i, y_j) = ||y_i - y_j||_2 = sqrt(sum_{l=1}^k (Y_{il} - Y_{jl})^2)
2. Group clustering and centroid calculation in projected space:
       c_{proj, g} = (1 / |C_g|) * sum_{i in C_g} y_i in R^k
3. Reconstruction mapping of cluster centroids back to the original category space:
       c_{orig, g} = c_{proj, g} * V_k^T + mu in R^n
4. Behavioral interpretation: identifying category preference lift and dominant purchasing dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd


@dataclass
class CustomerProfileCluster:
    """Represents a discovered customer behavioral archetype cluster.

    Attributes:
        cluster_id: Unique integer index of the cluster (0-indexed).
        archetype_name: Semantic label characterizing purchasing habits (e.g., 'Tech Enthusiasts').
        customer_ids: List of customer identifiers assigned to this profile.
        customer_indices: List of row indices corresponding to the assigned customers.
        projected_centroid: Center of gravity c_{proj} in the k-dimensional subspace, shape (k,).
        category_centroid: Re-projected centroid c_{orig} in the n-dimensional category space, shape (n,).
        dominant_categories: Top category names with highest spending / preference lift.
        average_spend: Mean aggregate spend of customers in this cluster.
        customer_count: Total number of customers belonging to this cluster |C_g|.
        percentage: Proportion of total customer base (|C_g| / m) * 100.
        cohesion: Average Euclidean distance from customers to their projected centroid.
    """
    cluster_id: int
    archetype_name: str
    customer_ids: List[str]
    customer_indices: List[int]
    projected_centroid: np.ndarray
    category_centroid: np.ndarray
    dominant_categories: List[str]
    average_spend: float
    customer_count: int
    percentage: float
    cohesion: float


@dataclass
class ProfilingResult:
    """Encapsulates the end-to-end customer clustering and profile synthesis result.

    Attributes:
        clusters: List of CustomerProfileCluster objects.
        labels: 1D array of shape (m,) containing cluster assignments for each customer.
        projected_centroids: 2D array of shape (K, k) containing all projected cluster centers.
        category_centroids: 2D array of shape (K, n) containing all reconstructed centers.
        k_dimensions: Dimensionality k of the projected subspace.
        n_clusters: Total number of formed clusters K.
        total_inertia: Sum of squared Euclidean distances to assigned centroids.
        customer_id_to_profile: Mapping from customer ID string to archetype name.
        category_names: List of product category column names.
        customer_ids: List of all customer IDs.
    """
    clusters: List[CustomerProfileCluster]
    labels: np.ndarray
    projected_centroids: np.ndarray
    category_centroids: np.ndarray
    k_dimensions: int
    n_clusters: int
    total_inertia: float
    customer_id_to_profile: Dict[str, str]
    category_names: List[str]
    customer_ids: List[str]

    def to_dataframe(self) -> pd.DataFrame:
        """Constructs a summary DataFrame displaying archetype profiles and category centroid spends."""
        records = []
        for c in self.clusters:
            row: Dict[str, Any] = {
                "Cluster": f"Group {c.cluster_id + 1}",
                "Archetype": c.archetype_name,
                "Customers": c.customer_count,
                "Share (%)": f"{c.percentage:.1f}%",
                "Avg Spend": f"${c.average_spend:.2f}",
                "Top Categories": ", ".join(c.dominant_categories[:3]),
            }
            for j, cat in enumerate(self.category_names):
                row[cat] = float(c.category_centroid[j])
            records.append(row)
        return pd.DataFrame(records)


class AnalyticalProfilingModule:
    """Customer profiling engine utilizing subspace Euclidean distances and centroid re-projection."""

    def __init__(self, random_state: Optional[int] = 42) -> None:
        """Initializes the profiling module.

        Args:
            random_state: Seed for reproducible centroid initialization.
        """
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    @staticmethod
    def compute_euclidean_distance_matrix(Y: np.ndarray) -> np.ndarray:
        """Computes the pairwise Euclidean distance matrix D in R^{m x m} in the projected space:

            d(y_i, y_j) = ||y_i - y_j||_2 = sqrt(sum_{l=1}^k (Y_{il} - Y_{jl})^2)

        Args:
            Y: Projected customer coordinate matrix in R^{m x k}.

        Returns:
            Symmetric distance matrix D in R^{m x m}.
        """
        m, _ = Y.shape
        D = np.zeros((m, m), dtype=np.float64)

        for i in range(m):
            diff = Y[i + 1 :] - Y[i]
            dists = np.sqrt(np.sum(diff * diff, axis=1))
            D[i, i + 1 :] = dists
            D[i + 1 :, i] = dists

        return D

    def _kmeans_pp_init(self, Y: np.ndarray, K: int) -> np.ndarray:
        """Initializes K centroids using the k-means++ probabilistic spread algorithm."""
        m, k = Y.shape
        centroids = np.zeros((K, k), dtype=np.float64)

        # 1. Choose first center uniformly at random
        first_idx = self._rng.integers(0, m)
        centroids[0] = Y[first_idx]

        # 2. Iteratively choose remaining centers with probability proportional to D(x)^2
        closest_dist_sq = np.sum((Y - centroids[0]) ** 2, axis=1)

        for c_idx in range(1, K):
            total_dist_sq = float(np.sum(closest_dist_sq))
            if total_dist_sq < 1e-12:
                # If all points are identical, pick random
                next_idx = self._rng.integers(0, m)
            else:
                probs = closest_dist_sq / total_dist_sq
                next_idx = self._rng.choice(m, p=probs)

            centroids[c_idx] = Y[next_idx]
            new_dist_sq = np.sum((Y - centroids[c_idx]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

        return centroids

    def _cluster_kmeans(
        self,
        Y: np.ndarray,
        K: int,
        max_iter: int = 100,
        n_init: int = 10,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Custom K-Means clustering algorithm implemented from first principles.

        Optimizes sum of squared Euclidean distances in R^k:
            Inertia = sum_{i=1}^m ||y_i - c_{proj, label(i)}||_2^2

        Returns:
            Tuple of (labels, projected_centroids, total_inertia).
        """
        m, k = Y.shape
        best_inertia = float("inf")
        best_labels = np.zeros(m, dtype=int)
        best_centroids = np.zeros((K, k), dtype=np.float64)

        for _ in range(n_init):
            centroids = self._kmeans_pp_init(Y, K)
            labels = np.zeros(m, dtype=int)

            for _ in range(max_iter):
                # 1. Assignment Step: assign each point to nearest centroid
                # dist_matrix shape: (m, K)
                diff = Y[:, np.newaxis, :] - centroids[np.newaxis, :, :]
                dists_sq = np.sum(diff * diff, axis=2)
                new_labels = np.argmin(dists_sq, axis=1)

                # 2. Update Step: recompute group centroids
                new_centroids = np.zeros_like(centroids)
                for g in range(K):
                    cluster_members = Y[new_labels == g]
                    if len(cluster_members) > 0:
                        new_centroids[g] = np.mean(cluster_members, axis=0)
                    else:
                        # Re-seed empty cluster with random point
                        new_centroids[g] = Y[self._rng.integers(0, m)]

                # Check convergence
                shift = float(np.max(np.linalg.norm(new_centroids - centroids, axis=1)))
                centroids = new_centroids
                labels = new_labels
                if shift < tol:
                    break

            # Compute total inertia for this restart
            diff = Y - centroids[labels]
            inertia = float(np.sum(diff * diff))

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centroids = centroids.copy()

        return best_labels, best_centroids, best_inertia

    @staticmethod
    def reproject_centroids_to_category_space(
        projected_centroids: np.ndarray,
        V_k: np.ndarray,
        mean_vector: np.ndarray,
    ) -> np.ndarray:
        """Transforms cluster centroids from k-dimensional projected space back to original category space:

            c_{orig, g} = c_{proj, g} * V_k^T + mu in R^n

        Args:
            projected_centroids: Array of shape (K, k) where row g is c_{proj, g}.
            V_k: Matrix of top k eigenvectors in R^{n x k}.
            mean_vector: Population column mean vector mu in R^n.

        Returns:
            Reconstructed centroids array of shape (K, n).
        """
        # (K, k) @ (k, n) -> (K, n)
        reconstructed = np.matmul(projected_centroids, V_k.T) + mean_vector
        # Customer spend cannot be negative in physical category space
        return np.maximum(0.0, reconstructed)

    @staticmethod
    def _synthesize_archetype_label(
        category_centroid: np.ndarray,
        mean_vector: np.ndarray,
        category_names: List[str],
    ) -> Tuple[str, List[str]]:
        """Synthesizes human-interpretable archetype names based on category spending lifts."""
        # Calculate percentage lift over dataset average: (c_j - mu_j) / max(mu_j, 1.0)
        safe_means = np.maximum(1e-6, mean_vector)
        lifts = (category_centroid - mean_vector) / safe_means

        # Sort categories by lift descending
        sorted_indices = np.argsort(lifts)[::-1]
        sorted_cats = [category_names[i] for i in sorted_indices]

        top_cat = sorted_cats[0]
        second_cat = sorted_cats[1] if len(sorted_cats) > 1 else ""

        # Domain category dictionary mapping
        archetype_domain_map: Dict[str, str] = {
            "electronics": "Tech Enthusiasts",
            "groceries": "Grocery & Essentials Focused",
            "apparel": "Fashion & Lifestyle Shoppers",
            "home_kitchen": "Home & Living Enthusiasts",
            "beauty_wellness": "Health & Beauty Aficionados",
            "books_media": "Culture & Media Lovers",
            "sports_outdoors": "Active & Fitness Enthusiasts",
            "automotive": "Automotive Specialists",
            "pet_supplies": "Pet Care Devotees",
            "garden_patio": "Outdoor & Gardeners",
        }

        matched_label = None
        for key, name in archetype_domain_map.items():
            if key in top_cat.lower():
                matched_label = name
                break

        if matched_label is None:
            # Descriptive synthesis based on top categories
            clean_top = top_cat.replace("_", " ").title()
            clean_second = second_cat.replace("_", " ").title() if second_cat else ""
            if clean_second:
                matched_label = f"{clean_top} & {clean_second} Segment"
            else:
                matched_label = f"{clean_top} Dominant"

        return matched_label, sorted_cats

    def profile_customers(
        self,
        projected_coordinates: np.ndarray,
        top_eigenvectors: np.ndarray,
        mean_vector: np.ndarray,
        n_clusters: int = 3,
        customer_ids: Optional[List[str]] = None,
        category_names: Optional[List[str]] = None,
    ) -> ProfilingResult:
        """Executes full profile synthesis pipeline:

        1. Groups customers in R^k using Euclidean distance.
        2. Computes group centroids c_{proj, g} in projected space.
        3. Transforms centroids back to category space: c_{orig, g} = c_{proj, g} * V_k^T + mu.
        4. Synthesizes archetype behavioral labels and dominant purchasing categories.

        Args:
            projected_coordinates: Y = X * V_k in R^{m x k}.
            top_eigenvectors: V_k in R^{n x k}.
            mean_vector: Column mean vector mu in R^n.
            n_clusters: Number of customer archetypes K to extract.
            customer_ids: Optional customer ID list of length m.
            category_names: Optional category name list of length n.

        Returns:
            ProfilingResult containing synthesized clusters, centroids, and interpretations.
        """
        Y = projected_coordinates.astype(np.float64)
        m, k = Y.shape
        n = len(mean_vector)

        # Resolve identifiers
        resolved_cust_ids = (
            customer_ids if customer_ids is not None else [f"CUST_{i+1:04d}" for i in range(m)]
        )
        resolved_cat_names = (
            category_names if category_names is not None else [f"Category_{j+1:02d}" for j in range(n)]
        )

        # 1. Cluster customers in k-dimensional subspace using custom K-Means
        labels, proj_centroids, total_inertia = self._cluster_kmeans(
            Y=Y,
            K=n_clusters,
            max_iter=150,
            n_init=10,
        )

        # 2. Transform centroids back to category space: c_{orig} = c_{proj} * V_k^T + mu
        cat_centroids = self.reproject_centroids_to_category_space(
            projected_centroids=proj_centroids,
            V_k=top_eigenvectors,
            mean_vector=mean_vector,
        )

        # 3. Build CustomerProfileCluster representations
        clusters: List[CustomerProfileCluster] = []
        cust_profile_map: Dict[str, str] = {}

        for g in range(n_clusters):
            member_indices = [i for i in range(m) if labels[i] == g]
            member_ids = [resolved_cust_ids[i] for i in member_indices]
            count = len(member_indices)
            pct = (count / float(m)) * 100.0 if m > 0 else 0.0

            # Reconstructed centroid for this cluster
            c_orig = cat_centroids[g]
            c_proj = proj_centroids[g]

            # Synthesize archetype name & top categories
            label_name, sorted_cats = self._synthesize_archetype_label(
                category_centroid=c_orig,
                mean_vector=mean_vector,
                category_names=resolved_cat_names,
            )

            # Cluster cohesion: average distance to centroid
            if count > 0:
                diffs = Y[member_indices] - c_proj
                cohesion = float(np.mean(np.sqrt(np.sum(diffs * diffs, axis=1))))
            else:
                cohesion = 0.0

            # Average customer spend in this archetype
            avg_spend = float(np.sum(c_orig))

            cluster_obj = CustomerProfileCluster(
                cluster_id=g,
                archetype_name=label_name,
                customer_ids=member_ids,
                customer_indices=member_indices,
                projected_centroid=c_proj,
                category_centroid=c_orig,
                dominant_categories=sorted_cats[:3],
                average_spend=avg_spend,
                customer_count=count,
                percentage=pct,
                cohesion=cohesion,
            )
            clusters.append(cluster_obj)

            for cid in member_ids:
                cust_profile_map[cid] = label_name

        return ProfilingResult(
            clusters=clusters,
            labels=labels,
            projected_centroids=proj_centroids,
            category_centroids=cat_centroids,
            k_dimensions=k,
            n_clusters=n_clusters,
            total_inertia=total_inertia,
            customer_id_to_profile=cust_profile_map,
            category_names=resolved_cat_names,
            customer_ids=resolved_cust_ids,
        )
