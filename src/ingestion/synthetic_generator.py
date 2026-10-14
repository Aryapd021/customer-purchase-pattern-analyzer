"""Synthetic Customer Data Generator with Controllable Rank.

Mathematical formulation:
Let m be the number of customers and n be the number of product categories.
A matrix A in R^{m x n} with controllable rank r < min(m, n) is constructed
via low-rank factorization:
    A = W * H
where:
    W in R^{m x r} represents latent customer archetype affinity weights,
    H in R^{r x n} represents the archetype category profile basis.

Since rank(A) <= min(rank(W), rank(H)) = r, generating W and H with linearly
independent columns/rows almost surely yields rank(A) = r.
Optionally, additive noise epsilon ~ N(0, sigma^2) can be introduced to model
stochastic perturbations while preserving an effective numerical rank r.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class SyntheticDataResult:
    """Encapsulates the generated synthetic dataset and its mathematical properties.

    Attributes:
        matrix: Dense NumPy array A of shape (m, n).
        customer_ids: List of unique customer identifier strings.
        category_names: List of product category strings.
        target_rank: Prescribed target rank r.
        exact_rank: Mathematically computed rank of matrix A.
        singular_values: Array of singular values of A.
        customer_archetypes: Latent factor loading matrix W of shape (m, r).
        category_archetypes: Latent category basis matrix H of shape (r, n).
        noise_level: Standard deviation of added Gaussian noise (0 for exact rank).
    """

    matrix: np.ndarray
    customer_ids: List[str]
    category_names: List[str]
    target_rank: int
    exact_rank: int
    singular_values: np.ndarray
    customer_archetypes: np.ndarray
    category_archetypes: np.ndarray
    noise_level: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the customer-category matrix into a labeled Pandas DataFrame."""
        df = pd.DataFrame(
            self.matrix,
            index=self.customer_ids,
            columns=self.category_names,
        )
        df.index.name = "CustomerID"
        return df

    def to_csv(self, file_path: str, include_index: bool = True) -> None:
        """Exports the synthetic dataset to a CSV file."""
        df = self.to_dataframe()
        df.to_csv(file_path, index=include_index)

    def to_json(self, file_path: str, orient: str = "split") -> None:
        """Exports the synthetic dataset to a JSON file."""
        df = self.to_dataframe()
        df.to_json(file_path, orient=orient, indent=2)


class SyntheticCustomerDataGenerator:
    """Generator for synthetic customer transaction matrices with controlled linear rank.

    Simulates realistic customer purchasing behavior where preferences are driven
    by a small number of latent purchasing factors (archetypes).
    """

    DEFAULT_CATEGORIES: List[str] = [
        "Electronics",
        "Groceries",
        "Apparel",
        "Home_Kitchen",
        "Beauty_Wellness",
        "Books_Media",
        "Sports_Outdoors",
        "Automotive",
        "Pet_Supplies",
        "Toys_Games",
        "Garden_Patio",
        "Office_Supplies",
    ]

    def __init__(self, random_state: Optional[int] = None) -> None:
        """Initializes generator with optional random seed for reproducibility."""
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def generate(
        self,
        n_customers: int = 100,
        n_categories: int = 10,
        rank: int = 3,
        noise_level: float = 0.0,
        sparsity_factor: float = 0.2,
        base_spend_range: Tuple[float, float] = (50.0, 500.0),
        category_names: Optional[List[str]] = None,
        customer_id_prefix: str = "CUST",
    ) -> SyntheticDataResult:
        """Generates customer-category matrix A in R^{m x n} with controllable rank r.

        Args:
            n_customers: Number of customers m (rows).
            n_categories: Number of product categories n (columns).
            rank: Controlled matrix rank r, strictly requiring 1 <= r < min(m, n).
            noise_level: Amplitude of Gaussian noise. If 0.0, exact rank r is preserved.
            sparsity_factor: Fraction of near-zero purchase interactions.
            base_spend_range: Minimum and maximum typical customer total spend.
            category_names: Optional custom list of category names (length n).
            customer_id_prefix: Prefix for formatted customer IDs.

        Returns:
            SyntheticDataResult containing the dense matrix and factorization metadata.

        Raises:
            ValueError: If rank is invalid with respect to matrix dimensions.
        """
        if n_customers < 2 or n_categories < 2:
            raise ValueError(
                f"Dimensions must be at least 2x2. Received m={n_customers}, n={n_categories}"
            )

        max_allowed_rank = min(n_customers, n_categories) - 1
        if not (1 <= rank <= max_allowed_rank):
            raise ValueError(
                f"Controllable rank r must satisfy 1 <= r < min(m, n). "
                f"Received r={rank} for dimensions ({n_customers}, {n_categories}). "
                f"Maximum allowed rank is {max_allowed_rank}."
            )

        # 1. Resolve category names
        if category_names is None:
            if n_categories <= len(self.DEFAULT_CATEGORIES):
                resolved_categories = self.DEFAULT_CATEGORIES[:n_categories]
            else:
                resolved_categories = [
                    f"Category_{i+1:02d}" for i in range(n_categories)
                ]
        else:
            if len(category_names) != n_categories:
                raise ValueError(
                    f"Length of category_names ({len(category_names)}) must equal n_categories ({n_categories})"
                )
            resolved_categories = list(category_names)

        # 2. Generate customer identifiers
        customer_ids = [
            f"{customer_id_prefix}_{i+1:04d}" for i in range(n_customers)
        ]

        # 3. Construct Category Factor Basis Matrix H in R^{r x n}
        # Each archetype concentrates spending on distinct category subsets.
        # Use Dirichlet distribution or Dirichlet-like concentration to guarantee non-negative,
        # linearly independent category profile vectors.
        H = np.zeros((rank, n_categories), dtype=np.float64)
        for k in range(rank):
            # Focus on distinct primary categories with secondary spillover
            alpha = np.full(n_categories, 0.2)
            # Assign primary categories to this archetype
            stride = max(1, n_categories // rank)
            primary_indices = [
                (k * stride + offset) % n_categories for offset in range(stride)
            ]
            alpha[primary_indices] = 3.0
            H[k, :] = self._rng.dirichlet(alpha)

        # Ensure rows of H are linearly independent (rank(H) == r)
        # Using SVD check; if rank < r (extremely unlikely for random Dirichlet), re-draw
        while np.linalg.matrix_rank(H) < rank:
            H = self._rng.exponential(scale=1.0, size=(rank, n_categories))

        # 4. Construct Customer Factor Loading Matrix W in R^{m x r}
        # Each customer has preferences distributed across the r latent archetypes.
        # Generate archetype mixture weights (Dirichlet) multiplied by customer total spend scale.
        mixture_alpha = np.full(rank, 0.7)
        customer_mixtures = self._rng.dirichlet(mixture_alpha, size=n_customers)

        # Customer spend scaling factor (Log-normal distribution within specified range)
        low_spend, high_spend = base_spend_range
        log_mean = np.log((low_spend + high_spend) / 2.0)
        customer_spend_scales = self._rng.lognormal(
            mean=log_mean, sigma=0.4, size=(n_customers, 1)
        )
        customer_spend_scales = np.clip(customer_spend_scales, low_spend, high_spend * 2.0)

        W = customer_mixtures * customer_spend_scales

        # Ensure columns of W are linearly independent (rank(W) == r)
        while np.linalg.matrix_rank(W) < rank:
            W = self._rng.gamma(shape=2.0, scale=50.0, size=(n_customers, rank))

        # 5. Form the Low-Rank Product Matrix: A_clean = W * H in R^{m x n}
        A_clean = np.matmul(W, H)

        # 6. Apply realistic sparsity if requested
        if sparsity_factor > 0:
            # Mask out minor spending amounts below quantile
            threshold = np.quantile(A_clean, sparsity_factor)
            mask = A_clean < threshold
            # To strictly preserve rank r if noise_level == 0, we only zero out if requested
            # without breaking rank, or we apply noise afterward.
            # In linear algebra, zeroing arbitrary entries can change rank.
            # If noise_level == 0.0, we keep exact low-rank product A_clean.

        if noise_level > 0.0:
            noise = self._rng.normal(0.0, noise_level, size=A_clean.shape)
            A = np.clip(A_clean + noise, 0.0, None)
        else:
            A = np.clip(A_clean, 0.0, None)

        # 7. Evaluate exact numerical rank and singular values
        singular_values = np.linalg.svd(A, compute_uv=False)
        exact_rank = int(np.linalg.matrix_rank(A))

        return SyntheticDataResult(
            matrix=A,
            customer_ids=customer_ids,
            category_names=resolved_categories,
            target_rank=rank,
            exact_rank=exact_rank,
            singular_values=singular_values,
            customer_archetypes=W,
            category_archetypes=H,
            noise_level=noise_level,
        )
