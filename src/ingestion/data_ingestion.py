"""Data Ingestion and Matrix Formulation Module.

Handles raw customer transaction ingestion from CSV/JSON formats,
missing value imputation, dense NumPy matrix formulation,
and mathematical vector normalization:
- L2 Normalization: a_i / ||a_i||_2 for each customer vector
- Mean Centering: X = A - 1 * mu^T where mu = (1/m) * sum(a_i)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import io
import json
import numpy as np
import pandas as pd


class NormalizationType(str, Enum):
    """Supported mathematical normalization strategies."""
    NONE = "none"
    L2 = "l2"
    MEAN_CENTER = "mean_center"
    STANDARDIZATION = "standardization"  # z-score: (x - mu) / sigma


class ImputationStrategy(str, Enum):
    """Supported missing value imputation strategies."""
    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"


@dataclass
class IngestionResult:
    """Container for the formulated matrix and its associated metadata.

    Attributes:
        matrix: Dense NumPy array A in R^{m x n} representing m customers and n categories.
        customer_ids: List of m customer identifier strings.
        category_names: List of n category names.
        normalization: Normalization technique applied.
        imputation_strategy: Imputation strategy used if missing values were present.
        missing_count_imputed: Total count of missing elements imputed.
        feature_means: Column-wise mean vector mu in R^n.
        row_norms: Original row Euclidean norms ||a_i||_2 before normalization.
    """
    matrix: np.ndarray
    customer_ids: List[str]
    category_names: List[str]
    normalization: NormalizationType
    imputation_strategy: Optional[ImputationStrategy]
    missing_count_imputed: int
    feature_means: np.ndarray
    row_norms: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns the dimensions (m, n) of the matrix."""
        return self.matrix.shape

    @property
    def n_customers(self) -> int:
        """Returns the number of customer rows m."""
        return self.matrix.shape[0]

    @property
    def n_categories(self) -> int:
        """Returns the number of category columns n."""
        return self.matrix.shape[1]

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the processed matrix back to a labeled DataFrame."""
        df = pd.DataFrame(
            self.matrix,
            index=self.customer_ids,
            columns=self.category_names,
        )
        df.index.name = "CustomerID"
        return df


class DataIngestionModule:
    """Engine for ingesting customer transaction datasets, formulating dense matrices,

    and executing vector normalization operations.
    """

    def __init__(
        self,
        customer_id_col: Optional[str] = None,
        imputation_strategy: Union[ImputationStrategy, str] = ImputationStrategy.ZERO,
    ) -> None:
        """Initializes the DataIngestionModule.

        Args:
            customer_id_col: Name of column containing customer IDs. If None, automatically detected.
            imputation_strategy: Strategy to handle missing / NaN values.
        """
        self.customer_id_col = customer_id_col
        if isinstance(imputation_strategy, str):
            imputation_strategy = ImputationStrategy(imputation_strategy.lower())
        self.imputation_strategy = imputation_strategy

    def parse_csv(
        self,
        source: Union[str, Path, io.StringIO, io.BytesIO],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Parses customer purchase records from a CSV file or buffer.

        Args:
            source: File path or stream.
            **kwargs: Extra parameters passed to pd.read_csv.

        Returns:
            Raw Pandas DataFrame.
        """
        return pd.read_csv(source, **kwargs)

    def parse_json(
        self,
        source: Union[str, Path, io.StringIO, io.BytesIO],
        orient: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Parses customer purchase records from a JSON file or buffer.

        Args:
            source: File path or stream.
            orient: JSON format orientation ('records', 'split', 'index', etc.).
            **kwargs: Extra parameters passed to pd.read_json.

        Returns:
            Raw Pandas DataFrame.
        """
        if orient is not None:
            return pd.read_json(source, orient=orient, **kwargs)
        return pd.read_json(source, **kwargs)

    def extract_matrix(
        self,
        df: pd.DataFrame,
        customer_id_col: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[str], List[str], np.ndarray]:
        """Extracts customer IDs, category names, and a dense float64 matrix from a DataFrame.

        Args:
            df: Source DataFrame.
            customer_id_col: Optional column name for customer ID.

        Returns:
            Tuple of:
                - Dense NumPy array A in R^{m x n} (unimputed, may contain NaNs)
                - List of customer IDs (length m)
                - List of category names (length n)
                - Boolean mask of missing/NaN values
        """
        df_copy = df.copy()

        # 1. Determine customer ID column
        target_id_col = customer_id_col or self.customer_id_col
        if target_id_col is not None and target_id_col in df_copy.columns:
            customer_ids = df_copy[target_id_col].astype(str).tolist()
            feature_df = df_copy.drop(columns=[target_id_col])
        elif df_copy.index.name is not None and df_copy.index.name.lower() in [
            "customerid", "customer_id", "id", "cust_id"
        ]:
            customer_ids = [str(x) for x in df_copy.index]
            feature_df = df_copy
        else:
            # Check for standard customer ID candidate column names
            candidates = ["customerid", "customer_id", "cust_id", "id", "customer"]
            found_col = None
            for col in df_copy.columns:
                if str(col).lower() in candidates:
                    found_col = col
                    break

            if found_col is not None:
                customer_ids = df_copy[found_col].astype(str).tolist()
                feature_df = df_copy.drop(columns=[found_col])
            else:
                customer_ids = [f"CUST_{i+1:04d}" for i in range(len(df_copy))]
                feature_df = df_copy

        # 2. Select numeric category columns only
        numeric_df = feature_df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            # Try to coerce non-numeric columns if possible
            coerced_df = feature_df.apply(pd.to_numeric, errors="coerce")
            numeric_cols = [c for c in coerced_df.columns if coerced_df[c].notna().any()]
            if not numeric_cols:
                raise ValueError("DataFrame contains no valid numeric category columns for matrix formation.")
            numeric_df = coerced_df[numeric_cols]

        category_names = [str(c) for c in numeric_df.columns]
        raw_matrix = numeric_df.to_numpy(dtype=np.float64, copy=True)
        nan_mask = np.isnan(raw_matrix)

        return raw_matrix, customer_ids, category_names, nan_mask

    def impute_missing(
        self,
        matrix: np.ndarray,
        strategy: Optional[Union[ImputationStrategy, str]] = None,
    ) -> Tuple[np.ndarray, int]:
        """Imputes missing (NaN) values in the matrix according to the chosen strategy.

        Args:
            matrix: Input array A in R^{m x n} with potential NaNs.
            strategy: Imputation method ('zero', 'mean', 'median', 'mode').

        Returns:
            Tuple of (imputed_matrix, total_imputed_count).
        """
        strat = strategy or self.imputation_strategy
        if isinstance(strat, str):
            strat = ImputationStrategy(strat.lower())

        imputed = matrix.copy()
        nan_mask = np.isnan(imputed)
        missing_count = int(np.sum(nan_mask))

        if missing_count == 0:
            return imputed, 0

        m, n = imputed.shape

        if strat == ImputationStrategy.ZERO:
            imputed[nan_mask] = 0.0

        elif strat == ImputationStrategy.MEAN:
            for j in range(n):
                col = imputed[:, j]
                col_nans = np.isnan(col)
                if np.all(col_nans):
                    col_mean = 0.0
                else:
                    col_mean = np.nanmean(col)
                imputed[col_nans, j] = col_mean

        elif strat == ImputationStrategy.MEDIAN:
            for j in range(n):
                col = imputed[:, j]
                col_nans = np.isnan(col)
                if np.all(col_nans):
                    col_median = 0.0
                else:
                    col_median = np.nanmedian(col)
                imputed[col_nans, j] = col_median

        elif strat == ImputationStrategy.MODE:
            for j in range(n):
                col = imputed[:, j]
                valid_vals = col[~np.isnan(col)]
                if len(valid_vals) == 0:
                    val_mode = 0.0
                else:
                    vals, counts = np.unique(valid_vals, return_counts=True)
                    val_mode = vals[np.argmax(counts)]
                imputed[np.isnan(col), j] = val_mode

        return imputed, missing_count

    @staticmethod
    def l2_normalize(
        matrix: np.ndarray,
        axis: int = 1,
        eps: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs L2 vector normalization.

        For customer rows (axis=1):
            a_i_norm = a_i / ||a_i||_2  where  ||a_i||_2 = sqrt(sum_{j=1}^n A_{ij}^2)
        Zero-norm vectors are preserved as zero vectors without division-by-zero.

        Args:
            matrix: Input matrix A in R^{m x n}.
            axis: Axis along which to compute norms (1 = row-wise customer vectors, 0 = column-wise).
            eps: Epsilon threshold for numerical stability.

        Returns:
            Tuple of (normalized_matrix, original_norms).
        """
        norms = np.linalg.norm(matrix, ord=2, axis=axis, keepdims=True)
        # Squeeze copy for metadata tracking
        recorded_norms = np.squeeze(norms.copy())
        if recorded_norms.ndim == 0:
            recorded_norms = np.array([recorded_norms.item()])

        # Safe division: if norm < eps, divide by 1.0 (vector remains zero)
        safe_norms = np.where(norms < eps, 1.0, norms)
        normalized = matrix / safe_norms

        # Explicitly zero out any entries where the original norm was virtually zero
        zero_mask = np.squeeze(norms < eps)
        if axis == 1:
            normalized[zero_mask, :] = 0.0
        else:
            normalized[:, zero_mask] = 0.0

        return normalized, recorded_norms

    @staticmethod
    def mean_center(
        matrix: np.ndarray,
        axis: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs column-wise (or row-wise) Mean Centering.

        For column-wise centering (axis=0, standard for PCA covariance calculation):
            mu_j = (1/m) * sum_{i=1}^m A_{ij}
            X = A - 1 * mu^T   where X_{ij} = A_{ij} - mu_j

        Args:
            matrix: Input matrix A in R^{m x n}.
            axis: Axis along which mean is computed (0 = column mean across customers, 1 = row mean).

        Returns:
            Tuple of (mean_centered_matrix, mean_vector).
        """
        means = np.mean(matrix, axis=axis, keepdims=True)
        centered = matrix - means
        recorded_means = np.squeeze(means.copy())
        if recorded_means.ndim == 0:
            recorded_means = np.array([recorded_means.item()])
        return centered, recorded_means

    def process(
        self,
        data: Union[str, Path, pd.DataFrame, np.ndarray, io.StringIO, io.BytesIO],
        normalization: Union[NormalizationType, str] = NormalizationType.NONE,
        imputation_strategy: Optional[Union[ImputationStrategy, str]] = None,
        customer_id_col: Optional[str] = None,
        json_orient: Optional[str] = None,
    ) -> IngestionResult:
        """End-to-end ingestion pipeline: parsing -> dense matrix -> imputation -> normalization.

        Args:
            data: CSV path, JSON path, DataFrame, or dense NumPy array.
            normalization: Desired vector normalization ('none', 'l2', 'mean_center').
            imputation_strategy: Strategy for handling missing values.
            customer_id_col: Optional column name for customer IDs.
            json_orient: Optional JSON structure orientation.

        Returns:
            IngestionResult containing the dense matrix and mathematical metadata.
        """
        if isinstance(normalization, str):
            normalization = NormalizationType(normalization.lower())

        # 1. Ingest input into DataFrame or NumPy array
        if isinstance(data, (str, Path)):
            path_obj = Path(data)
            suffix = path_obj.suffix.lower()
            if suffix in [".csv", ".txt"]:
                df = self.parse_csv(path_obj)
            elif suffix in [".json"]:
                df = self.parse_json(path_obj, orient=json_orient)
            else:
                # Attempt CSV first, fallback to JSON
                try:
                    df = self.parse_csv(path_obj)
                except Exception:
                    df = self.parse_json(path_obj, orient=json_orient)

            raw_matrix, cust_ids, cat_names, _ = self.extract_matrix(df, customer_id_col)

        elif isinstance(data, pd.DataFrame):
            raw_matrix, cust_ids, cat_names, _ = self.extract_matrix(data, customer_id_col)

        elif isinstance(data, np.ndarray):
            raw_matrix = data.astype(np.float64, copy=True)
            m, n = raw_matrix.shape
            cust_ids = [f"CUST_{i+1:04d}" for i in range(m)]
            cat_names = [f"Category_{j+1:02d}" for j in range(n)]

        elif isinstance(data, (io.StringIO, io.BytesIO)):
            # Determine if CSV or JSON
            content = data.getvalue() if isinstance(data, io.StringIO) else data.getvalue().decode("utf-8")
            data.seek(0)
            trimmed = content.strip()
            if trimmed.startswith("{") or trimmed.startswith("["):
                df = self.parse_json(data, orient=json_orient)
            else:
                df = self.parse_csv(data)
            raw_matrix, cust_ids, cat_names, _ = self.extract_matrix(df, customer_id_col)
        else:
            raise TypeError(f"Unsupported data source type: {type(data)}")

        # Ensure matrix is 2D
        if raw_matrix.ndim != 2:
            raise ValueError(f"Expected 2-dimensional matrix, got shape {raw_matrix.shape}")

        # 2. Impute Missing Values
        strat = imputation_strategy or self.imputation_strategy
        dense_matrix, imputed_count = self.impute_missing(raw_matrix, strategy=strat)

        # 3. Compute baseline metrics (column means and row norms before normalization)
        feature_means = np.mean(dense_matrix, axis=0)
        row_norms = np.linalg.norm(dense_matrix, ord=2, axis=1)

        # 4. Apply Mathematical Normalization
        final_matrix = dense_matrix
        if normalization == NormalizationType.L2:
            final_matrix, _ = self.l2_normalize(dense_matrix, axis=1)
        elif normalization == NormalizationType.MEAN_CENTER:
            final_matrix, _ = self.mean_center(dense_matrix, axis=0)
        elif normalization == NormalizationType.STANDARDIZATION:
            centered, _ = self.mean_center(dense_matrix, axis=0)
            stds = np.std(dense_matrix, axis=0, keepdims=True)
            safe_stds = np.where(stds < 1e-12, 1.0, stds)
            final_matrix = centered / safe_stds

        return IngestionResult(
            matrix=final_matrix,
            customer_ids=cust_ids,
            category_names=cat_names,
            normalization=normalization,
            imputation_strategy=strat if isinstance(strat, ImputationStrategy) else ImputationStrategy(strat),
            missing_count_imputed=imputed_count,
            feature_means=feature_means,
            row_norms=row_norms,
        )
