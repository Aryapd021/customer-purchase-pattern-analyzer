"""Unit and Mathematical Verification Tests for Data Ingestion & Synthetic Data Generation."""

import io
import json
import numpy as np
import pandas as pd
import pytest

from src.ingestion.synthetic_generator import (
    SyntheticCustomerDataGenerator,
    SyntheticDataResult,
)
from src.ingestion.data_ingestion import (
    DataIngestionModule,
    IngestionResult,
    NormalizationType,
    ImputationStrategy,
)


class TestSyntheticCustomerDataGenerator:
    """Mathematical validation for the synthetic data generator."""

    def test_rank_controllability_exact(self):
        """Verify that when noise_level=0, the matrix rank is mathematically exactly r."""
        generator = SyntheticCustomerDataGenerator(random_state=42)
        test_cases = [
            (50, 10, 3),
            (30, 8, 2),
            (100, 15, 5),
            (25, 6, 1),
            (20, 10, 7),
        ]

        for m, n, r in test_cases:
            res = generator.generate(
                n_customers=m,
                n_categories=n,
                rank=r,
                noise_level=0.0,
            )

            assert res.matrix.shape == (m, n), f"Expected shape {(m, n)}, got {res.matrix.shape}"
            assert res.target_rank == r
            assert res.exact_rank == r, f"Expected rank {r}, but computed {res.exact_rank}"
            assert np.linalg.matrix_rank(res.matrix) == r

            # Check that singular values beyond index r-1 are zero (within numerical precision)
            s = res.singular_values
            assert len(s) == min(m, n)
            assert s[r - 1] > 1e-6, f"Rank {r} singular value should be significant"
            if r < len(s):
                assert s[r] < 1e-10, f"Singular value at {r} should be effectively 0, got {s[r]}"

    def test_non_negativity_and_sparsity(self):
        """Verify spend values are strictly non-negative and sparsity operates correctly."""
        generator = SyntheticCustomerDataGenerator(random_state=123)
        res = generator.generate(n_customers=40, n_categories=8, rank=3, noise_level=0.0)

        assert np.all(res.matrix >= 0.0), "Transaction spend values must be non-negative"
        assert len(res.customer_ids) == 40
        assert len(res.category_names) == 8

    def test_invalid_rank_raises_error(self):
        """Rank must satisfy 1 <= r < min(m, n)."""
        generator = SyntheticCustomerDataGenerator(random_state=42)

        # r >= min(m, n)
        with pytest.raises(ValueError, match="Controllable rank r must satisfy"):
            generator.generate(n_customers=10, n_categories=5, rank=5)

        with pytest.raises(ValueError, match="Controllable rank r must satisfy"):
            generator.generate(n_customers=10, n_categories=5, rank=6)

        # r < 1
        with pytest.raises(ValueError, match="Controllable rank r must satisfy"):
            generator.generate(n_customers=10, n_categories=5, rank=0)

    def test_dataframe_and_serialization(self, tmp_path):
        """Verify export to DataFrame, CSV, and JSON."""
        generator = SyntheticCustomerDataGenerator(random_state=7)
        res = generator.generate(n_customers=15, n_categories=5, rank=2)

        # DataFrame export
        df = res.to_dataframe()
        assert df.shape == (15, 5)
        assert df.index.name == "CustomerID"
        assert list(df.columns) == res.category_names

        # CSV export
        csv_file = tmp_path / "synthetic_customers.csv"
        res.to_csv(str(csv_file))
        assert csv_file.exists()

        # Reload CSV and verify
        reloaded_df = pd.read_csv(csv_file, index_col=0)
        np.testing.assert_allclose(reloaded_df.to_numpy(), res.matrix, rtol=1e-5)

        # JSON export
        json_file = tmp_path / "synthetic_customers.json"
        res.to_json(str(json_file), orient="split")
        assert json_file.exists()


class TestDataIngestionModule:
    """Functional and mathematical tests for DataIngestionModule."""

    def test_benchmark_hand_calculated_matrix(self):
        """Validates against the hand-calculated benchmark matrix from the Masterplan:
        A_test = [[4, 0, 3],
                  [0, 5, 0],
                  [8, 0, 6]]
        Row 1 norm = 5.0
        Row 2 norm = 5.0
        Row 3 norm = 10.0
        L2 normalized:
          Row 1: [0.8, 0, 0.6]
          Row 2: [0, 1.0, 0]
          Row 3: [0.8, 0, 0.6]  (parallel to Row 1)
        """
        a_test = np.array([
            [4.0, 0.0, 3.0],
            [0.0, 5.0, 0.0],
            [8.0, 0.0, 6.0],
        ], dtype=np.float64)

        ingestion = DataIngestionModule()

        # 1. Unnormalized processing
        res_raw = ingestion.process(a_test, normalization=NormalizationType.NONE)
        np.testing.assert_allclose(res_raw.matrix, a_test)
        np.testing.assert_allclose(res_raw.row_norms, [5.0, 5.0, 10.0])

        # 2. L2 Normalization
        res_l2 = ingestion.process(a_test, normalization=NormalizationType.L2)
        expected_l2 = np.array([
            [0.8, 0.0, 0.6],
            [0.0, 1.0, 0.0],
            [0.8, 0.0, 0.6],
        ])
        np.testing.assert_allclose(res_l2.matrix, expected_l2, atol=1e-12)

        # Each normalized row must have L2 norm == 1.0
        computed_row_norms = np.linalg.norm(res_l2.matrix, ord=2, axis=1)
        np.testing.assert_allclose(computed_row_norms, [1.0, 1.0, 1.0], atol=1e-12)

        # 3. Mean Centering
        # Column means: [4.0, 1.666667, 3.0]
        res_mc = ingestion.process(a_test, normalization=NormalizationType.MEAN_CENTER)
        expected_col_means = np.array([4.0, 5.0 / 3.0, 3.0])
        np.testing.assert_allclose(res_mc.feature_means, expected_col_means, atol=1e-12)

        # Mean of centered columns must be identically 0
        col_means_after = np.mean(res_mc.matrix, axis=0)
        np.testing.assert_allclose(col_means_after, [0.0, 0.0, 0.0], atol=1e-12)

    def test_missing_value_imputation(self):
        """Test missing value detection and imputation strategies."""
        raw_data = np.array([
            [10.0, np.nan, 30.0],
            [20.0, 20.0, np.nan],
            [np.nan, 40.0, 50.0],
            [40.0, 60.0, 70.0],
        ])

        # Zero imputation
        mod_zero = DataIngestionModule(imputation_strategy=ImputationStrategy.ZERO)
        res_zero = mod_zero.process(raw_data)
        assert res_zero.missing_count_imputed == 3
        assert not np.isnan(res_zero.matrix).any()
        assert res_zero.matrix[0, 1] == 0.0
        assert res_zero.matrix[1, 2] == 0.0
        assert res_zero.matrix[2, 0] == 0.0

        # Mean imputation
        # Col 0 values: 10, 20, 40 -> mean = 70/3 = 23.3333
        # Col 1 values: 20, 40, 60 -> mean = 120/3 = 40.0
        # Col 2 values: 30, 50, 70 -> mean = 150/3 = 50.0
        mod_mean = DataIngestionModule(imputation_strategy=ImputationStrategy.MEAN)
        res_mean = mod_mean.process(raw_data)
        assert res_mean.matrix[0, 1] == pytest.approx(40.0)
        assert res_mean.matrix[1, 2] == pytest.approx(50.0)
        assert res_mean.matrix[2, 0] == pytest.approx(70.0 / 3.0)

        # Median imputation
        # Col 0 median: 20.0
        # Col 1 median: 40.0
        # Col 2 median: 50.0
        mod_median = DataIngestionModule(imputation_strategy=ImputationStrategy.MEDIAN)
        res_median = mod_median.process(raw_data)
        assert res_median.matrix[2, 0] == 20.0
        assert res_median.matrix[0, 1] == 40.0
        assert res_median.matrix[1, 2] == 50.0

    def test_csv_parsing_and_customer_id_detection(self):
        """Verify CSV parsing with explicit and implicit customer IDs."""
        csv_text = (
            "CustomerID,Electronics,Groceries,Apparel\n"
            "CUST_001,150.0,50.0,20.0\n"
            "CUST_002,0.0,120.0,80.0\n"
            "CUST_003,200.0,0.0,15.0\n"
        )
        buffer = io.StringIO(csv_text)
        ingestion = DataIngestionModule()
        result = ingestion.process(buffer)

        assert result.customer_ids == ["CUST_001", "CUST_002", "CUST_003"]
        assert result.category_names == ["Electronics", "Groceries", "Apparel"]
        assert result.matrix.shape == (3, 3)
        assert result.matrix.dtype == np.float64
        assert result.matrix[0, 0] == 150.0

    def test_json_parsing(self):
        """Verify JSON parsing in records format."""
        json_data = [
            {"CustomerID": "U1", "Cat_A": 10.5, "Cat_B": 25.0},
            {"CustomerID": "U2", "Cat_A": 0.0, "Cat_B": 50.0},
        ]
        buffer = io.StringIO(json.dumps(json_data))
        ingestion = DataIngestionModule()
        result = ingestion.process(buffer)

        assert result.customer_ids == ["U1", "U2"]
        assert result.category_names == ["Cat_A", "Cat_B"]
        assert result.matrix.shape == (2, 2)
        assert result.matrix[0, 1] == 25.0

    def test_l2_normalize_zero_vector(self):
        """L2 normalization of an all-zero customer row should safely remain all zeros."""
        zero_row_matrix = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ])
        ingestion = DataIngestionModule()
        res = ingestion.process(zero_row_matrix, normalization=NormalizationType.L2)

        np.testing.assert_allclose(res.matrix[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(res.matrix[1], [0.6, 0.8, 0.0])
