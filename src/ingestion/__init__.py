"""Data Ingestion and Matrix Formulation Module.

This package contains components for:
- Synthetic customer purchase matrix generation with controllable rank r < min(m, n).
- CSV and JSON parsing, data validation, and missing value imputation.
- Dense NumPy matrix conversion.
- Vector normalization (L2 normalization and Mean Centering).
"""

from .synthetic_generator import SyntheticCustomerDataGenerator, SyntheticDataResult
from .data_ingestion import DataIngestionModule, IngestionResult, NormalizationType, ImputationStrategy

__all__ = [
    "SyntheticCustomerDataGenerator",
    "SyntheticDataResult",
    "DataIngestionModule",
    "IngestionResult",
    "NormalizationType",
    "ImputationStrategy",
]
