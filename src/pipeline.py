"""Sequential Computational Pipeline Engine.

Implements the end-to-end sequential data pipeline where the mathematical output of
Stage k forms the direct input to Stage k+1:
Stage 1: Matrix Ingestion     --> Dense Matrix A in R^{m x n}
Stage 2: Rank & Basis         --> RREF, Rank r, Column/Null Space Bases
Stage 3: Inner Product        --> Gram Matrix G, Norms D, Cosine Similarity S
Stage 4: Spectral / PCA       --> Covariance C, Eigenvalues lambda, Eigenvectors V
Stage 5: Subspace Projection  --> Operator P = V_k * V_k^T, Coordinates Y = X * V_k
Stage 6: Profile Formation    --> Subspace Clustering, Centroid Re-projection, Archetypes
Stage 7: Verification         --> Theoretical Identities & Scipy Benchmarking
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

from src.ingestion.data_ingestion import (
    DataIngestionModule,
    IngestionResult,
    NormalizationType,
    ImputationStrategy,
)
from src.ingestion.synthetic_generator import (
    SyntheticCustomerDataGenerator,
    SyntheticDataResult,
)
from src.linear_algebra.engine import (
    LinearAlgebraEngine,
    RREFResult,
    SubspaceBases,
    SimilarityResult,
    SpectralResult,
    ProjectionResult,
)
from src.profiling.profiler import (
    AnalyticalProfilingModule,
    ProfilingResult,
)
from src.verification.engine import (
    VerificationEngine,
    BenchmarkReport,
    IdentityCheckResult,
    ScipyComparisonResult,
)
from src.visualization.engine import VisualizationEngine


@dataclass
class PipelineExecutionResult:
    """Encapsulates all stage-by-stage results of the linear algebra pipeline.

    Attributes:
        ingestion: Output of Stage 1 (Dense Matrix A, Customer IDs, Category Names).
        rref: Output of Stage 2 (RREF Matrix, Rank, Pivot & Free Columns).
        subspaces: Output of Stage 2 (Row Space, Column Space, Null Space Bases).
        similarity: Output of Stage 3 (Gram Matrix G, Norms D, Cosine Similarity S).
        spectral: Output of Stage 4 (Covariance C, Eigenvalues, Eigenvectors, Variance Ratios).
        projection: Output of Stage 5 (Projection Operator P, Subspace Coordinates Y, Reconstruction).
        profiling: Output of Stage 6 (Subspace Clusters, Centroids, Archetype Profiles).
        benchmark_report: Output of Stage 7 (Validation on A_test).
        identity_report: Output of Stage 7 (Checklist for Trace, Det, Orthogonality).
        scipy_report: Output of Stage 7 (Numerical stability comparison vs scipy.linalg).
    """
    ingestion: IngestionResult
    rref: RREFResult
    subspaces: SubspaceBases
    similarity: SimilarityResult
    spectral: SpectralResult
    projection: ProjectionResult
    profiling: ProfilingResult
    benchmark_report: BenchmarkReport
    identity_report: IdentityCheckResult
    scipy_report: ScipyComparisonResult


class SequentialMathematicalPipeline:
    """Executes the strictly sequential mathematical pipeline across all 6 core stages and verification."""

    def __init__(self, tolerance: float = 1e-10, random_state: int = 42) -> None:
        """Initializes pipeline components."""
        self.tolerance = tolerance
        self.random_state = random_state
        self.la_engine = LinearAlgebraEngine(tolerance=tolerance)
        self.profiler = AnalyticalProfilingModule(random_state=random_state)
        self.verifier = VerificationEngine(tolerance=1e-6)
        self.viz_engine = VisualizationEngine()

    def run(
        self,
        data_source: Union[str, Path, pd.DataFrame, np.ndarray, SyntheticDataResult],
        k_dimensions: Optional[int] = 3,
        n_clusters: int = 3,
        normalization: Union[NormalizationType, str] = NormalizationType.NONE,
        imputation_strategy: Union[ImputationStrategy, str] = ImputationStrategy.ZERO,
        customer_id_col: Optional[str] = None,
    ) -> PipelineExecutionResult:
        """Executes the complete sequential pipeline.

        Data strictly flows:
        Stage 1 (Ingestion) -> Stage 2 (Rank & Basis) -> Stage 3 (Inner Product) ->
        Stage 4 (Spectral / PCA) -> Stage 5 (Subspace Projection) -> Stage 6 (Profile Formation).
        """
        # =====================================================================
        # STAGE 1: Data Ingestion & Matrix Setup -> Matrix A in R^{m x n}
        # =====================================================================
        ingestion_engine = DataIngestionModule(
            customer_id_col=customer_id_col,
            imputation_strategy=imputation_strategy,
        )

        if isinstance(normalization, str):
            norm_enum = NormalizationType(normalization.lower())
        else:
            norm_enum = normalization

        if isinstance(imputation_strategy, str):
            imp_enum = ImputationStrategy(imputation_strategy.lower())
        else:
            imp_enum = imputation_strategy

        if isinstance(data_source, SyntheticDataResult):
            raw_matrix = data_source.matrix
            res_stage1 = IngestionResult(
                matrix=raw_matrix,
                customer_ids=data_source.customer_ids,
                category_names=data_source.category_names,
                normalization=norm_enum,
                imputation_strategy=imp_enum,
                missing_count_imputed=0,
                feature_means=np.mean(raw_matrix, axis=0),
                row_norms=np.linalg.norm(raw_matrix, ord=2, axis=1),
            )
        else:
            res_stage1 = ingestion_engine.process(
                data=data_source,
                normalization=norm_enum,
                imputation_strategy=imp_enum,
                customer_id_col=customer_id_col,
            )

        A = res_stage1.matrix

        # =====================================================================
        # STAGE 2: Rank & Basis Extraction -> RREF, Rank r, Subspaces
        # =====================================================================
        res_rref = self.la_engine.compute_rref(A)
        res_subspaces = self.la_engine.extract_subspace_bases(A)

        # =====================================================================
        # STAGE 3: Vector Inner Product & Cosine Similarity -> G, D, S
        # =====================================================================
        res_similarity = self.la_engine.compute_similarity_engine(A)

        # =====================================================================
        # STAGE 4: Covariance & Spectral Decomposition (PCA) -> C, lambda, V
        # =====================================================================
        res_spectral = self.la_engine.compute_spectral_decomposition(A, method="qr")

        # =====================================================================
        # STAGE 5: Subspace Orthogonal Projection -> P = V_k V_k^T, Y = X V_k
        # =====================================================================
        resolved_k = k_dimensions if k_dimensions is not None else min(3, res_rref.rank)
        res_projection = self.la_engine.compute_orthogonal_projection(res_spectral, k=resolved_k)

        # =====================================================================
        # STAGE 6: Profile Formation & Behavioral Interpretation
        # =====================================================================
        res_profiling = self.profiler.profile_customers(
            projected_coordinates=res_projection.projected_coordinates,
            top_eigenvectors=res_projection.top_eigenvectors,
            mean_vector=res_spectral.mean_vector,
            n_clusters=n_clusters,
            customer_ids=res_stage1.customer_ids,
            category_names=res_stage1.category_names,
        )

        # =====================================================================
        # STAGE 7: Mathematical Verification Engine
        # =====================================================================
        bench_report = self.verifier.verify_benchmark_properties()
        ident_report = self.verifier.verify_theoretical_identities(A)
        scipy_report = self.verifier.compare_against_scipy(A)

        return PipelineExecutionResult(
            ingestion=res_stage1,
            rref=res_rref,
            subspaces=res_subspaces,
            similarity=res_similarity,
            spectral=res_spectral,
            projection=res_projection,
            profiling=res_profiling,
            benchmark_report=bench_report,
            identity_report=ident_report,
            scipy_report=scipy_report,
        )
