"""Analytical Customer Profiling Package.

Provides customer clustering in k-dimensional projected subspace,
centroid re-projection back to category space, and behavioral archetype interpretation.
"""

from .profiler import (
    AnalyticalProfilingModule,
    CustomerProfileCluster,
    ProfilingResult,
)

__all__ = [
    "AnalyticalProfilingModule",
    "CustomerProfileCluster",
    "ProfilingResult",
]
