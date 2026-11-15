"""Evaluation package for clustering metrics, performance profiling, and statistical testing."""

from .clustering_metrics import compute_clustering_metrics
from .performance_metrics import extract_performance_profile
from .convergence import extract_convergence_trajectory
from .statistical_tests import aggregate_experiment_runs, run_statistical_significance_test

__all__ = [
    "compute_clustering_metrics",
    "extract_performance_profile",
    "extract_convergence_trajectory",
    "aggregate_experiment_runs",
    "run_statistical_significance_test",
]
