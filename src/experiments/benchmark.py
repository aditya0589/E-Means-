"""Core benchmarking engine for running single and paired clustering evaluations."""

from typing import Dict, Any, Optional, Type
import numpy as np
import pandas as pd

from ..algorithms.base import BaseClustering
from ..evaluation.clustering_metrics import compute_clustering_metrics
from ..evaluation.performance_metrics import extract_performance_profile
from ..evaluation.convergence import extract_convergence_trajectory


def run_single_benchmark(
    algo_cls: Type[BaseClustering],
    algo_name: str,
    X: np.ndarray,
    n_clusters: int,
    dataset_name: str,
    y_true: Optional[np.ndarray] = None,
    seed: int = 42,
    algo_kwargs: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], BaseClustering, pd.DataFrame]:
    """Execute a single algorithm fit, compute all metrics, and return result records."""
    kwargs = algo_kwargs.copy() if algo_kwargs else {}
    model = algo_cls(n_clusters=n_clusters, random_state=seed, **kwargs)

    # Fit algorithm
    model.fit(X)

    # Extract quality and performance metrics
    quality_metrics = compute_clustering_metrics(
        X=X,
        labels=model.labels_,
        centroids=model.centroids_,
        y_true=y_true,
        random_state=seed,
    )
    perf_metrics = extract_performance_profile(model)

    record: Dict[str, Any] = {
        "dataset": dataset_name,
        "algorithm": algo_name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_clusters": n_clusters,
        "random_seed": seed,
        **perf_metrics,
        **quality_metrics,
    }

    # Extract trajectory
    trajectory_df = extract_convergence_trajectory(model, algo_name, dataset_name)

    return record, model, trajectory_df
