"""Scalability experiment runner across dataset size, dimensions, and cluster count."""

from typing import List, Dict, Any, Type
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from ..algorithms.traditional_kmeans import TraditionalKMeans
from ..algorithms.kmeans_plus_plus import KMeansPlusPlus
from ..algorithms.elkan_kmeans import ElkanKMeans
from ..algorithms.emeans import EMeans
from .benchmark import run_single_benchmark
from ..utils.logging_config import get_logger

logger = get_logger("scalability_runner")

ALGO_MAP: Dict[str, Type] = {
    "Traditional (Lloyd)": TraditionalKMeans,
    "K-Means++": KMeansPlusPlus,
    "Elkan K-Means": ElkanKMeans,
    "E-Means (FAIR)": EMeans,
}


def run_sample_size_scalability(
    sample_sizes: List[int] = [500, 1000, 5000, 10000, 50000],
    n_features: int = 10,
    n_clusters: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Benchmark runtime and memory scaling across varying sample sizes."""
    records = []
    logger.info(f"Starting sample size scalability sweep: {sample_sizes}")

    for n in sample_sizes:
        X, _ = make_blobs(n_samples=n, n_features=n_features, centers=n_clusters, random_state=seed)
        for algo_name, algo_cls in ALGO_MAP.items():
            rec, _, _ = run_single_benchmark(
                algo_cls=algo_cls,
                algo_name=algo_name,
                X=X,
                n_clusters=n_clusters,
                dataset_name=f"scale_n_{n}",
                seed=seed,
            )
            records.append(rec)
            logger.info(f"  n={n} | {algo_name:20s} | Time: {rec['total_time_sec']:.4f}s | Inertia: {rec['inertia']:.1f}")

    return pd.DataFrame(records)


def run_dimensionality_scalability(
    dimensions: List[int] = [2, 10, 25, 50, 100],
    n_samples: int = 5000,
    n_clusters: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Benchmark runtime scaling across varying feature dimensions."""
    records = []
    logger.info(f"Starting feature dimensionality scalability sweep: {dimensions}")

    for d in dimensions:
        X, _ = make_blobs(n_samples=n_samples, n_features=d, centers=n_clusters, random_state=seed)
        for algo_name, algo_cls in ALGO_MAP.items():
            rec, _, _ = run_single_benchmark(
                algo_cls=algo_cls,
                algo_name=algo_name,
                X=X,
                n_clusters=n_clusters,
                dataset_name=f"scale_d_{d}",
                seed=seed,
            )
            records.append(rec)
            logger.info(f"  d={d} | {algo_name:20s} | Time: {rec['total_time_sec']:.4f}s")

    return pd.DataFrame(records)


def run_cluster_count_scalability(
    cluster_counts: List[int] = [2, 3, 5, 8, 10, 15, 20],
    n_samples: int = 5000,
    n_features: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Benchmark runtime scaling across varying number of clusters (k)."""
    records = []
    logger.info(f"Starting cluster count scalability sweep: {cluster_counts}")

    for k in cluster_counts:
        X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=k, random_state=seed)
        for algo_name, algo_cls in ALGO_MAP.items():
            rec, _, _ = run_single_benchmark(
                algo_cls=algo_cls,
                algo_name=algo_name,
                X=X,
                n_clusters=k,
                dataset_name=f"scale_k_{k}",
                seed=seed,
            )
            records.append(rec)
            logger.info(f"  k={k} | {algo_name:20s} | Time: {rec['total_time_sec']:.4f}s")

    return pd.DataFrame(records)
