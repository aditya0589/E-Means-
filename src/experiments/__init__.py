"""Experiments package for running benchmarks, scalability sweeps, and statistical evaluations."""

from .benchmark import run_single_benchmark
from .experiment_runner import ExperimentRunner
from .scalability import (
    run_sample_size_scalability,
    run_dimensionality_scalability,
    run_cluster_count_scalability,
)
from .configurations import (
    ALGORITHM_CLASSES,
    BENCHMARK_SEEDS,
    SAMPLE_SIZES,
    DIMENSION_SIZES,
    CLUSTER_COUNTS,
    NOISE_STD_LEVELS,
)

__all__ = [
    "run_single_benchmark",
    "ExperimentRunner",
    "run_sample_size_scalability",
    "run_dimensionality_scalability",
    "run_cluster_count_scalability",
    "ALGORITHM_CLASSES",
    "BENCHMARK_SEEDS",
    "SAMPLE_SIZES",
    "DIMENSION_SIZES",
    "CLUSTER_COUNTS",
    "NOISE_STD_LEVELS",
]
