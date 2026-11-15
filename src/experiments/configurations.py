"""Experiment configuration presets and hyperparameter specifications."""

from typing import Dict, Any, List

# Algorithm registry
ALGORITHM_CLASSES = {
    "Lloyd": "TraditionalKMeans",
    "KMeansPlusPlus": "KMeansPlusPlus",
    "Elkan": "ElkanKMeans",
    "EMeans": "EMeans",
}

# Standard random seeds for repeated trials
BENCHMARK_SEEDS: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# Dataset scale sizes for scalability testing
SAMPLE_SIZES: List[int] = [500, 1000, 5000, 10000, 50000]

# Feature dimension sweep
DIMENSION_SIZES: List[int] = [2, 10, 25, 50, 100]

# Cluster counts sweep
CLUSTER_COUNTS: List[int] = [2, 3, 5, 8, 10, 15, 20]

# Cluster overlap std dev sweep
NOISE_STD_LEVELS: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
