"""Real-world dataset loader module for clustering evaluation."""

from typing import Tuple, Dict, Any
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits


def load_dataset_by_name(name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load real-world benchmark datasets by name.

    Supported names: 'iris', 'wine', 'breast_cancer', 'digits'
    """
    clean_name = name.lower().strip()
    if clean_name == "iris":
        bunch = load_iris()
        meta = {
            "name": "iris",
            "n_samples": bunch.data.shape[0],
            "n_features": bunch.data.shape[1],
            "n_clusters": len(np.unique(bunch.target)),
            "feature_names": bunch.feature_names,
            "target_names": bunch.target_names.tolist(),
            "type": "real_world",
        }
        return bunch.data.astype(np.float64), bunch.target.astype(int), meta

    elif clean_name == "wine":
        bunch = load_wine()
        meta = {
            "name": "wine",
            "n_samples": bunch.data.shape[0],
            "n_features": bunch.data.shape[1],
            "n_clusters": len(np.unique(bunch.target)),
            "feature_names": bunch.feature_names,
            "target_names": bunch.target_names.tolist(),
            "type": "real_world",
        }
        return bunch.data.astype(np.float64), bunch.target.astype(int), meta

    elif clean_name in ["breast_cancer", "cancer"]:
        bunch = load_breast_cancer()
        meta = {
            "name": "breast_cancer",
            "n_samples": bunch.data.shape[0],
            "n_features": bunch.data.shape[1],
            "n_clusters": len(np.unique(bunch.target)),
            "feature_names": bunch.feature_names.tolist(),
            "target_names": bunch.target_names.tolist(),
            "type": "real_world",
        }
        return bunch.data.astype(np.float64), bunch.target.astype(int), meta

    elif clean_name == "digits":
        bunch = load_digits()
        meta = {
            "name": "digits",
            "n_samples": bunch.data.shape[0],
            "n_features": bunch.data.shape[1],
            "n_clusters": 10,
            "feature_names": [f"pixel_{i}" for i in range(bunch.data.shape[1])],
            "target_names": [str(i) for i in range(10)],
            "type": "real_world",
        }
        return bunch.data.astype(np.float64), bunch.target.astype(int), meta

    else:
        raise ValueError(
            f"Dataset '{name}' not recognized. Choose from: 'iris', 'wine', 'breast_cancer', 'digits'."
        )
