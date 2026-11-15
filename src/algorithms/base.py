"""Abstract base class for all clustering algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseClustering(ABC):
    """Abstract Base Class for clustering algorithms."""

    def __init__(self, n_clusters: int = 8, random_state: Optional[int] = 42):
        self.n_clusters = int(n_clusters)
        self.random_state = random_state
        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0
        self.n_iter_: int = 0
        self.init_time_: float = 0.0
        self.train_time_: float = 0.0
        self.total_time_: float = 0.0
        self.peak_memory_mb_: float = 0.0
        self.memory_delta_mb_: float = 0.0
        self.distance_calculations_: int = 0
        self.history_: Dict[str, list] = {
            "iteration": [],
            "inertia": [],
            "elapsed_time": [],
            "centroid_shift": [],
        }

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseClustering":
        """Compute clustering for input data matrix X."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the closest cluster index for each sample in X."""
        pass

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit clustering model and return cluster assignments for X."""
        return self.fit(X).predict(X)

    def get_centroids(self) -> np.ndarray:
        """Return the learned centroid coordinates of shape (n_clusters, n_features)."""
        if self.centroids_ is None:
            raise ValueError("Estimator has not been fitted yet.")
        return self.centroids_

    def get_metrics(self) -> Dict[str, Any]:
        """Return a dictionary of model metrics and profiling data."""
        return {
            "n_clusters": self.n_clusters,
            "inertia": self.inertia_,
            "n_iter": self.n_iter_,
            "init_time": self.init_time_,
            "train_time": self.train_time_,
            "total_time": self.total_time_,
            "peak_memory_mb": self.peak_memory_mb_,
            "memory_delta_mb": self.memory_delta_mb_,
            "distance_calculations": self.distance_calculations_,
        }

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate input array format, dimensions, and finite values."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_arr.shape}")
        if X_arr.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples ({X_arr.shape[0]}) must be >= n_clusters ({self.n_clusters})"
            )
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Input X contains NaN or infinite values.")
        return X_arr
