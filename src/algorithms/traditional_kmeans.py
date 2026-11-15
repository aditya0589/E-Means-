"""Traditional Lloyd's K-Means clustering algorithm implemented from scratch."""

import time
from typing import Optional
import numpy as np

from .base import BaseClustering
from ..utils.random_state import check_random_state
from ..utils.memory_monitor import MemoryTracker


class TraditionalKMeans(BaseClustering):
    """Lloyd's K-Means clustering algorithm from scratch.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters (k).
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence based on max centroid shift squared.
    random_state : int, optional, default=42
        Deterministic random seed.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = 42,
    ):
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _init_centroids(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Uniform random initialization from data samples without replacement."""
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        return X[indices].copy()

    @staticmethod
    def _compute_distances_sq(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Vectorized computation of squared Euclidean distances between X and centroids.

        ||X - C||^2 = ||X||^2 - 2 X C^T + ||C||^2
        """
        x_sq = np.sum(X ** 2, axis=1, keepdims=True)
        c_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T
        dist_sq = x_sq - 2.0 * np.dot(X, centroids.T) + c_sq
        return np.maximum(dist_sq, 0.0)

    def fit(self, X: np.ndarray) -> "TraditionalKMeans":
        """Compute Lloyd's K-Means clustering on X."""
        X_arr = self._validate_input(X)
        n_samples, n_features = X_arr.shape
        rng = check_random_state(self.random_state)

        mem_tracker = MemoryTracker()
        mem_tracker.start()
        start_total = time.perf_counter()

        # Step 1: Centroid Initialization
        start_init = time.perf_counter()
        self.centroids_ = self._init_centroids(X_arr, rng)
        self.init_time_ = time.perf_counter() - start_init

        # Reset history
        self.history_ = {
            "iteration": [],
            "inertia": [],
            "elapsed_time": [],
            "centroid_shift": [],
        }
        self.distance_calculations_ = 0

        # Step 2: Lloyd's Iterative EM Optimization Loop
        start_train = time.perf_counter()
        centroids = self.centroids_.copy()

        for it in range(1, self.max_iter + 1):
            # Expectation: Compute pairwise squared distances and assign labels
            dist_sq = self._compute_distances_sq(X_arr, centroids)
            self.distance_calculations_ += n_samples * self.n_clusters
            labels = np.argmin(dist_sq, axis=1)

            # Maximization: Recompute centroids as means of assigned points
            new_centroids = np.zeros_like(centroids)
            counts = np.bincount(labels, minlength=self.n_clusters)

            for j in range(self.n_clusters):
                if counts[j] > 0:
                    new_centroids[j] = np.mean(X_arr[labels == j], axis=0)
                else:
                    # Handle empty cluster: reinitialize to the farthest data point
                    min_dist_to_centers = np.min(dist_sq, axis=1)
                    farthest_idx = np.argmax(min_dist_to_centers)
                    new_centroids[j] = X_arr[farthest_idx].copy()
                    # Prevent multiple empty clusters picking the same point
                    dist_sq[farthest_idx, :] = -1.0

            # Compute centroid shift and convergence criteria
            shifts_sq = np.sum((new_centroids - centroids) ** 2, axis=1)
            max_shift_sq = np.max(shifts_sq)

            # Record step metrics
            current_inertia = float(np.sum(np.min(dist_sq, axis=1)))
            elapsed = time.perf_counter() - start_train

            self.history_["iteration"].append(it)
            self.history_["inertia"].append(current_inertia)
            self.history_["elapsed_time"].append(elapsed)
            self.history_["centroid_shift"].append(float(np.sqrt(max_shift_sq)))

            centroids = new_centroids

            if max_shift_sq <= self.tol:
                break

        self.n_iter_ = it
        self.centroids_ = centroids
        self.train_time_ = time.perf_counter() - start_train
        self.total_time_ = time.perf_counter() - start_total

        # Final assignments and inertia calculation
        final_dist_sq = self._compute_distances_sq(X_arr, self.centroids_)
        self.distance_calculations_ += n_samples * self.n_clusters
        self.labels_ = np.argmin(final_dist_sq, axis=1)
        self.inertia_ = float(np.sum(np.min(final_dist_sq, axis=1)))

        self.peak_memory_mb_, self.memory_delta_mb_ = mem_tracker.stop()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the closest cluster index for each point in X."""
        if self.centroids_ is None:
            raise ValueError("Estimator has not been fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        dist_sq = self._compute_distances_sq(X_arr, self.centroids_)
        return np.argmin(dist_sq, axis=1)
