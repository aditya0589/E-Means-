"""Elkan's accelerated K-Means clustering algorithm using the triangle inequality."""

import time
from typing import Optional
import numpy as np

from .base import BaseClustering
from .kmeans_plus_plus import KMeansPlusPlus
from ..utils.random_state import check_random_state
from ..utils.memory_monitor import MemoryTracker


class ElkanKMeans(BaseClustering):
    """Elkan's triangle-inequality accelerated K-Means clustering algorithm.

    Reference: Elkan, C. (2003). Using the triangle inequality to accelerate k-means.
    ICML 2003.
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
        """Initialize using K-Means++ logic."""
        kpp = KMeansPlusPlus(n_clusters=self.n_clusters, random_state=self.random_state)
        return kpp._init_centroids(X, rng)

    def fit(self, X: np.ndarray) -> "ElkanKMeans":
        """Fit Elkan's accelerated K-Means clustering on X."""
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

        self.history_ = {
            "iteration": [],
            "inertia": [],
            "elapsed_time": [],
            "centroid_shift": [],
        }
        self.distance_calculations_ = 0

        # Step 2: Initialize Bounds and Distance Caches
        start_train = time.perf_counter()
        centroids = self.centroids_.copy()

        # Initial exact distance calculation from all points to all centroids
        # lower_bounds[i, j] <= ||x_i - c_j||
        # upper_bounds[i] >= ||x_i - c(x_i)||
        diff = X_arr[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        init_dists = np.sqrt(np.maximum(np.sum(diff ** 2, axis=2), 0.0))
        self.distance_calculations_ += n_samples * self.n_clusters

        labels = np.argmin(init_dists, axis=1)
        upper_bounds = np.min(init_dists, axis=1)
        lower_bounds = init_dists.copy()
        bounds_stale = np.zeros(n_samples, dtype=bool)

        for it in range(1, self.max_iter + 1):
            # 1. Compute inter-centroid distances: d(c_j, c_{j'})
            c_diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            center_dists = np.sqrt(np.maximum(np.sum(c_diff ** 2, axis=2), 0.0))
            self.distance_calculations_ += (self.n_clusters * (self.n_clusters - 1)) // 2

            # s[j] = 0.5 * min_{j' != j} d(c_j, c_{j'})
            np.fill_diagonal(center_dists, np.inf)
            s = 0.5 * np.min(center_dists, axis=1)
            np.fill_diagonal(center_dists, 0.0)

            # 2. Point assignment step with triangle inequality pruning
            for i in range(n_samples):
                c_curr = labels[i]

                # Lemma 1: If upper_bound <= s[c_curr], current centroid is guaranteed closest
                if upper_bounds[i] <= s[c_curr]:
                    continue

                for j in range(self.n_clusters):
                    if j == c_curr:
                        continue

                    # Pruning Rule 3a: Upper bound <= Lower bound to center j
                    if upper_bounds[i] <= lower_bounds[i, j]:
                        continue

                    # Pruning Rule 3b: Upper bound <= 0.5 * distance between centroids
                    if upper_bounds[i] <= 0.5 * center_dists[c_curr, j]:
                        continue

                    # Need to verify bounds if stale
                    if bounds_stale[i]:
                        # Recompute exact distance to current center
                        dist_curr = np.linalg.norm(X_arr[i] - centroids[c_curr])
                        self.distance_calculations_ += 1
                        upper_bounds[i] = dist_curr
                        lower_bounds[i, c_curr] = dist_curr
                        bounds_stale[i] = False
                    else:
                        dist_curr = upper_bounds[i]

                    # Check if test center j can still be closer
                    if dist_curr > lower_bounds[i, j] or dist_curr > 0.5 * center_dists[c_curr, j]:
                        dist_cand = np.linalg.norm(X_arr[i] - centroids[j])
                        self.distance_calculations_ += 1
                        lower_bounds[i, j] = dist_cand

                        if dist_cand < dist_curr:
                            c_curr = j
                            dist_curr = dist_cand
                            upper_bounds[i] = dist_cand
                            labels[i] = j

            # 3. Update centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.bincount(labels, minlength=self.n_clusters)

            for j in range(self.n_clusters):
                if counts[j] > 0:
                    new_centroids[j] = np.mean(X_arr[labels == j], axis=0)
                else:
                    # Empty cluster handling
                    farthest_idx = np.argmax(upper_bounds)
                    new_centroids[j] = X_arr[farthest_idx].copy()
                    upper_bounds[farthest_idx] = 0.0

            # 4. Centroid shifts
            centroid_shifts = np.linalg.norm(new_centroids - centroids, axis=1)
            max_shift = np.max(centroid_shifts)

            # 5. Update bounds based on centroid movement
            # u(x) <- u(x) + shift[c(x)]
            upper_bounds += centroid_shifts[labels]
            # l(x, j) <- max(0, l(x, j) - shift[j])
            lower_bounds = np.maximum(0.0, lower_bounds - centroid_shifts[np.newaxis, :])
            bounds_stale[:] = True

            # Record step metrics
            current_inertia = float(np.sum(upper_bounds ** 2))
            elapsed = time.perf_counter() - start_train

            self.history_["iteration"].append(it)
            self.history_["inertia"].append(current_inertia)
            self.history_["elapsed_time"].append(elapsed)
            self.history_["centroid_shift"].append(float(max_shift))

            centroids = new_centroids

            if max_shift <= np.sqrt(self.tol):
                break

        self.n_iter_ = it
        self.centroids_ = centroids
        self.labels_ = labels
        self.train_time_ = time.perf_counter() - start_train
        self.total_time_ = time.perf_counter() - start_total

        # Final exact inertia
        final_dist_sq = np.sum((X_arr - self.centroids_[labels]) ** 2, axis=1)
        self.inertia_ = float(np.sum(final_dist_sq))

        self.peak_memory_mb_, self.memory_delta_mb_ = mem_tracker.stop()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the closest cluster index for each sample in X."""
        if self.centroids_ is None:
            raise ValueError("Estimator has not been fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        x_sq = np.sum(X_arr ** 2, axis=1, keepdims=True)
        c_sq = np.sum(self.centroids_ ** 2, axis=1, keepdims=True).T
        dist_sq = np.maximum(x_sq - 2.0 * np.dot(X_arr, self.centroids_.T) + c_sq, 0.0)
        return np.argmin(dist_sq, axis=1)
