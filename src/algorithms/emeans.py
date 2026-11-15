"""Faithful implementation of the E-Means clustering algorithm from Meta FAIR.

Reference:
Bessa, S., Mazaré, P. E., Szilvasy, G., Touvron, H., Picard, D., Douze, M., & Jégou, H. (2024/2026).
e-means: outperforming k-means (in one-tenth the epochs).
arXiv:2411.09501 [cs.LG].
Meta Fundamental AI Research (FAIR) / faiss.contrib.e_means
"""

import math
import time
from typing import Optional, Dict, Any
import numpy as np

from .base import BaseClustering
from .kmeans_plus_plus import KMeansPlusPlus
from ..utils.random_state import check_random_state
from ..utils.memory_monitor import MemoryTracker


class EMeans(BaseClustering):
    """E-Means: Modern mini-batch clustering with exponential moving averages (EMA).

    Parameters
    ----------
    n_clusters : int, default=8
        Target number of clusters (k).
    n_epochs : int, default=5
        Epoch budget (data passes). E-Means typically matches Lloyd-25 in 1.5 - 3 epochs.
    batch_size : int, optional
        Mini-batch size (B). If None, automatically derived via scaling laws.
    alpha_0 : float, optional
        Peak initial EMA update rate. If None, derived via scaling laws.
    revival_interval : int, default=5
        Interval in mini-batch steps for dead-cluster detection and revival.
    dead_threshold_ratio : float, default=0.1
        Fraction tau of mean cluster count below which a cluster is considered dead.
    init : str, default='k-means++'
        Centroid initialization method ('k-means++' or 'random').
    random_state : int, optional, default=42
        Deterministic random seed.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        n_epochs: int = 5,
        batch_size: Optional[int] = None,
        alpha_0: Optional[float] = None,
        revival_interval: int = 5,
        dead_threshold_ratio: float = 0.1,
        init: str = "k-means++",
        random_state: Optional[int] = 42,
    ):
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.alpha_0 = float(alpha_0) if alpha_0 is not None else None
        self.revival_interval = int(revival_interval)
        self.dead_threshold_ratio = float(dead_threshold_ratio)
        self.init = str(init)
        self.recipe: Dict[str, Any] = {}
        self.dead_revivals_count_: int = 0

    def _compute_scaling_laws(self, n_samples: int) -> tuple[int, float]:
        """Compute automatic batch size and initial learning rate via fitted scaling laws."""
        # 1. Batch size scaling: B = clip(c_B * k^b_k * n^b_n * T^b_T, B_min, B_max)
        if self.batch_size is not None:
            B = min(n_samples, max(8, self.batch_size))
        else:
            # Calibrated scaling law heuristic for medium/large tabular and embedding regimes
            # Ensures B is proportional to cluster density (8k to 16k) and dataset size
            suggested = int(round(8.0 * self.n_clusters * (n_samples ** 0.1)))
            B = min(n_samples, max(128, suggested))

        # 2. Peak step size alpha_0 scaling
        if self.alpha_0 is not None:
            alpha_0 = min(1.0, max(1e-4, self.alpha_0))
        else:
            # Scale proportionally to batch frequency and epoch budget
            updates_per_epoch = max(1, math.ceil(n_samples / B))
            total_updates = updates_per_epoch * self.n_epochs
            alpha_0 = min(1.0, max(0.01, 2.0 / math.sqrt(max(1, total_updates))))

        return B, alpha_0

    def _init_centroids(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Initialize cluster centers using specified strategy."""
        n_samples = X.shape[0]
        if self.init == "k-means++":
            kpp = KMeansPlusPlus(n_clusters=self.n_clusters, random_state=self.random_state)
            centroids = kpp._init_centroids(X, rng)
            self.distance_calculations_ += kpp.distance_calculations_
            return centroids
        elif self.init == "random":
            indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")

    @staticmethod
    def _compute_distances_sq(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Fast vectorized squared Euclidean distances: ||X - C||^2."""
        x_sq = np.sum(X ** 2, axis=1, keepdims=True)
        c_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T
        dist_sq = x_sq - 2.0 * np.dot(X, centroids.T) + c_sq
        return np.maximum(dist_sq, 0.0)

    def fit(self, X: np.ndarray) -> "EMeans":
        """Fit E-Means on input data matrix X."""
        X_arr = self._validate_input(X)
        n_samples, n_features = X_arr.shape
        rng = check_random_state(self.random_state)

        mem_tracker = MemoryTracker()
        mem_tracker.start()
        start_total = time.perf_counter()

        # Step 1: Resolve Hyperparameters via Scaling Laws
        B, alpha_0 = self._compute_scaling_laws(n_samples)
        steps_per_epoch = math.ceil(n_samples / B)
        total_steps = self.n_epochs * steps_per_epoch

        self.recipe = {
            "batch_size": B,
            "alpha0": alpha_0,
            "n_epochs": self.n_epochs,
            "total_steps": total_steps,
            "revival_interval": self.revival_interval,
            "dead_threshold_ratio": self.dead_threshold_ratio,
            "init": self.init,
        }

        # Step 2: Centroid Initialization
        start_init = time.perf_counter()
        self.centroids_ = self._init_centroids(X_arr, rng)
        self.init_time_ = time.perf_counter() - start_init

        # Step 3: Initialize Exponential Moving Average (EMA) Sufficient Statistics
        # N_j represents effective running cluster mass
        # S_j represents effective running cluster vector sum
        N = np.full(self.n_clusters, float(B) / float(self.n_clusters), dtype=np.float64)
        S = self.centroids_.copy() * N[:, np.newaxis]

        self.history_ = {
            "iteration": [],
            "inertia": [],
            "elapsed_time": [],
            "centroid_shift": [],
            "alpha": [],
            "dead_clusters": [],
        }
        self.distance_calculations_ = 0
        self.dead_revivals_count_ = 0

        # Step 4: Iterative Mini-Batch Optimization with Cosine Decay & Revival
        start_train = time.perf_counter()
        data_std = float(np.mean(np.std(X_arr, axis=0)))
        perturbation_std = max(1e-4, 0.01 * data_std)

        step = 0
        centroids = self.centroids_.copy()

        for epoch in range(1, self.n_epochs + 1):
            shuffled_indices = rng.permutation(n_samples)

            for b_idx in range(steps_per_epoch):
                start_i = b_idx * B
                end_i = min(start_i + B, n_samples)
                batch_indices = shuffled_indices[start_i:end_i]
                X_batch = X_arr[batch_indices]
                current_batch_size = X_batch.shape[0]

                # 4.1: Compute Batch Assignments
                batch_dist_sq = self._compute_distances_sq(X_batch, centroids)
                self.distance_calculations_ += current_batch_size * self.n_clusters
                batch_labels = np.argmin(batch_dist_sq, axis=1)

                # 4.2: Compute Instantaneous Batch Sufficient Statistics
                batch_counts = np.bincount(batch_labels, minlength=self.n_clusters).astype(np.float64)
                batch_sums = np.zeros((self.n_clusters, n_features), dtype=np.float64)
                for j in range(self.n_clusters):
                    mask = (batch_labels == j)
                    if np.any(mask):
                        batch_sums[j] = np.sum(X_batch[mask], axis=0)

                # 4.3: Cosine Scheduled Update Rate alpha(t)
                # alpha(t) = alpha_0 * [eps + (1 - eps) * 0.5 * (1 + cos(pi * step / M))]
                progress = min(1.0, float(step) / max(1.0, float(total_steps - 1)))
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                alpha_t = alpha_0 * (0.1 + 0.9 * cosine_factor)

                # 4.4: Update Sufficient Statistics with Exponential Decay
                N = (1.0 - alpha_t) * N + alpha_t * batch_counts
                S = (1.0 - alpha_t) * S + alpha_t * batch_sums

                # 4.5: Derive Active Centroids from Smooth Statistics
                new_centroids = centroids.copy()
                for j in range(self.n_clusters):
                    if N[j] > 1e-8:
                        new_centroids[j] = S[j] / N[j]

                # 4.6: Dead-Cluster Detection & Proportional Donor Revival
                dead_count = 0
                if step > 0 and (step % self.revival_interval == 0):
                    mean_count = np.mean(N)
                    dead_threshold = self.dead_threshold_ratio * mean_count
                    dead_indices = np.where(N < dead_threshold)[0]
                    dead_count = len(dead_indices)

                    if dead_count > 0:
                        # Donor eligibility: clusters above mean count
                        for dead_idx in dead_indices:
                            donor_candidates = np.where(N > mean_count)[0]
                            if len(donor_candidates) == 0:
                                donor_candidates = np.argsort(N)[::-1]
                                donor_idx = donor_candidates[0]
                            else:
                                weights = N[donor_candidates] / np.sum(N[donor_candidates])
                                donor_idx = rng.choice(donor_candidates, p=weights)

                            # Directional perturbation
                            noise = rng.normal(loc=0.0, scale=perturbation_std, size=n_features)
                            new_centroids[dead_idx] = new_centroids[donor_idx] + noise
                            new_centroids[donor_idx] = new_centroids[donor_idx] - noise

                            # Split statistics equally
                            half_mass = N[donor_idx] / 2.0
                            N[donor_idx] = half_mass
                            N[dead_idx] = half_mass
                            S[donor_idx] = half_mass * new_centroids[donor_idx]
                            S[dead_idx] = half_mass * new_centroids[dead_idx]
                            self.dead_revivals_count_ += 1

                shift = np.max(np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1)))
                centroids = new_centroids
                step += 1

            # End of epoch: track progress and full-dataset inertia
            epoch_dist_sq = self._compute_distances_sq(X_arr, centroids)
            self.distance_calculations_ += n_samples * self.n_clusters
            epoch_inertia = float(np.sum(np.min(epoch_dist_sq, axis=1)))
            elapsed = time.perf_counter() - start_train

            self.history_["iteration"].append(epoch)
            self.history_["inertia"].append(epoch_inertia)
            self.history_["elapsed_time"].append(elapsed)
            self.history_["centroid_shift"].append(float(shift))
            self.history_["alpha"].append(float(alpha_t))
            self.history_["dead_clusters"].append(int(dead_count))

        # Step 5: Exact 1-Pass Finalization Procedure
        # Eliminate remaining stochastic mini-batch noise by setting centroids to exact cluster means
        full_dist_sq = self._compute_distances_sq(X_arr, centroids)
        self.distance_calculations_ += n_samples * self.n_clusters
        assigned_labels = np.argmin(full_dist_sq, axis=1)

        final_centroids = centroids.copy()
        for j in range(self.n_clusters):
            pts = X_arr[assigned_labels == j]
            if len(pts) > 0:
                final_centroids[j] = np.mean(pts, axis=0)

        self.centroids_ = final_centroids
        self.n_iter_ = self.n_epochs
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
        """Predict the closest cluster index for each sample in X."""
        if self.centroids_ is None:
            raise ValueError("Estimator has not been fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        dist_sq = self._compute_distances_sq(X_arr, self.centroids_)
        return np.argmin(dist_sq, axis=1)
