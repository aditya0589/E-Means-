"""K-Means clustering with K-Means++ initialization from scratch."""

import time
from typing import Optional
import numpy as np

from .traditional_kmeans import TraditionalKMeans
from ..utils.random_state import check_random_state


class KMeansPlusPlus(TraditionalKMeans):
    """K-Means algorithm with K-Means++ probabilistic initialization.

    Reference: Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding.
    """

    def _init_centroids(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """K-Means++ initialization with D^2 probabilistic weighting."""
        n_samples, n_features = X.shape
        centroids = np.empty((self.n_clusters, n_features), dtype=np.float64)

        # Step 1: Pick the first center uniformly at random
        first_idx = rng.randint(0, n_samples)
        centroids[0] = X[first_idx]

        # Step 2: Compute squared distances to the first center
        min_dist_sq = np.sum((X - centroids[0]) ** 2, axis=1)
        self.distance_calculations_ += n_samples

        # Step 3: Choose remaining k - 1 centers
        for c_idx in range(1, self.n_clusters):
            total_dist_sq = np.sum(min_dist_sq)
            if total_dist_sq == 0:
                # All points overlap with existing centroids; pick remaining uniformly
                remaining_indices = rng.choice(n_samples, size=self.n_clusters - c_idx, replace=False)
                centroids[c_idx:] = X[remaining_indices]
                break

            probabilities = min_dist_sq / total_dist_sq
            cumulative_probs = np.cumsum(probabilities)
            r = rng.rand()
            chosen_idx = np.searchsorted(cumulative_probs, r)
            chosen_idx = min(chosen_idx, n_samples - 1)

            centroids[c_idx] = X[chosen_idx]

            # Update minimum squared distance to nearest centroid
            new_dist_sq = np.sum((X - centroids[c_idx]) ** 2, axis=1)
            self.distance_calculations_ += n_samples
            min_dist_sq = np.minimum(min_dist_sq, new_dist_sq)

        return centroids
