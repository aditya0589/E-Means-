"""Clustering quality evaluation metrics (internal and external)."""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)


def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    max_silhouette_samples: int = 5000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute comprehensive clustering quality metrics.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster assignment for each point.
    centroids : np.ndarray
        Centroid coordinates of shape (n_clusters, n_features).
    y_true : np.ndarray, optional
        Ground truth class labels, if available.
    max_silhouette_samples : int, default=5000
        Maximum sample size used for Silhouette Score to avoid quadratic memory overhead.
    random_state : int, default=42
        Seed for silhouette sampling.

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed evaluation metrics.
    """
    n_samples = X.shape[0]
    n_unique_labels = len(np.unique(labels))

    # Internal metrics
    # 1. Inertia (Within-Cluster Sum of Squares)
    assigned_centroids = centroids[labels]
    inertia = float(np.sum((X - assigned_centroids) ** 2))

    metrics: Dict[str, Any] = {
        "inertia": inertia,
        "n_clusters_found": n_unique_labels,
    }

    if 1 < n_unique_labels < n_samples:
        # 2. Silhouette Score
        if n_samples > max_silhouette_samples:
            rng = np.random.RandomState(random_state)
            sample_indices = rng.choice(n_samples, size=max_silhouette_samples, replace=False)
            metrics["silhouette"] = float(
                silhouette_score(X[sample_indices], labels[sample_indices])
            )
        else:
            metrics["silhouette"] = float(silhouette_score(X, labels))

        # 3. Davies-Bouldin Index (lower is better)
        metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))

        # 4. Calinski-Harabasz Index (higher is better)
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    else:
        metrics["silhouette"] = 0.0
        metrics["davies_bouldin"] = float("nan")
        metrics["calinski_harabasz"] = 0.0

    # External metrics (only if ground-truth labels provided and valid)
    if y_true is not None:
        valid_mask = (y_true >= 0)  # filter out noise points labeled as -1
        if np.any(valid_mask):
            y_eval = y_true[valid_mask]
            l_eval = labels[valid_mask]

            metrics["adjusted_rand_index"] = float(adjusted_rand_score(y_eval, l_eval))
            metrics["normalized_mutual_info"] = float(
                normalized_mutual_info_score(y_eval, l_eval)
            )
            homo, comp, v_meas = homogeneity_completeness_v_measure(y_eval, l_eval)
            metrics["homogeneity"] = float(homo)
            metrics["completeness"] = float(comp)
            metrics["v_measure"] = float(v_meas)

    return metrics
