"""Synthetic dataset generators for clustering evaluation."""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from ..utils.random_state import check_random_state


def generate_gaussian_blobs(
    n_samples: int = 1000,
    n_features: int = 2,
    n_clusters: int = 4,
    cluster_std: float = 1.0,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate isotropic Gaussian blobs."""
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    meta = {
        "name": "gaussian_blobs",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": n_clusters,
        "cluster_std": cluster_std,
        "type": "synthetic",
    }
    return X, y, meta


def generate_anisotropic_blobs(
    n_samples: int = 1000,
    n_clusters: int = 3,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate anisotropic (stretched / correlated) clusters."""
    rng = check_random_state(random_state)
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=n_clusters,
        random_state=random_state,
    )
    # Apply affine transformation matrix to stretch clusters
    transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X_aniso = np.dot(X, transformation)
    meta = {
        "name": "anisotropic_blobs",
        "n_samples": n_samples,
        "n_features": 2,
        "n_clusters": n_clusters,
        "type": "synthetic",
    }
    return X_aniso, y, meta


def generate_moons(
    n_samples: int = 1000,
    noise: float = 0.08,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate two interleaving half-moon clusters."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    meta = {
        "name": "moons",
        "n_samples": n_samples,
        "n_features": 2,
        "n_clusters": 2,
        "noise": noise,
        "type": "synthetic",
    }
    return X, y, meta


def generate_circles(
    n_samples: int = 1000,
    factor: float = 0.5,
    noise: float = 0.05,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate concentric circles."""
    X, y = make_circles(n_samples=n_samples, factor=factor, noise=noise, random_state=random_state)
    meta = {
        "name": "circles",
        "n_samples": n_samples,
        "n_features": 2,
        "n_clusters": 2,
        "factor": factor,
        "noise": noise,
        "type": "synthetic",
    }
    return X, y, meta


def generate_noisy_clusters(
    n_samples: int = 1000,
    n_features: int = 2,
    n_clusters: int = 4,
    noise_ratio: float = 0.1,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate Gaussian blobs with added uniform background noise."""
    rng = check_random_state(random_state)
    n_noise = int(round(n_samples * noise_ratio))
    n_clean = n_samples - n_noise

    X_clean, y_clean = make_blobs(
        n_samples=n_clean,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=rng.randint(0, 10000),
    )

    low = np.min(X_clean, axis=0) - 2.0
    high = np.max(X_clean, axis=0) + 2.0
    X_noise = rng.uniform(low=low, high=high, size=(n_noise, n_features))
    y_noise = np.full(n_noise, -1, dtype=int)

    X_combined = np.vstack([X_clean, X_noise])
    y_combined = np.concatenate([y_clean, y_noise])

    shuffle_idx = rng.permutation(n_samples)
    X_shuffled = X_combined[shuffle_idx]
    y_shuffled = y_combined[shuffle_idx]

    meta = {
        "name": "noisy_clusters",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "type": "synthetic",
    }
    return X_shuffled, y_shuffled, meta


def generate_high_dimensional_dataset(
    n_samples: int = 5000,
    n_features: int = 50,
    n_clusters: int = 5,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate high-dimensional Gaussian clusters."""
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.5,
        random_state=random_state,
    )
    meta = {
        "name": f"high_dim_{n_features}d",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": n_clusters,
        "type": "synthetic",
    }
    return X, y, meta
