"""Unit tests for K-Means++ initialization and clustering."""

import numpy as np
from src.algorithms.kmeans_plus_plus import KMeansPlusPlus


def test_kmeans_plus_plus_initialization():
    """Verify K-Means++ selects k distinct centers dispersed across data."""
    X = np.vstack([
        np.random.normal(loc=-10.0, scale=0.5, size=(50, 2)),
        np.random.normal(loc=0.0, scale=0.5, size=(50, 2)),
        np.random.normal(loc=10.0, scale=0.5, size=(50, 2)),
    ])

    model = KMeansPlusPlus(n_clusters=3, random_state=42)
    model.fit(X)

    centroids = model.centroids_
    assert centroids.shape == (3, 2)

    # Centers should be separated by roughly 10 units
    min_dist = np.min([
        np.linalg.norm(centroids[0] - centroids[1]),
        np.linalg.norm(centroids[0] - centroids[2]),
        np.linalg.norm(centroids[1] - centroids[2]),
    ])
    assert min_dist > 5.0


def test_kmeans_plus_plus_reproducibility():
    """Verify deterministic behavior with fixed random_state."""
    X = np.random.randn(150, 4)
    m1 = KMeansPlusPlus(n_clusters=4, random_state=99).fit(X)
    m2 = KMeansPlusPlus(n_clusters=4, random_state=99).fit(X)

    np.testing.assert_allclose(m1.centroids_, m2.centroids_)
    assert np.isclose(m1.inertia_, m2.inertia_)
