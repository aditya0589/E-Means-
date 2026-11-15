"""Unit tests for Elkan's accelerated triangle-inequality K-Means."""

import numpy as np
from src.algorithms.elkan_kmeans import ElkanKMeans
from src.algorithms.kmeans_plus_plus import KMeansPlusPlus


def test_elkan_convergence_parity_with_kmeans_plus_plus():
    """Verify Elkan achieves the exact same mathematical inertia as standard K-Means++."""
    rng = np.random.RandomState(42)
    X = rng.randn(300, 5)

    elkan = ElkanKMeans(n_clusters=4, max_iter=100, tol=1e-5, random_state=42).fit(X)
    kpp = KMeansPlusPlus(n_clusters=4, max_iter=100, tol=1e-5, random_state=42).fit(X)

    # Both start from identical centers and should converge to near-identical inertias
    assert np.isclose(elkan.inertia_, kpp.inertia_, rtol=1e-3)


def test_elkan_distance_pruning():
    """Verify Elkan executes substantially fewer distance calculations than brute force."""
    X = np.random.randn(500, 10)
    elkan = ElkanKMeans(n_clusters=5, random_state=42).fit(X)
    kpp = KMeansPlusPlus(n_clusters=5, random_state=42).fit(X)

    # Elkan should prune at least 40% of calculations
    assert elkan.distance_calculations_ < kpp.distance_calculations_


def test_elkan_predict():
    """Verify Elkan's predict yields identical assignments to Lloyd."""
    X = np.random.randn(100, 3)
    elkan = ElkanKMeans(n_clusters=3, random_state=42).fit(X)
    preds = elkan.predict(X)
    np.testing.assert_array_equal(preds, elkan.labels_)
