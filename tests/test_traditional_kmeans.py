"""Unit tests for Traditional Lloyd's K-Means implementation."""

import pytest
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from src.algorithms.traditional_kmeans import TraditionalKMeans


def test_initialization_shape_and_range():
    """Verify centroid initialization produces correct matrix shape and values from data."""
    X = np.random.randn(100, 5)
    model = TraditionalKMeans(n_clusters=4, random_state=42)
    model.fit(X)

    centroids = model.get_centroids()
    assert centroids.shape == (4, 5)
    assert model.labels_.shape == (100,)
    assert set(np.unique(model.labels_)).issubset({0, 1, 2, 3})


def test_inertia_calculation():
    """Verify inertia is exactly the sum of squared distances to assigned centroids."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
    model = TraditionalKMeans(n_clusters=2, random_state=42)
    model.fit(X)

    assigned_centroids = model.centroids_[model.labels_]
    expected_inertia = float(np.sum((X - assigned_centroids) ** 2))
    assert np.isclose(model.inertia_, expected_inertia, rtol=1e-5)


def test_prediction_on_unseen_data():
    """Verify predict assigns unseen samples to the closest centroid."""
    X = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]])
    model = TraditionalKMeans(n_clusters=2, random_state=42)
    model.fit(X)

    X_test = np.array([[0.05, 0.05], [9.9, 9.9]])
    preds = model.predict(X_test)
    assert preds[0] == model.predict([[0.0, 0.0]])[0]
    assert preds[1] == model.predict([[10.0, 10.0]])[0]


def test_reproducibility():
    """Verify identical random seeds yield identical clustering outputs."""
    X = np.random.randn(200, 4)
    m1 = TraditionalKMeans(n_clusters=3, random_state=123).fit(X)
    m2 = TraditionalKMeans(n_clusters=3, random_state=123).fit(X)

    np.testing.assert_allclose(m1.centroids_, m2.centroids_)
    np.testing.assert_array_equal(m1.labels_, m2.labels_)
    assert np.isclose(m1.inertia_, m2.inertia_)


def test_empty_cluster_handling():
    """Verify algorithm recovers gracefully from degenerate cluster collapse."""
    # Data with distinct outlier
    X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [100.0, 100.0]])
    model = TraditionalKMeans(n_clusters=3, max_iter=20, random_state=42)
    model.fit(X)

    # Should partition without error and produce 3 centroids
    assert model.centroids_.shape == (3, 2)
    assert not np.isnan(model.centroids_).any()


def test_duplicate_samples():
    """Verify algorithm operates stably on duplicate identical data points."""
    X = np.tile([1.0, 2.0, 3.0], (50, 1))
    model = TraditionalKMeans(n_clusters=1, random_state=42)
    model.fit(X)
    assert np.isclose(model.inertia_, 0.0)


def test_invalid_inputs():
    """Verify proper exceptions raised for invalid dimensions, NaNs, and k > n."""
    model = TraditionalKMeans(n_clusters=5)

    with pytest.raises(ValueError):
        model.fit(np.random.randn(3, 4))  # n_samples < n_clusters

    with pytest.raises(ValueError):
        model.fit(np.random.randn(10))  # 1D array instead of 2D

    with pytest.raises(ValueError):
        X_nan = np.random.randn(20, 2)
        X_nan[0, 0] = np.nan
        model.fit(X_nan)  # NaNs present


def test_cross_check_against_sklearn():
    """Cross-check TraditionalKMeans with scikit-learn KMeans given identical initial centers."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)

    # Pick identical initial centers
    init_centers = X[:3].copy()

    # Sklearn
    sk_model = SklearnKMeans(n_clusters=3, init=init_centers, n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd")
    sk_model.fit(X)

    # Our implementation
    our_model = TraditionalKMeans(n_clusters=3, max_iter=300, tol=1e-4, random_state=42)
    # Monkey-patch initialization to use identical starting points
    our_model._init_centroids = lambda data, rand: init_centers.copy()
    our_model.fit(X)

    # Check that final inertias are within 1% of each other
    assert np.isclose(our_model.inertia_, sk_model.inertia_, rtol=0.01)
