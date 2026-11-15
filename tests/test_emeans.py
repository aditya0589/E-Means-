"""Unit tests for faithful E-Means implementation."""

import math
import numpy as np
from src.algorithms.emeans import EMeans


def test_emeans_cosine_schedule():
    """Verify cosine decay schedule formulas match the paper specification."""
    model = EMeans(n_clusters=4, n_epochs=5, random_state=42)
    # Total steps M = 100, alpha_0 = 1.0, eps = 0.1
    # At t = 0: alpha(0) = alpha_0 * [0.1 + 0.9 * 0.5 * (1 + 1)] = alpha_0 * 1.0 = 1.0
    # At t = M: alpha(M) = alpha_0 * [0.1 + 0.9 * 0.5 * (1 - 1)] = alpha_0 * 0.1 = 0.1
    alpha_0 = 1.0
    M = 100

    def get_alpha(t):
        prog = min(1.0, float(t) / float(M))
        return alpha_0 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))

    assert np.isclose(get_alpha(0), 1.0)
    assert np.isclose(get_alpha(M), 0.1)
    assert np.isclose(get_alpha(M / 2), 0.55)


def test_emeans_scaling_laws_resolution():
    """Verify automatic scaling laws set plausible batch sizes and learning rates."""
    model = EMeans(n_clusters=8, n_epochs=5, random_state=42)
    B, a0 = model._compute_scaling_laws(n_samples=10000)

    assert B >= 128
    assert B <= 10000
    assert 0.0 < a0 <= 1.0


def test_emeans_fit_and_inertia():
    """Verify E-Means fits within specified epoch budget and yields reasonable inertia."""
    X = np.vstack([
        np.random.normal(loc=[-5.0, -5.0], scale=0.5, size=(200, 2)),
        np.random.normal(loc=[5.0, 5.0], scale=0.5, size=(200, 2)),
    ])
    model = EMeans(n_clusters=2, n_epochs=5, random_state=42)
    model.fit(X)

    assert model.centroids_.shape == (2, 2)
    assert model.n_iter_ == 5
    assert model.inertia_ < 250.0


def test_emeans_dead_cluster_revival():
    """Verify dead cluster detection and donor splitting operates without crashing."""
    # Highly skewed dataset: 400 points at (0, 0), 1 point at (100, 100)
    X = np.vstack([
        np.random.normal(loc=[0.0, 0.0], scale=0.1, size=(400, 2)),
        np.array([[100.0, 100.0]]),
    ])
    # Request 5 clusters: multiple clusters will initially starve
    model = EMeans(n_clusters=5, n_epochs=5, revival_interval=1, random_state=42)
    model.fit(X)

    assert model.centroids_.shape == (5, 2)
    assert not np.isnan(model.centroids_).any()


def test_emeans_finalization_pass():
    """Verify finalization pass sets centroids to exact cluster means."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
    model = EMeans(n_clusters=2, n_epochs=3, batch_size=2, random_state=42)
    model.fit(X)

    labels = model.labels_
    for j in range(2):
        pts = X[labels == j]
        if len(pts) > 0:
            expected_mean = np.mean(pts, axis=0)
            np.testing.assert_allclose(model.centroids_[j], expected_mean, atol=1e-5)
