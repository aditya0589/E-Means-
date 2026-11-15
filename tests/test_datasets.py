"""Unit tests for synthetic and real-world dataset loaders and preprocessing."""

import numpy as np
from src.datasets.synthetic import (
    generate_gaussian_blobs,
    generate_anisotropic_blobs,
    generate_moons,
    generate_circles,
    generate_noisy_clusters,
    generate_high_dimensional_dataset,
)
from src.datasets.loader import load_dataset_by_name
from src.datasets.preprocessing import preprocess_features, reduce_to_2d_for_visualization


def test_synthetic_generators():
    """Verify all synthetic generators return expected shapes and valid numbers."""
    X_b, y_b, m_b = generate_gaussian_blobs(n_samples=100, n_features=3, n_clusters=3)
    assert X_b.shape == (100, 3)
    assert len(np.unique(y_b)) == 3

    X_a, y_a, _ = generate_anisotropic_blobs(n_samples=100, n_clusters=2)
    assert X_a.shape == (100, 2)

    X_m, y_m, _ = generate_moons(n_samples=100)
    assert X_m.shape == (100, 2)

    X_c, y_c, _ = generate_circles(n_samples=100)
    assert X_c.shape == (100, 2)

    X_n, y_n, _ = generate_noisy_clusters(n_samples=100, noise_ratio=0.1)
    assert X_n.shape == (100, 2)

    X_h, y_h, _ = generate_high_dimensional_dataset(n_samples=200, n_features=25, n_clusters=4)
    assert X_h.shape == (200, 25)


def test_real_world_loaders():
    """Verify real-world loaders load standard benchmark datasets."""
    for dname, expected_k in [("iris", 3), ("wine", 3), ("breast_cancer", 2), ("digits", 10)]:
        X, y, meta = load_dataset_by_name(dname)
        assert X.ndim == 2
        assert len(np.unique(y)) == expected_k
        assert meta["n_clusters"] == expected_k


def test_preprocessing_and_pca():
    """Verify StandardScaler and PCA 2D projection."""
    X = np.random.randn(100, 5) * 10.0 + 5.0
    X_std, scaler = preprocess_features(X, method="standard")
    np.testing.assert_allclose(np.mean(X_std, axis=0), np.zeros(5), atol=1e-5)
    np.testing.assert_allclose(np.std(X_std, axis=0), np.ones(5), atol=1e-5)

    X_2d, pca = reduce_to_2d_for_visualization(X_std)
    assert X_2d.shape == (100, 2)
