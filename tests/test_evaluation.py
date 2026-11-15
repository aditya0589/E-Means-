"""Unit tests for evaluation metrics and statistical tests."""

import numpy as np
import pandas as pd
from src.evaluation.clustering_metrics import compute_clustering_metrics
from src.evaluation.statistical_tests import (
    aggregate_experiment_runs,
    run_statistical_significance_test,
)


def test_clustering_metrics_computation():
    """Verify internal and external clustering metrics calculation."""
    # 2 well separated clusters
    X = np.vstack([
        np.zeros((50, 2)),
        np.ones((50, 2)) * 10.0,
    ])
    labels = np.array([0] * 50 + [1] * 50)
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    y_true = labels.copy()

    metrics = compute_clustering_metrics(X, labels, centroids, y_true=y_true)

    assert metrics["inertia"] == 0.0
    assert metrics["silhouette"] > 0.8
    assert metrics["davies_bouldin"] < 0.2
    assert metrics["adjusted_rand_index"] == 1.0
    assert metrics["normalized_mutual_info"] == 1.0


def test_statistical_tests():
    """Verify paired t-test and Wilcoxon test outputs."""
    a = np.array([10.0, 12.0, 11.0, 13.0, 12.5])
    b = np.array([8.0, 9.0, 8.5, 9.5, 9.0])

    res = run_statistical_significance_test(a, b, "AlgoA", "AlgoB")
    assert res["mean_diff"] > 0
    assert res["p_val_ttest"] < 0.05


def test_aggregation():
    """Verify repeated trials aggregation calculates correct mean, std, and CI."""
    df = pd.DataFrame({
        "dataset": ["blobs"] * 4,
        "algorithm": ["Lloyd"] * 4,
        "inertia": [100.0, 102.0, 98.0, 100.0],
    })
    agg = aggregate_experiment_runs(df, ["dataset", "algorithm"], ["inertia"])
    assert "inertia_mean" in agg.columns
    assert np.isclose(agg["inertia_mean"].iloc[0], 100.0)
    assert "inertia_ci95" in agg.columns
