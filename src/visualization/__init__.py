"""Visualization package for publication-quality cluster, convergence, and performance figures."""

from .cluster_plots import plot_clusters_2d, plot_side_by_side_clusters
from .convergence_plots import plot_inertia_vs_iterations, plot_inertia_vs_time
from .performance_plots import (
    plot_execution_time_comparison,
    plot_scalability_curve,
    plot_boxplots_by_metric,
    plot_metric_bars,
)

__all__ = [
    "plot_clusters_2d",
    "plot_side_by_side_clusters",
    "plot_inertia_vs_iterations",
    "plot_inertia_vs_time",
    "plot_execution_time_comparison",
    "plot_scalability_curve",
    "plot_boxplots_by_metric",
    "plot_metric_bars",
]
