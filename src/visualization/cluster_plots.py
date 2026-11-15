"""Publication-quality cluster visualization utilities."""

from typing import Optional, List, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def plot_clusters_2d(
    X_2d: np.ndarray,
    labels: np.ndarray,
    centroids_2d: np.ndarray,
    title: str = "Cluster Assignments",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot 2D scatter of clustered points with distinct centroid markers."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    scatter = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.6,
        s=25,
        edgecolors="none",
    )
    # Centroids
    ax.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c="red",
        marker="X",
        s=140,
        linewidths=1.5,
        edgecolors="black",
        label="Centroids",
        zorder=10,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Feature 1 / PC1", fontsize=10)
    ax.set_ylabel("Feature 2 / PC2", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_side_by_side_clusters(
    X_2d: np.ndarray,
    results_dict: Dict[str, tuple],
    dataset_name: str = "Dataset",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot side-by-side 2D cluster comparison across algorithms.

    results_dict: {algorithm_name: (labels, centroids_2d)}
    """
    n_algos = len(results_dict)
    fig, axes = plt.subplots(1, n_algos, figsize=(5 * n_algos, 4.5), dpi=dpi)
    if n_algos == 1:
        axes = [axes]

    for ax, (algo_name, (labels, centroids_2d)) in zip(axes, results_dict.items()):
        ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=labels,
            cmap="tab10",
            alpha=0.6,
            s=20,
            edgecolors="none",
        )
        ax.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            c="red",
            marker="X",
            s=120,
            linewidths=1.5,
            edgecolors="black",
            zorder=10,
        )
        ax.set_title(f"{algo_name}\n({dataset_name})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Dimension 1", fontsize=9)
        ax.set_ylabel("Dimension 2", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig
