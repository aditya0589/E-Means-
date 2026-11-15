"""Convergence and trajectory visualization utilities."""

from typing import Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_inertia_vs_iterations(
    convergence_df: pd.DataFrame,
    title: str = "Inertia Reduction vs Iterations",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot inertia reduction across iterations for multiple algorithms."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    for algo in convergence_df["algorithm"].unique():
        sub = convergence_df[convergence_df["algorithm"] == algo]
        ax.plot(
            sub["iteration"],
            sub["inertia"],
            marker="o",
            markersize=4,
            linewidth=2,
            label=algo,
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Iteration / Epoch", fontsize=10)
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_inertia_vs_time(
    convergence_df: pd.DataFrame,
    title: str = "Inertia Reduction vs Elapsed Wall-Clock Time",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot inertia trajectory as a function of wall-clock time in seconds."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    for algo in convergence_df["algorithm"].unique():
        sub = convergence_df[convergence_df["algorithm"] == algo]
        ax.plot(
            sub["elapsed_time"],
            sub["inertia"],
            marker="s",
            markersize=4,
            linewidth=2,
            label=algo,
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Elapsed Time (seconds)", fontsize=10)
    ax.set_ylabel("Inertia (WCSS)", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig
