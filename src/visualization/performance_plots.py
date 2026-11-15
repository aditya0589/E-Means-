"""Performance, scalability, and statistical comparative visualization."""

from typing import Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_execution_time_comparison(
    df_results: pd.DataFrame,
    title: str = "Execution Time Comparison Across Algorithms",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Bar chart comparing initialization and training execution times."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    algorithms = df_results["algorithm"].tolist()
    init_times = df_results["init_time_sec"].tolist()
    train_times = df_results["train_time_sec"].tolist()

    x = range(len(algorithms))
    width = 0.55

    ax.bar(x, init_times, width, label="Initialization Time", color="#4C72B0")
    ax.bar(x, train_times, width, bottom=init_times, label="Training Time", color="#55A868")

    ax.set_ylabel("Execution Time (seconds)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15, ha="right", fontsize=9)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_scalability_curve(
    df_scalability: pd.DataFrame,
    param_col: str = "n_samples",
    metric_col: str = "total_time_sec",
    param_label: str = "Sample Size (n)",
    metric_label: str = "Total Runtime (s)",
    title: str = "Scalability Analysis",
    log_scale: bool = False,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot scalability curve (e.g., sample size or dimensions vs runtime)."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    for algo in df_scalability["algorithm"].unique():
        sub = df_scalability[df_scalability["algorithm"] == algo]
        sub = sub.sort_values(by=param_col)
        ax.plot(
            sub[param_col],
            sub[metric_col],
            marker="o",
            linewidth=2,
            label=algo,
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(param_label, fontsize=10)
    ax.set_ylabel(metric_label, fontsize=10)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_boxplots_by_metric(
    df_runs: pd.DataFrame,
    metric_col: str = "inertia",
    title: str = "Algorithm Metric Distribution Across Random Seeds",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Boxplot displaying distribution of metrics across repeated trials."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    sns.boxplot(
        data=df_runs,
        x="algorithm",
        y=metric_col,
        palette="Set2",
        ax=ax,
        width=0.45,
    )
    sns.stripplot(
        data=df_runs,
        x="algorithm",
        y=metric_col,
        color="black",
        alpha=0.6,
        size=5,
        jitter=0.15,
        ax=ax,
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Algorithm", fontsize=10)
    ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_metric_bars(
    df_results: pd.DataFrame,
    metrics: List[str] = ["silhouette", "davies_bouldin"],
    title: str = "Clustering Quality Metrics Comparison",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Grouped bar chart for multiple evaluation metrics."""
    available_metrics = [m for m in metrics if m in df_results.columns]
    if not available_metrics:
        return plt.figure()

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 4.5), dpi=dpi)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        x = range(len(df_results["algorithm"]))
        ax.bar(
            x,
            df_results[metric],
            color="#5C82A6",
            width=0.5,
            edgecolor="#333333",
        )
        ax.set_title(metric.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(df_results["algorithm"], rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig
