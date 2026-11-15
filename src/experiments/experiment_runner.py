"""Comprehensive experiment orchestrator executing Experiments 1 through 8."""

from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import pandas as pd
from tabulate import tabulate

from ..algorithms.traditional_kmeans import TraditionalKMeans
from ..algorithms.kmeans_plus_plus import KMeansPlusPlus
from ..algorithms.elkan_kmeans import ElkanKMeans
from ..algorithms.emeans import EMeans

from ..datasets.synthetic import (
    generate_gaussian_blobs,
    generate_anisotropic_blobs,
    generate_moons,
    generate_circles,
    generate_noisy_clusters,
)
from ..datasets.loader import load_dataset_by_name
from ..datasets.preprocessing import preprocess_features, reduce_to_2d_for_visualization

from ..evaluation.clustering_metrics import compute_clustering_metrics
from ..evaluation.performance_metrics import extract_performance_profile
from ..evaluation.statistical_tests import aggregate_experiment_runs, run_statistical_significance_test

from ..visualization.cluster_plots import plot_side_by_side_clusters
from ..visualization.convergence_plots import plot_inertia_vs_iterations, plot_inertia_vs_time
from ..visualization.performance_plots import (
    plot_execution_time_comparison,
    plot_scalability_curve,
    plot_boxplots_by_metric,
    plot_metric_bars,
)

from .benchmark import run_single_benchmark
from .scalability import (
    run_sample_size_scalability,
    run_dimensionality_scalability,
    run_cluster_count_scalability,
)
from .configurations import BENCHMARK_SEEDS, SAMPLE_SIZES, DIMENSION_SIZES, CLUSTER_COUNTS, NOISE_STD_LEVELS
from ..utils.result_saver import ResultSaver
from ..utils.logging_config import get_logger

logger = get_logger("experiment_runner")

ALGORITHMS = {
    "Traditional (Lloyd)": TraditionalKMeans,
    "K-Means++": KMeansPlusPlus,
    "Elkan K-Means": ElkanKMeans,
    "E-Means (FAIR)": EMeans,
}


class ExperimentRunner:
    """Manages execution, logging, visualization, and result serialization for all experiments."""

    def __init__(self, results_dir: str = "results"):
        self.saver = ResultSaver(results_dir)

    def run_experiment_1_basic_comparison(self) -> pd.DataFrame:
        """Experiment 1: Basic Performance Comparison across diverse datasets."""
        logger.info("=== Running Experiment 1: Basic Performance Comparison ===")
        datasets = [
            ("blobs", *generate_gaussian_blobs(n_samples=1000, n_clusters=4, random_state=42)),
            ("anisotropic", *generate_anisotropic_blobs(n_samples=1000, n_clusters=3, random_state=42)),
            ("iris", *load_dataset_by_name("iris")),
            ("wine", *load_dataset_by_name("wine")),
            ("cancer", *load_dataset_by_name("breast_cancer")),
            ("digits", *load_dataset_by_name("digits")),
        ]

        all_records = []
        for dname, X_raw, y_true, meta in datasets:
            X_norm, _ = preprocess_features(X_raw, method="standard")
            k = meta["n_clusters"]
            logger.info(f"Evaluating dataset '{dname}' (n={X_norm.shape[0]}, d={X_norm.shape[1]}, k={k})")

            clusters_2d_dict = {}
            X_2d, _ = reduce_to_2d_for_visualization(X_norm)

            for algo_name, algo_cls in ALGORITHMS.items():
                rec, model, _ = run_single_benchmark(
                    algo_cls=algo_cls,
                    algo_name=algo_name,
                    X=X_norm,
                    n_clusters=k,
                    dataset_name=dname,
                    y_true=y_true,
                    seed=42,
                )
                all_records.append(rec)

                # Centroids projected to 2D
                if X_norm.shape[1] == 2:
                    centroids_2d = model.centroids_
                else:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2, random_state=42)
                    pca.fit(X_norm)
                    centroids_2d = pca.transform(model.centroids_)
                clusters_2d_dict[algo_name] = (model.labels_, centroids_2d)

            # Generate side-by-side clustering plot
            plot_side_by_side_clusters(
                X_2d=X_2d,
                results_dict=clusters_2d_dict,
                dataset_name=dname.capitalize(),
                save_path=self.saver.get_plot_path("clustering", f"clusters_{dname}"),
            )

        df = pd.DataFrame(all_records)
        self.saver.save_raw_results(df, "experiment_results.csv")

        # Plot performance comparison for blobs
        df_blobs = df[df["dataset"] == "blobs"]
        plot_execution_time_comparison(
            df_blobs,
            title="Execution Time Comparison (Blobs Dataset)",
            save_path=self.saver.get_plot_path("performance", "exec_time_blobs"),
        )
        plot_metric_bars(
            df_blobs,
            metrics=["silhouette", "davies_bouldin"],
            title="Clustering Quality (Blobs Dataset)",
            save_path=self.saver.get_plot_path("performance", "metrics_blobs"),
        )
        return df

    def run_experiment_2_convergence(self) -> pd.DataFrame:
        """Experiment 2: Convergence Comparison (Inertia reduction over iterations and time)."""
        logger.info("=== Running Experiment 2: Convergence Comparison ===")
        X_raw, _, _ = generate_gaussian_blobs(n_samples=5000, n_clusters=5, random_state=42)
        X_norm, _ = preprocess_features(X_raw)

        trajectories = []
        for algo_name, algo_cls in ALGORITHMS.items():
            _, _, traj_df = run_single_benchmark(
                algo_cls=algo_cls,
                algo_name=algo_name,
                X=X_norm,
                n_clusters=5,
                dataset_name="convergence_blobs",
                seed=42,
            )
            trajectories.append(traj_df)

        df_conv = pd.concat(trajectories, ignore_index=True)
        self.saver.save_raw_results(df_conv, "convergence_history.csv")

        plot_inertia_vs_iterations(
            df_conv,
            title="Convergence: Inertia vs Iterations / Epochs",
            save_path=self.saver.get_plot_path("convergence", "inertia_vs_iterations"),
        )
        plot_inertia_vs_time(
            df_conv,
            title="Convergence: Inertia vs Elapsed Wall-Clock Time",
            save_path=self.saver.get_plot_path("convergence", "inertia_vs_time"),
        )
        return df_conv

    def run_experiment_3_initialization_sensitivity(self) -> pd.DataFrame:
        """Experiment 3: Initialization Sensitivity across multiple random seeds."""
        logger.info("=== Running Experiment 3: Initialization Sensitivity ===")
        X_raw, y_true, _ = generate_gaussian_blobs(n_samples=2000, n_clusters=4, cluster_std=1.8, random_state=42)
        X_norm, _ = preprocess_features(X_raw)

        records = []
        for seed in BENCHMARK_SEEDS:
            for algo_name, algo_cls in ALGORITHMS.items():
                rec, _, _ = run_single_benchmark(
                    algo_cls=algo_cls,
                    algo_name=algo_name,
                    X=X_norm,
                    n_clusters=4,
                    dataset_name="init_sensitivity",
                    y_true=y_true,
                    seed=seed,
                )
                records.append(rec)

        df_init = pd.DataFrame(records)
        self.saver.save_raw_results(df_init, "initialization_sensitivity.csv")

        plot_boxplots_by_metric(
            df_init,
            metric_col="inertia",
            title="Inertia Stability Across 10 Initialization Seeds",
            save_path=self.saver.get_plot_path("statistical", "init_sensitivity_inertia_boxplot"),
        )
        if "adjusted_rand_index" in df_init.columns:
            plot_boxplots_by_metric(
                df_init,
                metric_col="adjusted_rand_index",
                title="Adjusted Rand Index Stability Across 10 Seeds",
                save_path=self.saver.get_plot_path("statistical", "init_sensitivity_ari_boxplot"),
            )
        return df_init

    def run_experiment_4_scalability(self) -> Dict[str, pd.DataFrame]:
        """Experiments 4, 5, 6: Scalability across sample size, dimension, and cluster count."""
        logger.info("=== Running Experiments 4, 5, 6: Scalability Suite ===")
        df_n = run_sample_size_scalability(sample_sizes=SAMPLE_SIZES)
        df_d = run_dimensionality_scalability(dimensions=DIMENSION_SIZES)
        df_k = run_cluster_count_scalability(cluster_counts=CLUSTER_COUNTS)

        df_all = pd.concat([df_n, df_d, df_k], ignore_index=True)
        self.saver.save_raw_results(df_all, "scalability_results.csv")

        # Plots
        plot_scalability_curve(
            df_n,
            param_col="n_samples",
            metric_col="total_time_sec",
            param_label="Sample Size (n)",
            metric_label="Runtime (seconds)",
            title="Sample Size Scalability (n vs Runtime)",
            save_path=self.saver.get_plot_path("scalability", "scalability_sample_size"),
        )
        plot_scalability_curve(
            df_n,
            param_col="n_samples",
            metric_col="peak_memory_mb",
            param_label="Sample Size (n)",
            metric_label="Peak Memory (MB)",
            title="Sample Size vs Memory Footprint",
            save_path=self.saver.get_plot_path("scalability", "scalability_memory"),
        )
        plot_scalability_curve(
            df_d,
            param_col="n_features",
            metric_col="total_time_sec",
            param_label="Feature Dimensions (d)",
            metric_label="Runtime (seconds)",
            title="Dimensionality Scalability (d vs Runtime)",
            save_path=self.saver.get_plot_path("scalability", "scalability_dimensions"),
        )
        plot_scalability_curve(
            df_k,
            param_col="n_clusters",
            metric_col="total_time_sec",
            param_label="Number of Clusters (k)",
            metric_label="Runtime (seconds)",
            title="Cluster Count Scalability (k vs Runtime)",
            save_path=self.saver.get_plot_path("scalability", "scalability_clusters"),
        )
        return {"sample_size": df_n, "dimension": df_d, "cluster_count": df_k}

    def run_experiment_7_noise_overlap(self) -> pd.DataFrame:
        """Experiment 7: Noise and Overlapping Clusters."""
        logger.info("=== Running Experiment 7: Noise & Cluster Overlap ===")
        records = []
        for std in NOISE_STD_LEVELS:
            X_raw, y_true, _ = generate_gaussian_blobs(
                n_samples=2000, n_clusters=4, cluster_std=std, random_state=42
            )
            X_norm, _ = preprocess_features(X_raw)

            for algo_name, algo_cls in ALGORITHMS.items():
                rec, _, _ = run_single_benchmark(
                    algo_cls=algo_cls,
                    algo_name=algo_name,
                    X=X_norm,
                    n_clusters=4,
                    dataset_name=f"overlap_std_{std:.1f}",
                    y_true=y_true,
                    seed=42,
                )
                rec["cluster_std"] = std
                records.append(rec)

        df_noise = pd.DataFrame(records)
        self.saver.save_raw_results(df_noise, "noise_overlap_results.csv")

        plot_scalability_curve(
            df_noise,
            param_col="cluster_std",
            metric_col="silhouette",
            param_label="Cluster Standard Deviation (Overlap)",
            metric_label="Silhouette Score",
            title="Silhouette Degradation Under Increasing Cluster Overlap",
            save_path=self.saver.get_plot_path("performance", "noise_overlap_silhouette"),
        )
        return df_noise

    def run_experiment_8_repeated_trials(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Experiment 8: Repeated Statistical Trials with statistical hypothesis testing."""
        logger.info("=== Running Experiment 8: Repeated Statistical Trials ===")
        # Run 10 seeds on real-world Wine and Digits
        benchmarks = [
            ("wine", *load_dataset_by_name("wine")),
            ("digits", *load_dataset_by_name("digits")),
        ]

        repeated_records = []
        for dname, X_raw, y_true, meta in benchmarks:
            X_norm, _ = preprocess_features(X_raw)
            k = meta["n_clusters"]

            for seed in BENCHMARK_SEEDS:
                for algo_name, algo_cls in ALGORITHMS.items():
                    rec, _, _ = run_single_benchmark(
                        algo_cls=algo_cls,
                        algo_name=algo_name,
                        X=X_norm,
                        n_clusters=k,
                        dataset_name=dname,
                        y_true=y_true,
                        seed=seed,
                    )
                    repeated_records.append(rec)

        df_runs = pd.DataFrame(repeated_records)
        self.saver.save_raw_results(df_runs, "repeated_runs.csv")

        # Aggregate statistical summaries
        metric_cols = ["inertia", "total_time_sec", "silhouette", "adjusted_rand_index"]
        df_agg = aggregate_experiment_runs(
            df_runs, group_cols=["dataset", "algorithm"], metric_cols=metric_cols
        )
        self.saver.save_processed_results(df_agg, "aggregated_results.csv")

        # Run statistical tests: EMeans vs Lloyd, and EMeans vs KMeans++
        stat_records = []
        for dname in ["wine", "digits"]:
            sub_d = df_runs[df_runs["dataset"] == dname]
            emeans_inertia = sub_d[sub_d["algorithm"] == "E-Means (FAIR)"]["inertia"].values
            lloyd_inertia = sub_d[sub_d["algorithm"] == "Traditional (Lloyd)"]["inertia"].values
            kpp_inertia = sub_d[sub_d["algorithm"] == "K-Means++"]["inertia"].values

            emeans_time = sub_d[sub_d["algorithm"] == "E-Means (FAIR)"]["total_time_sec"].values
            lloyd_time = sub_d[sub_d["algorithm"] == "Traditional (Lloyd)"]["total_time_sec"].values

            t_res_inertia = run_statistical_significance_test(
                emeans_inertia, lloyd_inertia, "E-Means", "Lloyd"
            )
            t_res_inertia["dataset"] = dname
            t_res_inertia["metric"] = "inertia"
            stat_records.append(t_res_inertia)

            t_res_time = run_statistical_significance_test(
                emeans_time, lloyd_time, "E-Means", "Lloyd"
            )
            t_res_time["dataset"] = dname
            t_res_time["metric"] = "total_time_sec"
            stat_records.append(t_res_time)

        df_stats = pd.DataFrame(stat_records)
        self.saver.save_processed_results(df_stats, "statistical_results.csv")

        # Boxplots
        plot_boxplots_by_metric(
            df_runs[df_runs["dataset"] == "wine"],
            metric_col="inertia",
            title="Wine Dataset: Inertia Stability (10 Seeds)",
            save_path=self.saver.get_plot_path("statistical", "wine_inertia_boxplot"),
        )
        plot_boxplots_by_metric(
            df_runs[df_runs["dataset"] == "digits"],
            metric_col="total_time_sec",
            title="Digits Dataset: Runtime Distribution (10 Seeds)",
            save_path=self.saver.get_plot_path("statistical", "digits_runtime_boxplot"),
        )

        return df_runs, df_agg, df_stats

    def run_all(self):
        """Execute full experimental benchmarking pipeline."""
        logger.info("==================================================")
        logger.info("STARTING FULL EXPERIMENTAL BENCHMARKING SUITE")
        logger.info("==================================================")

        self.run_experiment_1_basic_comparison()
        self.run_experiment_2_convergence()
        self.run_experiment_3_initialization_sensitivity()
        self.run_experiment_4_scalability()
        self.run_experiment_7_noise_overlap()
        self.run_experiment_8_repeated_trials()

        logger.info("==================================================")
        logger.info("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        logger.info("==================================================")
