"""Main CLI entry point for the E-Means vs Traditional K-Means research suite.

Usage:
    python -m src.main --experiment all
    python -m src.main --experiment 1
    python -m src.main --report
"""

import argparse
import sys
from pathlib import Path
from tabulate import tabulate

from .experiments.experiment_runner import ExperimentRunner
from .report_generator import generate_research_pdf_report
from .utils.logging_config import get_logger

logger = get_logger("main")


def main():
    parser = argparse.ArgumentParser(
        description="E-Means vs Traditional K-Means vs Optimized K-Means Research Suite"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "4", "7", "8", "smoke"],
        help="Specify which experiment to execute (default: 'all')",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate comprehensive PDF research report after experiments",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Output directory for results and plots (default: 'results')",
    )

    args = parser.parse_args()
    runner = ExperimentRunner(results_dir=args.results_dir)

    logger.info("Initializing E-Means Research Benchmarking Framework")

    if args.experiment == "smoke":
        logger.info("Running Smoke Test...")
        from sklearn.datasets import make_blobs
        from .experiments.benchmark import run_single_benchmark
        from .experiments.experiment_runner import ALGORITHMS

        X, y = make_blobs(n_samples=200, n_features=2, centers=3, random_state=42)
        smoke_rows = []
        for name, cls in ALGORITHMS.items():
            rec, _, _ = run_single_benchmark(
                algo_cls=cls,
                algo_name=name,
                X=X,
                n_clusters=3,
                dataset_name="smoke_test",
                y_true=y,
                seed=42,
            )
            smoke_rows.append([
                name,
                f"{rec['inertia']:.2f}",
                f"{rec['total_time_sec']:.4f}s",
                rec['n_iter'],
                rec['distance_calculations'],
            ])
        print(tabulate(smoke_rows, headers=["Algorithm", "Inertia", "Runtime", "Iter/Epoch", "Dist Calcs"], tablefmt="grid"))
        logger.info("Smoke test passed successfully!")
        return

    if args.experiment == "1":
        runner.run_experiment_1_basic_comparison()
    elif args.experiment == "2":
        runner.run_experiment_2_convergence()
    elif args.experiment == "3":
        runner.run_experiment_3_initialization_sensitivity()
    elif args.experiment == "4":
        runner.run_experiment_4_scalability()
    elif args.experiment == "7":
        runner.run_experiment_7_noise_overlap()
    elif args.experiment == "8":
        runner.run_experiment_8_repeated_trials()
    elif args.experiment == "all":
        runner.run_all()

    # Generate PDF report
    if args.report or args.experiment == "all":
        logger.info("Generating automated PDF research report...")
        pdf_path = generate_research_pdf_report(results_dir=args.results_dir)
        logger.info(f"Report generated successfully: {pdf_path}")


if __name__ == "__main__":
    main()
