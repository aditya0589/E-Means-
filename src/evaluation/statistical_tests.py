"""Statistical testing and aggregation for repeated trials."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats


def aggregate_experiment_runs(df_runs: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    """Aggregate repeated experiment runs computing mean, std, median, and 95% CI."""
    agg_funcs = {
        metric: ["mean", "std", "median", "count"] for metric in metric_cols if metric in df_runs.columns
    }
    grouped = df_runs.groupby(group_cols).agg(agg_funcs)

    # Flatten multi-index columns
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index()

    # Calculate 95% confidence intervals: 1.96 * std / sqrt(n)
    for metric in metric_cols:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"
        ci_col = f"{metric}_ci95"

        if mean_col in grouped.columns and std_col in grouped.columns:
            n = grouped[count_col]
            std = grouped[std_col].fillna(0.0)
            ci = 1.96 * (std / np.sqrt(np.maximum(1, n)))
            grouped[ci_col] = ci

    return grouped


def run_statistical_significance_test(
    series_a: np.ndarray,
    series_b: np.ndarray,
    name_a: str = "Algorithm A",
    name_b: str = "Algorithm B",
) -> Dict[str, Any]:
    """Perform paired t-test and Wilcoxon signed-rank test between two paired result series."""
    a = np.asarray(series_a, dtype=np.float64)
    b = np.asarray(series_b, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError("Series A and B must have identical sample length for paired tests.")

    diff = a - b
    mean_diff = float(np.mean(diff))

    result = {
        "comparison": f"{name_a} vs {name_b}",
        "n_samples": len(a),
        "mean_diff": mean_diff,
    }

    # 1. Paired Student's t-test
    if np.all(diff == 0.0):
        result["t_stat"] = 0.0
        result["p_val_ttest"] = 1.0
    else:
        t_stat, p_ttest = stats.ttest_rel(a, b)
        result["t_stat"] = float(t_stat)
        result["p_val_ttest"] = float(p_ttest)

    # 2. Wilcoxon signed-rank test
    try:
        if np.all(diff == 0.0):
            result["wilcoxon_stat"] = 0.0
            result["p_val_wilcoxon"] = 1.0
        else:
            w_stat, p_wilc = stats.wilcoxon(a, b)
            result["wilcoxon_stat"] = float(w_stat)
            result["p_val_wilcoxon"] = float(p_wilc)
    except Exception as exc:
        result["wilcoxon_stat"] = float("nan")
        result["p_val_wilcoxon"] = float("nan")

    return result
