"""Convergence trajectory analysis and comparison."""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


def extract_convergence_trajectory(model: Any, model_name: str, dataset_name: str) -> pd.DataFrame:
    """Convert model history dictionary to a structured pandas DataFrame."""
    history: Dict[str, list] = getattr(model, "history_", {})
    if not history or "iteration" not in history or len(history["iteration"]) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(history)
    df["algorithm"] = model_name
    df["dataset"] = dataset_name
    return df
