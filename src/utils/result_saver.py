"""Automated persistence utility for experiment tables, raw runs, and aggregated stats."""

import os
from pathlib import Path
from typing import Union
import pandas as pd


class ResultSaver:
    """Manages saving and appending experiment data to CSV and JSON formats."""

    def __init__(self, base_dir: Union[str, Path] = "results"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.plots_dir = self.base_dir / "plots"
        self.reports_dir = self.base_dir / "reports"

        for directory in [self.raw_dir, self.processed_dir, self.plots_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def save_raw_results(self, df: pd.DataFrame, filename: str = "experiment_results.csv", append: bool = True) -> Path:
        """Save raw experiment rows without destructive overwrite."""
        filepath = self.raw_dir / filename
        if append and filepath.exists():
            df_existing = pd.read_csv(filepath)
            df_combined = pd.concat([df_existing, df], ignore_index=True)
            df_combined.drop_duplicates(inplace=True)
            df_combined.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
        return filepath

    def save_processed_results(self, df: pd.DataFrame, filename: str = "aggregated_results.csv") -> Path:
        """Save aggregated statistical summaries."""
        filepath = self.processed_dir / filename
        df.to_csv(filepath, index=False)
        return filepath

    def get_plot_path(self, category: str, filename: str) -> Path:
        """Generate full path for a plot and ensure subdirectory exists."""
        cat_dir = self.plots_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        return cat_dir / filename
