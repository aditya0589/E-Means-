"""Utility to generate and serialize all synthetic and real-world datasets to disk."""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from .loader import load_dataset_by_name
from .synthetic import (
    generate_gaussian_blobs,
    generate_anisotropic_blobs,
    generate_moons,
    generate_circles,
    generate_noisy_clusters,
    generate_high_dimensional_dataset,
)
from .preprocessing import preprocess_features
from ..utils.logging_config import get_logger

logger = get_logger("save_datasets")


def generate_and_save_all_datasets(data_dir: str = "data") -> List[Dict[str, Any]]:
    """Generate, standardize, and serialize all benchmark datasets to data/raw and data/processed."""
    base_dir = Path(data_dir)
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    catalog: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. Real-World Datasets
    # -------------------------------------------------------------------------
    real_world_names = ["iris", "wine", "breast_cancer", "digits"]
    logger.info("Extracting and saving real-world benchmark datasets...")

    for name in real_world_names:
        X_raw, y_true, meta = load_dataset_by_name(name)
        feat_cols = meta.get("feature_names", [f"feat_{i}" for i in range(X_raw.shape[1])])

        # Raw dataframe
        df_raw = pd.DataFrame(X_raw, columns=feat_cols)
        df_raw["target"] = y_true
        raw_path = raw_dir / f"{meta['name']}.csv"
        df_raw.to_csv(raw_path, index=False)

        # Standardized dataframe
        X_std, _ = preprocess_features(X_raw, method="standard")
        df_std = pd.DataFrame(X_std, columns=feat_cols)
        df_std["target"] = y_true
        processed_path = processed_dir / f"{meta['name']}_standardized.csv"
        df_std.to_csv(processed_path, index=False)

        entry = {
            "name": meta["name"],
            "type": "real_world",
            "n_samples": int(X_raw.shape[0]),
            "n_features": int(X_raw.shape[1]),
            "n_clusters": int(meta["n_clusters"]),
            "raw_file": str(raw_path.as_posix()),
            "processed_file": str(processed_path.as_posix()),
        }
        catalog.append(entry)
        logger.info(f"  Saved {name} (n={entry['n_samples']}, d={entry['n_features']}, k={entry['n_clusters']})")

    # -------------------------------------------------------------------------
    # 2. Synthetic Benchmark Datasets
    # -------------------------------------------------------------------------
    logger.info("Generating and saving synthetic benchmark datasets...")
    synthetic_generators = [
        ("blobs_1000", *generate_gaussian_blobs(n_samples=1000, n_clusters=4, random_state=42)),
        ("anisotropic_1000", *generate_anisotropic_blobs(n_samples=1000, n_clusters=3, random_state=42)),
        ("moons_1000", *generate_moons(n_samples=1000, noise=0.08, random_state=42)),
        ("circles_1000", *generate_circles(n_samples=1000, noise=0.05, random_state=42)),
        ("noisy_1000", *generate_noisy_clusters(n_samples=1000, n_clusters=4, noise_ratio=0.1, random_state=42)),
        ("high_dim_50d_5000", *generate_high_dimensional_dataset(n_samples=5000, n_features=50, n_clusters=5, random_state=42)),
    ]

    # Additional scalability sizes for blobs
    for n in [500, 5000, 10000, 50000]:
        synthetic_generators.append(
            (f"blobs_{n}", *generate_gaussian_blobs(n_samples=n, n_features=10, n_clusters=5, random_state=42))
        )

    for name, X_raw, y_true, meta in synthetic_generators:
        feat_cols = [f"feat_{i}" for i in range(X_raw.shape[1])]

        # Raw dataframe
        df_raw = pd.DataFrame(X_raw, columns=feat_cols)
        df_raw["target"] = y_true
        raw_path = raw_dir / f"{name}.csv"
        df_raw.to_csv(raw_path, index=False)

        # Standardized dataframe
        X_std, _ = preprocess_features(X_raw, method="standard")
        df_std = pd.DataFrame(X_std, columns=feat_cols)
        df_std["target"] = y_true
        processed_path = processed_dir / f"{name}_standardized.csv"
        df_std.to_csv(processed_path, index=False)

        entry = {
            "name": name,
            "type": "synthetic",
            "n_samples": int(X_raw.shape[0]),
            "n_features": int(X_raw.shape[1]),
            "n_clusters": int(meta["n_clusters"]),
            "raw_file": str(raw_path.as_posix()),
            "processed_file": str(processed_path.as_posix()),
        }
        catalog.append(entry)
        logger.info(f"  Saved {name} (n={entry['n_samples']}, d={entry['n_features']}, k={entry['n_clusters']})")

    # -------------------------------------------------------------------------
    # 3. Save Metadata Catalog
    # -------------------------------------------------------------------------
    df_catalog = pd.DataFrame(catalog)
    catalog_csv = base_dir / "metadata.csv"
    catalog_json = base_dir / "dataset_catalog.json"

    df_catalog.to_csv(catalog_csv, index=False)
    with open(catalog_json, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    logger.info(f"Dataset catalog saved to {catalog_csv} and {catalog_json}")
    logger.info(f"Successfully generated and serialized {len(catalog)} datasets!")
    return catalog


if __name__ == "__main__":
    generate_and_save_all_datasets()
