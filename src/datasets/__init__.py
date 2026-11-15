"""Datasets module containing synthetic generators, real-world loaders, and preprocessing."""

from .synthetic import (
    generate_gaussian_blobs,
    generate_anisotropic_blobs,
    generate_moons,
    generate_circles,
    generate_noisy_clusters,
    generate_high_dimensional_dataset,
)
from .loader import load_dataset_by_name
from .preprocessing import preprocess_features, reduce_to_2d_for_visualization
from .save_datasets import generate_and_save_all_datasets

__all__ = [
    "generate_gaussian_blobs",
    "generate_anisotropic_blobs",
    "generate_moons",
    "generate_circles",
    "generate_noisy_clusters",
    "generate_high_dimensional_dataset",
    "load_dataset_by_name",
    "preprocess_features",
    "reduce_to_2d_for_visualization",
    "generate_and_save_all_datasets",
]
