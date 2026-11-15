"""Performance profiling and extraction utilities."""

from typing import Dict, Any


def extract_performance_profile(model: Any) -> Dict[str, Any]:
    """Extract computational execution metrics from a fitted model instance."""
    return {
        "init_time_sec": float(getattr(model, "init_time_", 0.0)),
        "train_time_sec": float(getattr(model, "train_time_", 0.0)),
        "total_time_sec": float(getattr(model, "total_time_", 0.0)),
        "n_iter": int(getattr(model, "n_iter_", 0)),
        "peak_memory_mb": float(getattr(model, "peak_memory_mb_", 0.0)),
        "memory_delta_mb": float(getattr(model, "memory_delta_mb_", 0.0)),
        "distance_calculations": int(getattr(model, "distance_calculations_", 0)),
    }
