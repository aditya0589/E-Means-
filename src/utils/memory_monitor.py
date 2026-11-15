"""Memory profiling utility using psutil."""

import os
import psutil
from typing import Tuple


class MemoryTracker:
    """Context manager and tracker for monitoring peak RSS memory consumption in megabytes."""

    def __init__(self):
        self._process = psutil.Process(os.getpid())
        self._initial_memory: float = 0.0
        self._peak_memory: float = 0.0
        self._memory_delta: float = 0.0

    def start(self) -> float:
        """Start memory tracking session."""
        self._initial_memory = self._get_rss_mb()
        self._peak_memory = self._initial_memory
        return self._initial_memory

    def stop(self) -> Tuple[float, float]:
        """Stop memory tracking and return (peak_mb, delta_mb)."""
        current = self._get_rss_mb()
        self._peak_memory = max(self._peak_memory, current)
        self._memory_delta = max(0.0, self._peak_memory - self._initial_memory)
        return self._peak_memory, self._memory_delta

    def _get_rss_mb(self) -> float:
        """Current resident set size in megabytes."""
        return self._process.memory_info().rss / (1024.0 * 1024.0)

    @property
    def peak_memory_mb(self) -> float:
        return self._peak_memory

    @property
    def memory_delta_mb(self) -> float:
        return self._memory_delta

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
