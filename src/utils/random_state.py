"""Deterministic random state management utilities."""

from typing import Optional, Union
import numpy as np


def check_random_state(seed: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None) -> np.random.RandomState:
    """Ensure a deterministic numpy RandomState instance.

    Parameters
    ----------
    seed : int, RandomState, or None
        Seed or existing RandomState instance.

    Returns
    -------
    np.random.RandomState
        Deterministic RandomState instance.
    """
    if seed is None:
        return np.random.RandomState(42)
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(int(seed))
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f"Invalid random state seed: {seed}")
