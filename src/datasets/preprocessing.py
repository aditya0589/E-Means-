"""Data preprocessing utilities for clustering workflows."""

from typing import Tuple, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def preprocess_features(
    X: np.ndarray,
    method: str = "standard",
) -> Tuple[np.ndarray, object]:
    """Scale input features using StandardScaler or MinMaxScaler.

    Parameters
    ----------
    X : np.ndarray
        Raw numerical data matrix of shape (n_samples, n_features).
    method : str, default='standard'
        Scaling method ('standard' or 'minmax' or 'none').

    Returns
    -------
    X_scaled : np.ndarray
        Preprocessed feature matrix.
    scaler : object
        Fitted scaler instance.
    """
    X_clean = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    if method == "standard":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
    elif method == "minmax":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_clean)
    elif method == "none":
        scaler = None
        X_scaled = X_clean
    else:
        raise ValueError(f"Unknown scaling method '{method}'. Choose 'standard', 'minmax', or 'none'.")

    return X_scaled, scaler


def reduce_to_2d_for_visualization(
    X: np.ndarray,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, Optional[PCA]]:
    """Project data to 2D for plotting via PCA if d > 2.

    If d == 2, returns original coordinates.
    """
    if X.shape[1] == 2:
        return X.copy(), None

    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X)
    return X_2d, pca
