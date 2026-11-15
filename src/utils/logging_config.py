"""Standardized logging configuration for research experimentation."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "emeans_research", level: int = logging.INFO) -> logging.Logger:
    """Retrieve a configured logger instance with formatted console output.

    Parameters
    ----------
    name : str
        Logger name identifier.
    level : int
        Logging level (default: logging.INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
