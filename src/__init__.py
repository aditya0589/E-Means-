"""E-Means vs Traditional K-Means Research Package."""

import os
import tempfile

# Ensure Matplotlib uses a writable cache directory
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_emeans"))

__version__ = "1.0.0"
