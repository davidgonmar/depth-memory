"""Linear Depth Memory (LDM).

``config`` is the model configuration. ``components`` holds the learned parameter modules of the
reference and flexible backends, ``parameters`` those of the fast and fast-inference backends, and
``backends`` the implementations a run selects by name (see ``backends/BACKENDS.md``).
"""

from .config import LDMConfig

__all__ = ["LDMConfig"]
