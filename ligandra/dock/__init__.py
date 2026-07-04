"""Structure-based docking layer (optional, M8).

``DockingEngine`` ABC + ``DOCKERS`` registry (in :mod:`ligandra.dock.base`).
Importing this package registers the built-in engines.  A docking score can feed
the scoring layer (M7) and condition the diffusion generator (M6).
"""

from __future__ import annotations

# Import concrete engines so their @register decorators run.
from ligandra.dock import vina as _vina  # noqa: E402,F401
from ligandra.dock.base import DOCKERS, DockingEngine, DockingResult

__all__ = ["DOCKERS", "DockingEngine", "DockingResult"]
