"""Ligandra — a target-agnostic, config-driven CADD platform.

Layers: ``data -> featurize -> predict -> generate -> score -> rank``.  Every
layer is a registry of plugins behind a small ABC, so new models/generators are
one-class additions.  See the README for the architecture overview.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Quiet RDKit's very chatty C++ logger by default (parse warnings, etc.).
try:  # pragma: no cover
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

from ligandra.core import (  # noqa: E402
    Molecule,
    MoleculeSet,
    Registry,
    set_global_seed,
)

__all__ = ["Molecule", "MoleculeSet", "Registry", "set_global_seed", "__version__"]
