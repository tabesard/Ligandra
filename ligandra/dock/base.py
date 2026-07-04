"""Docking-engine contract and registry (M8).

A :class:`DockingEngine` scores molecules against a receptor pocket.  The score
(kcal/mol, lower = better) can feed the scoring layer (M7) and condition the
structure-based diffusion generator (M6).  AutoDock Vina / gnina / DiffDock back
ends are registered here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ligandra.core.molecule import MoleculeSet
from ligandra.core.registry import Registry

#: Registry of docking engines. Add with ``@DOCKERS.register("name")``.
DOCKERS: Registry[DockingEngine] = Registry("docking engine")


@dataclass
class DockingResult:
    score: float  # kcal/mol (lower is better); NaN if the ligand could not dock
    pose_block: str | None = None  # SDF/PDBQT pose


class DockingEngine(ABC):
    """Dock molecules into a receptor pocket."""

    @abstractmethod
    def dock(self, mols: MoleculeSet, receptor: str, **kwargs) -> list[DockingResult]: ...
