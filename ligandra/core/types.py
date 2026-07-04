"""Shared enums and lightweight dataclasses used across layers.

Kept dependency-free (stdlib only) so the whole package imports on a bare
interpreter and the type contracts are always available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Endpoint(str, Enum):
    """Bioactivity endpoints the platform understands.

    ``is_concentration`` marks endpoints measured as a molar concentration that
    should be converted to a p-value (``pXC50 = -log10(mol/L)``).  Percent
    endpoints are used directly.
    """

    IC50 = "IC50"
    EC50 = "EC50"
    KI = "Ki"
    KD = "Kd"
    PERCENT_INHIBITION = "%inhibition"

    @property
    def is_concentration(self) -> bool:
        return self in {Endpoint.IC50, Endpoint.EC50, Endpoint.KI, Endpoint.KD}

    @property
    def p_name(self) -> str:
        """Name of the log-transformed column, e.g. ``pIC50`` / ``pKi``."""
        mapping = {
            Endpoint.IC50: "pIC50",
            Endpoint.EC50: "pEC50",
            Endpoint.KI: "pKi",
            Endpoint.KD: "pKd",
            Endpoint.PERCENT_INHIBITION: "percent_inhibition",
        }
        return mapping[self]


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class SplitStrategy(str, Enum):
    RANDOM = "random"
    SCAFFOLD = "scaffold"


class ActivityLabel(str, Enum):
    ACTIVE = "active"
    INTERMEDIATE = "intermediate"
    INACTIVE = "inactive"
    UNDEFINED = "undefined"


# --- Normalized column names emitted by every DataSource -----------------
class Column:
    MOLECULE_ID = "molecule_id"
    SMILES = "smiles"
    TARGET_ID = "target_id"
    ENDPOINT = "endpoint"
    VALUE = "value"
    UNITS = "units"
    RELATION = "relation"
    ASSAY_METADATA = "assay_metadata"

    #: minimal schema required from a raw fetch
    RAW_SCHEMA = (MOLECULE_ID, SMILES, TARGET_ID, ENDPOINT, VALUE, UNITS)


@dataclass(frozen=True)
class Target:
    """A protein target returned by ``DataSource.search_targets``."""

    target_id: str
    name: str
    organism: str | None = None
    target_type: str | None = None
    source: str = "unknown"
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Objective:
    """One term in a multi-objective scoring function.

    ``weight`` is used by the weighted-sum aggregator; ``minimize`` flips the
    desirability so that lower raw scores are better (handled by the scorer).
    """

    scorer: str
    weight: float = 1.0
    minimize: bool = False
    params: dict = field(default_factory=dict)
