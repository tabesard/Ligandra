"""AutoDock Vina wrapper — score parsing tested offline; clear error when absent."""

from __future__ import annotations

import pytest

from ligandra.core.molecule import MoleculeSet
from ligandra.dock import DOCKERS
from ligandra.dock.base import DockingResult
from ligandra.dock.vina import VinaEngine, VinaNotInstalledError, best_score, parse_vina_scores

_POSE_PDBQT = """MODEL 1
REMARK VINA RESULT:    -8.3      0.000      0.000
ATOM      1  C   LIG     1
ENDMDL
MODEL 2
REMARK VINA RESULT:    -7.1      1.234      2.345
ENDMDL
"""

_STDOUT_TABLE = """
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -9.2        0.000      0.000
   2       -8.0        1.100      2.200
"""


def test_vina_registered():
    assert "vina" in DOCKERS.available()


def test_parse_scores_from_pose_remarks():
    assert parse_vina_scores(_POSE_PDBQT) == [-8.3, -7.1]
    assert best_score(_POSE_PDBQT) == -8.3


def test_parse_scores_from_stdout_table():
    assert parse_vina_scores(_STDOUT_TABLE) == [-9.2, -8.0]
    assert best_score(_STDOUT_TABLE) == -9.2


def test_best_score_none_when_unparseable():
    assert best_score("no docking here") is None


def test_dock_raises_clear_error_without_binary(tmp_path):
    receptor = tmp_path / "rec.pdbqt"
    receptor.write_text("REMARK receptor", encoding="utf-8")
    engine = VinaEngine(vina_binary="vina_definitely_not_installed_xyz")
    with pytest.raises(VinaNotInstalledError):
        engine.dock(MoleculeSet.from_smiles(["CCO"]), str(receptor))


def test_docking_result_dataclass():
    r = DockingResult(score=-7.5, pose_block="MODEL")
    assert r.score == -7.5 and r.pose_block == "MODEL"
