"""AutoDock Vina docking back end (real implementation).

Prepares 3D ligands from RDKit, converts them to PDBQT (Meeko preferred, Open
Babel fallback), runs the ``vina`` executable against a prepared receptor, and
parses the binding affinity from Vina's output.  A clear, actionable error is
raised when the external ``vina`` binary or a ligand-prep tool is not installed —
so the wrapper is structurally complete and its parsing is unit-tested offline,
even though a live dock needs those external assets.

Docking scores (kcal/mol, lower = better) can feed the scoring layer (M7) and
condition the structure-based diffusion generator (M6/M8).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from ligandra.core.molecule import Molecule, MoleculeSet
from ligandra.dock.base import DOCKERS, DockingEngine, DockingResult


class VinaNotInstalledError(RuntimeError):
    """Raised when the external ``vina`` binary (or a prep tool) is missing."""


def parse_vina_scores(output: str) -> list[float]:
    """Extract per-mode affinities (kcal/mol) from Vina output.

    Handles both the machine-readable pose file (``REMARK VINA RESULT: <aff> ...``)
    and Vina's stdout results table::

        mode |   affinity | dist from best mode
             | (kcal/mol) | rmsd l.b.| rmsd u.b.
        -----+------------+----------+----------
           1       -7.5        0.000      0.000

    Returns the scores in mode order (best first, as Vina emits them).
    """
    scores: list[float] = []
    # 1) REMARK lines from an output PDBQT (most reliable).
    for line in output.splitlines():
        if "VINA RESULT" in line:
            tail = line.split(":", 1)[-1].split()
            if tail:
                try:
                    scores.append(float(tail[0]))
                except ValueError:
                    pass
    if scores:
        return scores
    # 2) Fall back to the stdout results table.
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                scores.append(float(parts[1]))
            except ValueError:
                continue
    return scores


def best_score(output: str) -> float | None:
    """Best (lowest) affinity in Vina output, or ``None`` if unparseable."""
    scores = parse_vina_scores(output)
    return min(scores) if scores else None


@DOCKERS.register("vina")
class VinaEngine(DockingEngine):
    """AutoDock Vina back end.

    Parameters
    ----------
    center, box_size:
        Search-box center and size (Å).  Required for a real dock; pull from the
        PDB data source / a pocket-detection step (M8).
    exhaustiveness, num_modes, seed:
        Standard Vina search controls.
    vina_binary:
        Name/path of the Vina executable (default ``"vina"``; ``qvina2`` etc.
        also work).
    """

    def __init__(
        self,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
        exhaustiveness: int = 8,
        num_modes: int = 9,
        seed: int = 42,
        vina_binary: str = "vina",
    ) -> None:
        self.center = center
        self.box_size = box_size
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.seed = seed
        self.vina_binary = vina_binary

    # -- availability ----------------------------------------------------
    def _resolve_binary(self) -> str:
        path = shutil.which(self.vina_binary)
        if path is None:
            raise VinaNotInstalledError(
                f"AutoDock Vina executable {self.vina_binary!r} not found on PATH. "
                "Install it from https://vina.scripps.edu/ (or `conda install -c "
                "conda-forge vina`) to run docking."
            )
        return path

    # -- ligand prep -----------------------------------------------------
    def _ligand_to_pdbqt(self, mol: Molecule, path: Path) -> None:
        """Embed a 3D conformer and write a PDBQT for ``mol``."""
        rdmol = mol.embed_3d(seed=self.seed)  # ETKDG + MMFF, adds Hs
        # Preferred: Meeko (RDKit-native, no external process).
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy

            prep = MoleculePreparation()
            setup = prep.prepare(rdmol)[0]
            pdbqt, ok, _ = PDBQTWriterLegacy.write_string(setup)
            if not ok:
                raise RuntimeError("Meeko failed to write PDBQT.")
            path.write_text(pdbqt, encoding="utf-8")
            return
        except ImportError:
            pass
        # Fallback: Open Babel CLI.
        obabel = shutil.which("obabel")
        if obabel is None:
            raise VinaNotInstalledError(
                "No ligand-prep tool found. Install Meeko (`pip install meeko`) "
                "or Open Babel (`conda install -c conda-forge openbabel`) to "
                "convert ligands to PDBQT."
            )
        from rdkit import Chem

        with tempfile.NamedTemporaryFile("w", suffix=".mol", delete=False) as fh:
            fh.write(Chem.MolToMolBlock(rdmol))
            mol_path = fh.name
        subprocess.run(  # noqa: S603
            [obabel, mol_path, "-O", str(path), "--partialcharge", "gasteiger"],
            check=True,
            capture_output=True,
        )

    # -- dock ------------------------------------------------------------
    def dock(self, mols: MoleculeSet, receptor: str, **kwargs) -> list[DockingResult]:
        """Dock each molecule into ``receptor`` (a prepared ``.pdbqt``)."""
        binary = self._resolve_binary()
        receptor_path = Path(receptor)
        if not receptor_path.exists():
            raise FileNotFoundError(f"Receptor PDBQT not found: {receptor}")

        center = kwargs.get("center", self.center)
        box = kwargs.get("box_size", self.box_size)
        results: list[DockingResult] = []
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            for i, mol in enumerate(mols):
                if not mol.is_valid:
                    results.append(DockingResult(score=float("nan")))
                    continue
                lig = tmpdir / f"lig_{i}.pdbqt"
                out = tmpdir / f"out_{i}.pdbqt"
                self._ligand_to_pdbqt(mol, lig)
                cmd = [
                    binary,
                    "--receptor",
                    str(receptor_path),
                    "--ligand",
                    str(lig),
                    "--out",
                    str(out),
                    "--center_x",
                    str(center[0]),
                    "--center_y",
                    str(center[1]),
                    "--center_z",
                    str(center[2]),
                    "--size_x",
                    str(box[0]),
                    "--size_y",
                    str(box[1]),
                    "--size_z",
                    str(box[2]),
                    "--exhaustiveness",
                    str(self.exhaustiveness),
                    "--num_modes",
                    str(self.num_modes),
                    "--seed",
                    str(self.seed),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
                pose = out.read_text(encoding="utf-8") if out.exists() else ""
                score = best_score(pose) or best_score(proc.stdout)
                results.append(
                    DockingResult(
                        score=score if score is not None else float("nan"),
                        pose_block=pose or None,
                    )
                )
        return results


__all__ = ["VinaEngine", "VinaNotInstalledError", "parse_vina_scores", "best_score"]
