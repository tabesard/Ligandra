"""Structure-based 3D diffusion generator (M6 / M8) — DiffSBDD/EDM-style.

A genuine equivariant denoising diffusion model over **all-atom** 3D point
clouds (coordinates + atom types), the paradigm behind EDM / DiffSBDD /
TargetDiff:

* an **E(3)-equivariant GNN (EGNN)** denoiser predicts the noise on atom
  coordinates (equivariantly) and atom-type features (invariantly);
* a **DDPM** forward/reverse process diffuses a molecule's all-atom cloud;
* generation samples an atom cloud from noise and reconstructs a molecule via
  RDKit bond perception (reliable on all-atom geometry, incl. hydrogens);
* **pocket conditioning** (the DiffSBDD idea) is done by adding the receptor's
  atoms as *fixed context nodes* — they shape the denoiser's messages but are
  never noised or moved, so generation is conditioned on the 3D pocket at
  inference (Section 7.3).

The model trains on 3D conformers of the target's actives.  It is a real,
runnable diffusion model; note that *de-novo* validity at drug scale needs
GPU-scale pretraining on protein–ligand complexes (CrossDocked/PDBbind), exactly
as the literature shows — on tiny CPU training it demonstrably runs, trains,
samples, and reconstructs, but yields fewer valid molecules than a pretrained
checkpoint would.  Requires ``torch``.
"""

from __future__ import annotations

import numpy as np

from ligandra.core.molecule import Molecule, MoleculeSet
from ligandra.generate.base import GENERATORS, Generator
from ligandra.score.base import ScoringFunction

#: Elements the model can place (index = atom-type id).
_ELEMENTS = ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
_ELEM_INDEX = {e: i for i, e in enumerate(_ELEMENTS)}


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The diffusion generator needs `torch`. Install with `pip install torch`."
        ) from exc


# --------------------------------------------------------------------------
# Molecule <-> all-atom point cloud
# --------------------------------------------------------------------------
def mol_to_cloud(smiles: str, seed: int = 42):
    """SMILES -> (coords (N,3), type_ids (N,)) for all atoms incl. H, or None."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:  # pragma: no cover - forcefield may be missing
        pass
    conf = mol.GetConformer()
    types = []
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        if sym not in _ELEM_INDEX:
            return None
        types.append(_ELEM_INDEX[sym])
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32
    )
    coords -= coords.mean(axis=0, keepdims=True)  # zero centre of mass
    return coords, np.array(types, dtype=np.int64)


def cloud_to_mol(coords: np.ndarray, type_ids: np.ndarray) -> str | None:
    """All-atom cloud -> canonical SMILES via RDKit bond perception, or None."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    if len(type_ids) == 0:
        return None
    rw = Chem.RWMol()
    for t in type_ids:
        rw.AddAtom(Chem.Atom(_ELEMENTS[int(t)]))
    conf = Chem.Conformer(len(type_ids))
    for i, p in enumerate(coords):
        conf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))
    rw.AddConformer(conf)
    mol = rw.GetMol()
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
    except (ValueError, RuntimeError, Chem.AtomValenceException):
        return None
    if not smi or Chem.MolFromSmiles(smi) is None:
        return None
    # keep the largest connected fragment (parent-fragment curation)
    if "." in smi:
        frags = [f for f in smi.split(".") if Chem.MolFromSmiles(f) is not None]
        if not frags:
            return None
        smi = max(frags, key=lambda f: Chem.MolFromSmiles(f).GetNumAtoms())
        if Chem.MolFromSmiles(smi).GetNumAtoms() < 3:
            return None
    return Chem.CanonSmiles(smi)


def parse_pocket_pdb(pdb_path: str, max_atoms: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Read ATOM/HETATM records from a PDB into (coords, type_ids) context nodes."""
    coords, types = [], []
    with open(pdb_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            elem = line[76:78].strip() or line[12:16].strip()[:1]
            elem = elem.capitalize()
            if elem not in _ELEM_INDEX:
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
            types.append(_ELEM_INDEX[elem])
            if len(coords) >= max_atoms:
                break
    return np.asarray(coords, dtype=np.float32), np.asarray(types, dtype=np.int64)


@GENERATORS.register("diffusion")
class DiffusionGenerator(Generator):
    """All-atom E(3)-equivariant diffusion generator with pocket conditioning."""

    requires_structure = False  # runs unconditionally; a pocket conditions it

    def __init__(
        self,
        seed_smiles: list[str] | None = None,
        pocket: str | None = None,
        hidden: int = 64,
        n_layers: int = 3,
        timesteps: int = 100,
        epochs: int = 100,
        lr: float = 1e-3,
        refine_frac: float = 0.15,
        max_atoms: int = 64,
        pocket_max_atoms: int = 120,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        self.seed_smiles = list(seed_smiles or [])
        self.pocket = pocket
        self.hidden = hidden
        self.n_layers = n_layers
        self.timesteps = timesteps
        self.epochs = epochs
        self.lr = lr
        self.refine_frac = refine_frac
        self.max_atoms = max_atoms
        self.pocket_max_atoms = pocket_max_atoms
        self.device = device
        self.seed = seed
        self._net = None
        self._clouds: list = []
        self._sizes: list[int] = []
        self._pocket_ctx = None  # (coords, type_ids) or None

    def set_seeds(self, smiles: list[str]) -> None:
        self.seed_smiles = list(smiles)

    # -- diffusion schedule ---------------------------------------------
    def _make_schedule(self):
        import torch

        # cosine schedule (Nichol & Dhariwal)
        s, T = 0.008, self.timesteps
        t = torch.linspace(0, T, T + 1)
        f = torch.cos(((t / T) + s) / (1 + s) * (np.pi / 2)) ** 2
        abar = f / f[0]
        betas = (1 - abar[1:] / abar[:-1]).clamp(1e-5, 0.999)
        self._betas = betas
        self._alphas = 1.0 - betas
        self._abar = torch.cumprod(self._alphas, dim=0)

    # -- net -------------------------------------------------------------
    def _build_net(self):
        import torch
        import torch.nn as nn

        K = len(_ELEMENTS)
        H = self.hidden
        n_layers = self.n_layers
        feat_in = K + 2  # type feats + context flag + time

        def mlp(i, o):
            return nn.Sequential(nn.Linear(i, H), nn.SiLU(), nn.Linear(H, o))

        class EGNNLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.edge = mlp(2 * H + 1, H)
                self.coord = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, 1))
                self.node = mlp(2 * H, H)

            def forward(self, h, x, movable):
                n = h.size(0)
                diff = x.unsqueeze(1) - x.unsqueeze(0)  # (n,n,3)
                dist2 = (diff**2).sum(-1, keepdim=True)  # (n,n,1)
                hi = h.unsqueeze(1).expand(n, n, H)
                hj = h.unsqueeze(0).expand(n, n, H)
                m = self.edge(torch.cat([hi, hj, dist2], dim=-1))  # (n,n,H)
                off = (1.0 - torch.eye(n, device=h.device)).unsqueeze(-1)
                m = m * off
                # equivariant coordinate update (normalized relative vectors),
                # averaged over neighbours so magnitude does not grow with n.
                cw = self.coord(m).clamp(-10.0, 10.0)  # (n,n,1)
                # eps inside sqrt: the diagonal has dist2=0 and d/dx sqrt(x)=inf at
                # 0 would make gradients NaN (0*inf); the eps keeps backward finite.
                dist = (dist2 + 1e-8).sqrt()
                dx = (diff / (dist + 1.0) * cw).sum(1) / max(n - 1, 1)  # (n,3)
                x = x + dx * movable.unsqueeze(-1)
                h = h + self.node(torch.cat([h, m.sum(1) / max(n - 1, 1)], dim=-1))
                return h, x

        class EGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(feat_in, H)
                self.layers = nn.ModuleList([EGNNLayer() for _ in range(n_layers)])
                self.type_out = nn.Linear(H, K)

            def forward(self, feats, x, movable):
                h = self.embed(feats)
                x0 = x
                for layer in self.layers:
                    h, x = layer(h, x, movable)
                return self.type_out(h), (x - x0)  # eps_h, eps_x

        return EGNN()

    # -- training --------------------------------------------------------
    def _prepare_data(self):
        self._clouds, self._sizes = [], []
        for i, smi in enumerate(self.seed_smiles):
            cloud = mol_to_cloud(smi, seed=self.seed + i)
            if cloud is None:
                continue
            coords, types = cloud
            if len(types) <= self.max_atoms:
                self._clouds.append((coords, types))
                self._sizes.append(len(types))
        if self.pocket:
            pc, pt = parse_pocket_pdb(self.pocket, self.pocket_max_atoms)
            if len(pt):
                pc = pc - pc.mean(axis=0, keepdims=True)
                self._pocket_ctx = (pc, pt)

    def _ensure_trained(self):
        _require_torch()
        if self._net is not None:
            return
        if not self.seed_smiles:
            raise ValueError("DiffusionGenerator needs seed molecules; call set_seeds().")
        import torch

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._make_schedule()
        self._prepare_data()
        if not self._clouds:
            raise ValueError("No 3D conformers could be built from the seed molecules.")
        self._net = self._build_net().to(self.device)
        self._train()

    def _onehot(self, type_ids):
        import torch

        oh = torch.zeros(len(type_ids), len(_ELEMENTS), device=self.device)
        oh[torch.arange(len(type_ids)), torch.as_tensor(type_ids, device=self.device)] = 1.0
        return oh

    def _context_feats(self):
        """Return (coords, feats, movable) tensors for the fixed pocket nodes."""
        import torch

        if self._pocket_ctx is None:
            return None
        pc, pt = self._pocket_ctx
        x = torch.tensor(pc, device=self.device)
        h0 = self._onehot(pt)
        flag = torch.ones(len(pt), 1, device=self.device)  # context flag = 1
        movable = torch.zeros(len(pt), device=self.device)
        return x, h0, flag, movable

    def _train(self):
        import torch
        import torch.nn.functional as F

        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        ctx = self._context_feats()
        self._net.train()
        for _ in range(self.epochs):
            for coords, types in self._clouds:
                x0 = torch.tensor(coords, device=self.device)
                x0 = x0 - x0.mean(0, keepdim=True)
                h0 = self._onehot(types)
                n = x0.size(0)
                t = int(torch.randint(0, self.timesteps, (1,)))
                abar = self._abar[t]
                eps_x = torch.randn_like(x0)
                eps_x = eps_x - eps_x.mean(0, keepdim=True)  # zero-CoM noise
                eps_h = torch.randn_like(h0)
                x_t = abar.sqrt() * x0 + (1 - abar).sqrt() * eps_x
                h_t = abar.sqrt() * h0 + (1 - abar).sqrt() * eps_h
                tnorm = torch.full((n, 1), t / self.timesteps, device=self.device)
                flag = torch.zeros(n, 1, device=self.device)
                feats = torch.cat([h_t, flag, tnorm], dim=-1)
                x_in, movable = x_t, torch.ones(n, device=self.device)
                if ctx is not None:
                    cx, ch, cflag, cmov = ctx
                    ctnorm = torch.full((cx.size(0), 1), t / self.timesteps, device=self.device)
                    feats = torch.cat([feats, torch.cat([ch, cflag, ctnorm], dim=-1)], dim=0)
                    x_in = torch.cat([x_t, cx], dim=0)
                    movable = torch.cat([movable, cmov], dim=0)
                pred_h, pred_x = self._net(feats, x_in, movable)
                pred_h, pred_x = pred_h[:n], pred_x[:n]
                pred_x = pred_x - pred_x.mean(0, keepdim=True)
                loss = F.mse_loss(pred_x, eps_x) + F.mse_loss(pred_h, eps_h)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()

    # -- sampling --------------------------------------------------------
    def _denoise(self, x, h, from_t: int | None = None):
        """Ancestral DDPM sampling from step ``from_t`` (default: full ``T``)."""
        import torch

        ctx = self._context_feats()
        n = x.size(0)
        T = self.timesteps if from_t is None else from_t
        self._net.eval()
        with torch.no_grad():
            for t in reversed(range(T)):
                abar = self._abar[t]
                alpha = self._alphas[t]
                beta = self._betas[t]
                tnorm = torch.full((n, 1), t / self.timesteps, device=self.device)
                flag = torch.zeros(n, 1, device=self.device)
                feats = torch.cat([h, flag, tnorm], dim=-1)
                x_in, movable = x, torch.ones(n, device=self.device)
                if ctx is not None:
                    cx, ch, cflag, cmov = ctx
                    ctnorm = torch.full((cx.size(0), 1), t / self.timesteps, device=self.device)
                    feats = torch.cat([feats, torch.cat([ch, cflag, ctnorm], dim=-1)], dim=0)
                    x_in = torch.cat([x, cx], dim=0)
                    movable = torch.cat([movable, cmov], dim=0)
                pred_h, pred_x = self._net(feats, x_in, movable)
                pred_h, pred_x = pred_h[:n], pred_x[:n]
                pred_x = pred_x - pred_x.mean(0, keepdim=True)
                coef = (1 - alpha) / (1 - abar).sqrt()
                x = (x - coef * pred_x) / alpha.sqrt()
                h = (h - coef * pred_h) / alpha.sqrt()
                if t > 0:
                    zx = torch.randn_like(x)
                    zx = zx - zx.mean(0, keepdim=True)
                    x = x + beta.sqrt() * zx
                    h = h + beta.sqrt() * torch.randn_like(h)
                x = x - x.mean(0, keepdim=True)
        return x, h

    def _sample_size(self) -> int:
        return int(np.random.choice(self._sizes)) if self._sizes else 10

    def _sample_cloud(self):
        import torch

        n = self._sample_size()
        x = torch.randn(n, 3, device=self.device)
        x = x - x.mean(0, keepdim=True)
        h = torch.randn(n, len(_ELEMENTS), device=self.device)
        x, h = self._denoise(x, h)
        type_ids = h.argmax(dim=-1).cpu().numpy()
        return x.cpu().numpy(), type_ids

    def sample(self, n: int) -> MoleculeSet:
        self._ensure_trained()
        mols = []
        for _ in range(n):
            coords, types = self._sample_cloud()
            smi = cloud_to_mol(coords, types)
            mols.append(Molecule(smi if smi else ""))
        return MoleculeSet(mols)

    # -- optimization via partial-diffusion of the elite set ------------
    def optimize_for_target(self, objective: ScoringFunction, budget: int) -> MoleculeSet:
        """Sample, score, and evolve elites by partial re-diffusion (SDEdit-style)."""
        self._ensure_trained()
        import torch

        collected: dict[str, float] = {}
        # seed the elite set with the training conformers (reconstructable), so
        # refinement explores novel neighbours of known-good molecules from step 1
        elite_clouds: list = [(c.copy(), t.copy()) for c, t in self._clouds[:8]]
        evals = 0
        batch = 16
        # light re-noising keeps elite variants in the reconstructable regime
        sdedit_t = max(1, int(self.timesteps * self.refine_frac))
        while evals < budget:
            clouds = []
            # first round (or top-up) from the prior; later rounds perturb elites
            for _ in range(batch):
                if elite_clouds and np.random.rand() < 0.7:
                    coords, types = elite_clouds[np.random.randint(len(elite_clouds))]
                    x0 = torch.tensor(coords, device=self.device)
                    abar = self._abar[sdedit_t]
                    eps = torch.randn_like(x0)
                    eps = eps - eps.mean(0, keepdim=True)
                    x = abar.sqrt() * x0 + (1 - abar).sqrt() * eps
                    h = self._onehot(types)
                    h = abar.sqrt() * h + (1 - abar).sqrt() * torch.randn_like(h)
                    x, h = self._denoise(x, h, from_t=sdedit_t)
                    clouds.append((x.cpu().numpy(), h.argmax(-1).cpu().numpy()))
                else:
                    clouds.append(self._sample_cloud())
            smiles = [cloud_to_mol(c, t) or "" for c, t in clouds]
            rewards = np.asarray(objective(MoleculeSet.from_smiles(smiles)), dtype=float)
            evals += batch
            scored = sorted(zip(clouds, smiles, rewards), key=lambda z: z[2], reverse=True)
            for _, s, r in scored:
                if s:
                    collected[s] = max(collected.get(s, -1e9), float(r))
            elite_clouds = [c for c, s, _ in scored[:4] if s]
        ranked = sorted(collected.items(), key=lambda kv: kv[1], reverse=True)
        return MoleculeSet([Molecule(s, props={"objective_score": r}) for s, r in ranked])


__all__ = ["DiffusionGenerator", "mol_to_cloud", "cloud_to_mol", "parse_pocket_pdb"]
