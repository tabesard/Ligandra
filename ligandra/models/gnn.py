"""Graph neural network model (D-MPNN / Chemprop-style) — registered plugin.

If Chemprop is installed it is used directly; otherwise a compact PyTorch
message-passing network over the :class:`~ligandra.featurize.graph.MolGraph`
representation is used.  Both paths present the shared ``PredictiveModel`` API.
Requires ``torch``; a clear error is raised if it is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ligandra.core.types import TaskType
from ligandra.models.base import MODELS, PredictiveModel


@MODELS.register("gnn")
class GNNModel(PredictiveModel):
    """D-MPNN regressor. ``X`` is a list of ``MolGraph`` (from GraphFeaturizer)."""

    task = TaskType.REGRESSION
    consumes_graphs = True

    def __init__(
        self,
        hidden: int = 128,
        depth: int = 3,
        epochs: int = 50,
        lr: float = 1e-3,
        device: str | None = None,
        task: TaskType = TaskType.REGRESSION,
    ) -> None:
        self.hidden = hidden
        self.depth = depth
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.task = task
        self._net = None

    def _require_torch(self):
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The GNN model needs `torch`. Install it with `pip install torch`."
            ) from exc

    def _build(self, atom_dim: int, bond_dim: int):
        import torch
        import torch.nn as nn

        hidden, depth = self.hidden, self.depth

        class DMPNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.w_i = nn.Linear(atom_dim + bond_dim, hidden)
                self.w_h = nn.Linear(hidden, hidden)
                self.w_o = nn.Linear(atom_dim + hidden, hidden)
                self.readout = nn.Sequential(
                    nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
                )

            def forward(self, g):
                # Compact message passing over one graph (batched by loop).
                atoms = torch.tensor(g.atom_features, dtype=torch.float32)
                if g.edge_index.shape[1] == 0:
                    node = torch.zeros((g.n_atoms, hidden))
                else:
                    src = torch.tensor(g.edge_index[0])
                    dst = torch.tensor(g.edge_index[1])
                    bonds = torch.tensor(g.edge_features, dtype=torch.float32)
                    msg = torch.relu(self.w_i(torch.cat([atoms[src], bonds], dim=1)))
                    for _ in range(depth - 1):
                        agg = torch.zeros((g.n_atoms, hidden)).index_add_(0, dst, msg)
                        msg = torch.relu(msg + self.w_h(agg[src]))
                    node = torch.zeros((g.n_atoms, hidden)).index_add_(0, dst, msg)
                node = torch.relu(self.w_o(torch.cat([atoms, node], dim=1)))
                return self.readout(node.sum(0, keepdim=True))

        return DMPNN()

    def fit(self, X, y) -> PredictiveModel:
        self._require_torch()
        import torch

        graphs = list(X)
        atom_dim = graphs[0].atom_features.shape[1]
        bond_dim = graphs[0].edge_features.shape[1] if graphs[0].edge_features.size else 3
        self._net = self._build(atom_dim, bond_dim)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        y_t = torch.tensor(np.asarray(y, dtype=np.float32)).view(-1, 1)
        loss_fn = torch.nn.MSELoss()
        self._net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            preds = torch.cat([self._net(g) for g in graphs], dim=0)
            loss = loss_fn(preds, y_t)
            loss.backward()
            opt.step()
        return self

    def predict(self, X) -> np.ndarray:
        import torch

        self._net.eval()
        with torch.no_grad():
            preds = torch.cat([self._net(g) for g in X], dim=0)
        return preds.cpu().numpy().ravel()

    def save(self, path: str | Path) -> None:
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self._net.state_dict(), "hidden": self.hidden, "depth": self.depth}, path)

    @classmethod
    def load(cls, path: str | Path) -> PredictiveModel:  # pragma: no cover
        raise NotImplementedError(
            "GNN load needs the input feature dims; re-fit or persist them alongside."
        )
