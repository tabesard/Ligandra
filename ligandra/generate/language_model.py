"""SMILES language-model generator with REINFORCE policy optimization (M6).

A char-level GRU over SMILES that can be pretrained on actives and then
policy-optimized (REINVENT-style) against the target objective.  Requires
``torch``; a clear error is raised if it is missing.  This is the [SHOULD]
primary generative engine; the CPU-only ``graph_ga`` is the zero-dependency
default so the pipeline always runs.
"""

from __future__ import annotations

import numpy as np

from ligandra.core.molecule import Molecule, MoleculeSet
from ligandra.generate.base import GENERATORS, Generator
from ligandra.score.base import ScoringFunction

_BOS, _EOS, _PAD = "^", "$", " "


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The SMILES language-model generator needs `torch`. Install it with "
            "`pip install torch`."
        ) from exc


@GENERATORS.register("smiles_lm")
class SmilesLMGenerator(Generator):
    """Char-level GRU SMILES generator + REINFORCE fine-tuning."""

    def __init__(
        self,
        seed_smiles: list[str] | None = None,
        embed_dim: int = 128,
        hidden: int = 256,
        max_len: int = 120,
        device: str | None = None,
        pretrain_epochs: int = 5,
        seed: int = 42,
    ) -> None:
        self.seed_smiles = list(seed_smiles or [])
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.max_len = max_len
        self.device = device
        self.pretrain_epochs = pretrain_epochs
        self.seed = seed
        self._net = None
        self._vocab: list[str] = []
        self._stoi: dict[str, int] = {}

    def set_seeds(self, smiles: list[str]) -> None:
        self.seed_smiles = list(smiles)

    # -- vocab / model ---------------------------------------------------
    def _build_vocab(self) -> None:
        chars = set()
        for s in self.seed_smiles:
            chars.update(s)
        self._vocab = [_PAD, _BOS, _EOS] + sorted(chars)
        self._stoi = {c: i for i, c in enumerate(self._vocab)}

    def _build_net(self):
        import torch.nn as nn

        vocab = len(self._vocab)

        class GRULM(nn.Module):
            def __init__(self, embed_dim, hidden):
                super().__init__()
                self.emb = nn.Embedding(vocab, embed_dim)
                self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
                self.out = nn.Linear(hidden, vocab)

            def forward(self, x, h=None):
                e = self.emb(x)
                o, h = self.gru(e, h)
                return self.out(o), h

        return GRULM(self.embed_dim, self.hidden)

    def _ensure_trained(self):
        _require_torch()
        if self._net is not None:
            return
        if not self.seed_smiles:
            raise ValueError("SmilesLMGenerator needs seed molecules; call set_seeds().")
        import torch

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        self._build_vocab()
        self._net = self._build_net().to(self.device)
        self._pretrain()

    def _encode(self, s: str):
        import torch

        idx = [self._stoi[_BOS]] + [self._stoi[c] for c in s] + [self._stoi[_EOS]]
        return torch.tensor(idx, dtype=torch.long)

    def _pretrain(self):
        import torch
        import torch.nn.functional as F

        opt = torch.optim.Adam(self._net.parameters(), lr=1e-3)
        seqs = [self._encode(s) for s in self.seed_smiles if len(s) < self.max_len]
        self._net.train()
        for _ in range(self.pretrain_epochs):
            for seq in seqs:
                seq = seq.to(self.device).unsqueeze(0)
                logits, _ = self._net(seq[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, len(self._vocab)), seq[0, 1:])
                opt.zero_grad()
                loss.backward()
                opt.step()

    # -- sampling --------------------------------------------------------
    def _sample_one(self):
        import torch

        self._net.eval()
        tok = torch.tensor([[self._stoi[_BOS]]], device=self.device)
        h = None
        chars: list[str] = []
        with torch.no_grad():
            for _ in range(self.max_len):
                logits, h = self._net(tok, h)
                probs = torch.softmax(logits[0, -1], dim=-1)
                nxt = int(torch.multinomial(probs, 1))
                ch = self._vocab[nxt]
                if ch == _EOS:
                    break
                if ch not in (_BOS, _PAD):
                    chars.append(ch)
                tok = torch.tensor([[nxt]], device=self.device)
        return "".join(chars)

    def sample(self, n: int) -> MoleculeSet:
        self._ensure_trained()
        return MoleculeSet([Molecule(self._sample_one()) for _ in range(n)])

    def optimize_for_target(self, objective: ScoringFunction, budget: int) -> MoleculeSet:
        """REINFORCE: reward = objective desirability of each sampled molecule."""
        self._ensure_trained()
        import torch

        opt = torch.optim.Adam(self._net.parameters(), lr=5e-4)
        collected: dict[str, float] = {}
        batch = 32
        evals = 0
        while evals < budget:
            self._net.train()
            batch_smiles, logps = [], []
            for _ in range(batch):
                tok = torch.tensor([[self._stoi[_BOS]]], device=self.device)
                h, lp, chars = None, [], []
                for _ in range(self.max_len):
                    logits, h = self._net(tok, h)
                    probs = torch.softmax(logits[0, -1], dim=-1)
                    nxt = int(torch.multinomial(probs, 1))
                    lp.append(torch.log(probs[nxt] + 1e-9))
                    ch = self._vocab[nxt]
                    if ch == _EOS:
                        break
                    if ch not in (_BOS, _PAD):
                        chars.append(ch)
                    tok = torch.tensor([[nxt]], device=self.device)
                batch_smiles.append("".join(chars))
                logps.append(torch.stack(lp).sum() if lp else torch.tensor(0.0, device=self.device))
            rewards = np.asarray(objective(MoleculeSet.from_smiles(batch_smiles)), dtype=float)
            evals += batch
            for s, r in zip(batch_smiles, rewards):
                collected[s] = float(r)
            baseline = rewards.mean()
            loss = -torch.stack([lp * (r - baseline) for lp, r in zip(logps, rewards)]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        ranked = sorted(collected.items(), key=lambda kv: kv[1], reverse=True)
        return MoleculeSet([Molecule(s, props={"objective_score": r}) for s, r in ranked])
