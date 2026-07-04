"""Transformer-decoder SMILES generator (M6).

A decoder-only (GPT-style) causal Transformer language model over SMILES
characters.  It is pretrained on the target's actives (MLE) and then
policy-optimized (REINFORCE) against the target objective — the same transfer
idea as the SMILES-LM generator, with a Transformer backbone.  A larger
pretrained transformer prior can be registered in ``PRETRAINED`` (family
``generative_lm``, e.g. ``smiles-lm``) and swapped in as the initialization.

Requires ``torch``; ``graph_ga`` stays the zero-dependency default.
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
            "The transformer generator needs `torch`. Install with `pip install torch`."
        ) from exc


@GENERATORS.register("transformer")
class TransformerGenerator(Generator):
    """Decoder-only causal Transformer LM over SMILES + REINFORCE."""

    def __init__(
        self,
        seed_smiles: list[str] | None = None,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        max_len: int = 120,
        pretrain_epochs: int = 20,
        lr: float = 3e-4,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        self.seed_smiles = list(seed_smiles or [])
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.max_len = max_len
        self.pretrain_epochs = pretrain_epochs
        self.lr = lr
        self.device = device
        self.seed = seed
        self._net = None
        self._vocab: list[str] = []
        self._stoi: dict[str, int] = {}

    def set_seeds(self, smiles: list[str]) -> None:
        self.seed_smiles = list(smiles)

    # -- vocab / model ---------------------------------------------------
    def _build_vocab(self) -> None:
        chars: set[str] = set()
        for s in self.seed_smiles:
            chars.update(s)
        self._vocab = [_PAD, _BOS, _EOS] + sorted(chars)
        self._stoi = {c: i for i, c in enumerate(self._vocab)}

    def _build_net(self):
        import torch
        import torch.nn as nn

        V = len(self._vocab)
        D, L = self.d_model, self.max_len
        nhead, dim_ff, num_layers = self.nhead, self.dim_ff, self.num_layers

        class DecoderLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.tok = nn.Embedding(V, D, padding_idx=0)
                self.pos = nn.Embedding(L + 2, D)
                layer = nn.TransformerEncoderLayer(D, nhead, dim_ff, batch_first=True)
                self.enc = nn.TransformerEncoder(layer, num_layers)
                self.out = nn.Linear(D, V)

            def forward(self, x):
                T = x.size(1)
                positions = torch.arange(T, device=x.device).unsqueeze(0)
                h = self.tok(x) + self.pos(positions)
                # bool causal mask (True = disallowed) — same dtype as the
                # padding mask, which torch now requires to avoid a deprecation.
                mask = torch.triu(
                    torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
                )
                pad_mask = x == 0
                h = self.enc(h, mask=mask, src_key_padding_mask=pad_mask)
                return self.out(h)

        return DecoderLM()

    def _ensure_trained(self):
        _require_torch()
        if self._net is not None:
            return
        if not self.seed_smiles:
            raise ValueError("TransformerGenerator needs seed molecules; call set_seeds().")
        import torch

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        self._build_vocab()
        self._net = self._build_net().to(self.device)
        self._pretrain()

    def _encode_seq(self, s: str):
        import torch

        idx = [self._stoi[_BOS]] + [self._stoi[c] for c in s] + [self._stoi[_EOS]]
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    def _pretrain(self):
        import torch
        import torch.nn.functional as F

        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        seqs = [self._encode_seq(s) for s in self.seed_smiles if len(s) < self.max_len]
        self._net.train()
        for _ in range(self.pretrain_epochs):
            for seq in seqs:
                seq = seq.unsqueeze(0)
                logits = self._net(seq[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, len(self._vocab)),
                    seq[0, 1:],
                    ignore_index=self._stoi[_PAD],
                )
                opt.zero_grad()
                loss.backward()
                opt.step()

    # -- sampling --------------------------------------------------------
    def _sample_one(self, collect_logp: bool = False):
        import torch

        toks = [self._stoi[_BOS]]
        chars: list[str] = []
        logps = []
        for _ in range(self.max_len):
            x = torch.tensor([toks], device=self.device)
            logits = self._net(x)[0, -1]
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
            if collect_logp:
                logps.append(torch.log(probs[nxt] + 1e-9))
            ch = self._vocab[nxt]
            if ch == _EOS:
                break
            if ch not in (_BOS, _PAD):
                chars.append(ch)
            toks.append(nxt)
        smi = "".join(chars)
        if collect_logp:
            lp = torch.stack(logps).sum() if logps else torch.tensor(0.0, device=self.device)
            return smi, lp
        return smi

    def sample(self, n: int) -> MoleculeSet:
        self._ensure_trained()
        import torch

        self._net.eval()
        with torch.no_grad():
            mols = [Molecule(self._sample_one()) for _ in range(n)]
        return MoleculeSet(mols)

    def optimize_for_target(self, objective: ScoringFunction, budget: int) -> MoleculeSet:
        """REINFORCE with a mean baseline; reward = objective desirability."""
        self._ensure_trained()
        import torch

        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        collected: dict[str, float] = {}
        batch, evals = 16, 0
        while evals < budget:
            self._net.train()
            smiles, logps = [], []
            for _ in range(batch):
                s, lp = self._sample_one(collect_logp=True)
                smiles.append(s)
                logps.append(lp)
            rewards = np.asarray(objective(MoleculeSet.from_smiles(smiles)), dtype=float)
            evals += batch
            for s, r in zip(smiles, rewards):
                collected[s] = max(collected.get(s, -1e9), float(r))
            baseline = float(rewards.mean())
            loss = -torch.stack(
                [lp * (float(r) - baseline) for lp, r in zip(logps, rewards)]
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        ranked = sorted(collected.items(), key=lambda kv: kv[1], reverse=True)
        return MoleculeSet([Molecule(s, props={"objective_score": r}) for s, r in ranked])
