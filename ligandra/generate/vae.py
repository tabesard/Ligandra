"""Latent-space VAE generator (M6).

A char-level SMILES variational autoencoder: a GRU encoder maps a molecule to a
Gaussian latent ``(mu, logvar)``; a GRU decoder reconstructs SMILES from a latent
sample.  Trained on the target's actives with reconstruction + KL loss.

* ``sample(n)`` decodes fresh molecules from the prior ``N(0, I)``.
* ``optimize_for_target`` performs **continuous optimization in latent space**:
  it seeds a population from the encoded actives and evolves it by Gaussian
  perturbation, decoding + scoring each candidate against the target objective.

Requires ``torch`` (a clear error is raised otherwise); the CPU-only ``graph_ga``
remains the zero-dependency default.
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
            "The VAE generator needs `torch`. Install it with `pip install torch`."
        ) from exc


@GENERATORS.register("vae")
class VAEGenerator(Generator):
    """Char-level SMILES VAE with latent-space optimization."""

    def __init__(
        self,
        seed_smiles: list[str] | None = None,
        embed_dim: int = 64,
        hidden: int = 256,
        latent: int = 64,
        max_len: int = 120,
        epochs: int = 40,
        lr: float = 1e-3,
        beta: float = 0.1,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        self.seed_smiles = list(seed_smiles or [])
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.latent = latent
        self.max_len = max_len
        self.epochs = epochs
        self.lr = lr
        self.beta = beta
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
        import torch.nn as nn

        V, E, H, Z = len(self._vocab), self.embed_dim, self.hidden, self.latent

        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(V, E, padding_idx=0)
                self.enc = nn.GRU(E, H, batch_first=True)
                self.fc_mu = nn.Linear(H, Z)
                self.fc_lv = nn.Linear(H, Z)
                self.z2h = nn.Linear(Z, H)
                self.dec = nn.GRU(E, H, batch_first=True)
                self.out = nn.Linear(H, V)

            def encode(self, x):
                _, h = self.enc(self.emb(x))
                h = h[-1]
                return self.fc_mu(h), self.fc_lv(h)

            def decode_step(self, tok, h):
                o, h = self.dec(self.emb(tok), h)
                return self.out(o[:, -1]), h

            def h0(self, z):
                import torch

                return torch.tanh(self.z2h(z)).unsqueeze(0)

        return VAE()

    def _ensure_trained(self):
        _require_torch()
        if self._net is not None:
            return
        if not self.seed_smiles:
            raise ValueError("VAEGenerator needs seed molecules; call set_seeds().")
        import torch

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        self._build_vocab()
        self._net = self._build_net().to(self.device)
        self._train()

    def _encode_seq(self, s: str):
        import torch

        idx = [self._stoi[_BOS]] + [self._stoi[c] for c in s] + [self._stoi[_EOS]]
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    def _train(self):
        import torch
        import torch.nn.functional as F

        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        seqs = [self._encode_seq(s) for s in self.seed_smiles if len(s) < self.max_len]
        self._net.train()
        for _ in range(self.epochs):
            for seq in seqs:
                seq = seq.unsqueeze(0)  # (1, T)
                mu, logvar = self._net.encode(seq)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                h = self._net.h0(z)
                inp, tgt = seq[:, :-1], seq[:, 1:]
                o, _ = self._net.dec(self._net.emb(inp), h)
                logits = self._net.out(o)
                recon = F.cross_entropy(
                    logits.reshape(-1, len(self._vocab)),
                    tgt.reshape(-1),
                    ignore_index=self._stoi[_PAD],
                )
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon + self.beta * kl
                opt.zero_grad()
                loss.backward()
                opt.step()

    # -- decoding --------------------------------------------------------
    def _decode(self, z, stochastic: bool = True) -> str:
        import torch

        self._net.eval()
        h = self._net.h0(z if z.dim() == 2 else z.unsqueeze(0))
        tok = torch.tensor([[self._stoi[_BOS]]], device=self.device)
        chars: list[str] = []
        with torch.no_grad():
            for _ in range(self.max_len):
                logits, h = self._net.decode_step(tok, h)
                probs = torch.softmax(logits[0], dim=-1)
                nxt = int(torch.multinomial(probs, 1)) if stochastic else int(probs.argmax())
                ch = self._vocab[nxt]
                if ch == _EOS:
                    break
                if ch not in (_BOS, _PAD):
                    chars.append(ch)
                tok = torch.tensor([[nxt]], device=self.device)
        return "".join(chars)

    def _prior_sample(self):
        import torch

        return torch.randn(1, self.latent, device=self.device)

    def sample(self, n: int) -> MoleculeSet:
        self._ensure_trained()
        return MoleculeSet([Molecule(self._decode(self._prior_sample())) for _ in range(n)])

    # -- latent-space optimization --------------------------------------
    def optimize_for_target(self, objective: ScoringFunction, budget: int) -> MoleculeSet:
        """Evolve a latent population toward higher objective desirability."""
        self._ensure_trained()
        import torch

        # Seed the population from the encoded actives (their posterior means).
        anchors = []
        self._net.eval()
        with torch.no_grad():
            for s in self.seed_smiles[:64]:
                if len(s) >= self.max_len:
                    continue
                mu, _ = self._net.encode(self._encode_seq(s).unsqueeze(0))
                anchors.append(mu.squeeze(0))
        if not anchors:
            anchors = [self._prior_sample().squeeze(0)]

        pop = [a.clone() for a in anchors]
        while len(pop) < 32:
            pop.append(self._prior_sample().squeeze(0))

        collected: dict[str, float] = {}
        evals = 0
        sigma = 0.5
        while evals < budget:
            smiles = [self._decode(z) for z in pop]
            rewards = np.asarray(objective(MoleculeSet.from_smiles(smiles)), dtype=float)
            evals += len(pop)
            scored = sorted(zip(pop, smiles, rewards), key=lambda t: t[2], reverse=True)
            for _, s, r in scored:
                if s:
                    collected[s] = max(collected.get(s, -1e9), float(r))
            # elitism: keep top parents, breed children by Gaussian perturbation
            parents = [z for z, _, _ in scored[: max(2, len(scored) // 4)]]
            children = []
            for z in parents:
                for _ in range(3):
                    children.append(z + sigma * torch.randn_like(z))
            pop = parents + children
            sigma *= 0.95  # anneal exploration
        ranked = sorted(collected.items(), key=lambda kv: kv[1], reverse=True)
        return MoleculeSet([Molecule(s, props={"objective_score": r}) for s, r in ranked])
