"""The preprocessing-consistency contract (Section 7.2).

A pretrained molecular foundation model bakes its large-scale chemistry into the
released weights.  For transfer learning to be *correct*, every downstream
molecule must be encoded through the **exact same** representation pipeline the
checkpoint saw during pretraining: same SMILES standardization/canonicalization,
same tokenizer/vocab, same max-length/padding.  If they diverge, embeddings are
out-of-distribution and fine-tuning silently learns nothing transferable.

:class:`PreprocessingSpec` makes that a *bound contract* rather than a
convention.  It captures the full identity of a checkpoint's input pipeline and
exposes:

* :meth:`PreprocessingSpec.preprocess` — the single, canonical way to turn a raw
  SMILES into token ids (an :class:`EncodedMol`); and
* :meth:`PreprocessingSpec.hash` — a stable ``preprocess_hash`` that is stored in
  the model registry and asserted identical across the fine-tune set, any later
  inference set, and any generator that reuses the model.

Two tokenizer back ends are supported:

``char``
    A fully self-contained, deterministic SMILES regex tokenizer with an
    explicit vocabulary.  Needs **no** heavy dependencies, so the whole contract
    is testable and usable on a bare interpreter (this is what the built-in
    ``reference-char`` checkpoint uses).
``hf``
    A Hugging Face tokenizer referenced by ``tokenizer_id`` (used by the real
    ChemBERTa / MoLFormer checkpoints).  Requires ``transformers``.

Either way, standardization/canonicalization runs first, through the exact
``canonicalizer`` function named in the spec (a dotted import path), so the
contract covers the *whole* pipeline and not just tokenization.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

# --- SMILES regex tokenizer ------------------------------------------------
# The widely used atom-wise SMILES pattern (Schwaller et al.): keeps bracketed
# atoms, two-letter halogens, ring bonds and %NN ring closures as single tokens.
_SMILES_TOKEN_RE = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|se|@@|%\d{2}|B|C|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\*|\$|[0-9])"
)

#: Explicit, ordered default vocabulary for the ``char`` back end.  Ordered so
#: token ids are stable and part of the ``preprocess_hash``.  Anything not in
#: here maps to ``<unk>`` — deterministically.
DEFAULT_SPECIAL_TOKENS: dict[str, str] = {
    "pad": "<pad>",
    "unk": "<unk>",
    "bos": "<bos>",
    "eos": "<eos>",
}

_CORE_SMILES_TOKENS: tuple[str, ...] = (
    # organic subset + aromatic
    "C",
    "c",
    "N",
    "n",
    "O",
    "o",
    "S",
    "s",
    "P",
    "p",
    "F",
    "Cl",
    "Br",
    "I",
    "B",
    "b",
    "Si",
    "Se",
    "se",
    # common bracketed atoms
    "[nH]",
    "[C@H]",
    "[C@@H]",
    "[C@]",
    "[C@@]",
    "[NH]",
    "[NH2]",
    "[N+]",
    "[O-]",
    "[n+]",
    "[S+]",
    "[N-]",
    "[OH]",
    "[H]",
    "[2H]",
    # bonds / branches / structure
    "(",
    ")",
    "=",
    "#",
    "-",
    "+",
    "\\",
    "/",
    ":",
    "~",
    "@",
    "@@",
    ".",
    "*",
    # ring closures
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "%10",
    "%11",
    "%12",
    "%13",
    "%14",
    "%15",
)

DEFAULT_SMILES_VOCAB: tuple[str, ...] = tuple(DEFAULT_SPECIAL_TOKENS.values()) + _CORE_SMILES_TOKENS


@dataclass(frozen=True)
class EncodedMol:
    """The immutable result of running a molecule through a spec's pipeline."""

    canonical_smiles: str
    representation: str  # "smiles" | "selfies"
    tokens: tuple[str, ...]
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    def ids_array(self) -> np.ndarray:
        return np.asarray(self.input_ids, dtype=np.int64)

    def mask_array(self) -> np.ndarray:
        return np.asarray(self.attention_mask, dtype=np.int64)


def _resolve_callable(dotted: str):
    """Import and return the callable named by a dotted path ``pkg.mod.attr``."""
    module_path, _, attr = dotted.rpartition(".")
    if not module_path:
        raise ValueError(f"Not a dotted import path: {dotted!r}")
    return getattr(importlib.import_module(module_path), attr)


@lru_cache(maxsize=8)
def _load_hf_tokenizer(tokenizer_id: str):  # pragma: no cover - needs transformers
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "This checkpoint's PreprocessingSpec uses a Hugging Face tokenizer "
            f"({tokenizer_id!r}); install `transformers` to use it."
        ) from exc
    return AutoTokenizer.from_pretrained(tokenizer_id)


@dataclass(frozen=True)
class PreprocessingSpec:
    """A bound description of a checkpoint's input pipeline.

    Parameters
    ----------
    tokenizer_id:
        Identity of the tokenizer.  For ``backend="hf"`` this is the Hugging
        Face repo id; for ``backend="char"`` it is a free label (e.g.
        ``"char:smiles-v1"``) that participates in the hash.
    representation:
        ``"smiles"`` or ``"selfies"``.
    canonicalizer:
        Dotted path to the *exact* standardization function applied before
        tokenization (e.g. ``ligandra.curate.standardize.standardize_smiles``).
    rdkit_version:
        RDKit version the checkpoint's canonicalization was defined against —
        pinned into the hash so a toolkit change is detectable.
    max_length:
        Padding/truncation length.
    special_tokens:
        Role → token string (``pad``/``unk``/``bos``/``eos``).
    backend:
        ``"char"`` (self-contained regex tokenizer) or ``"hf"``.
    vocab:
        Ordered vocabulary for the ``char`` back end (ignored for ``hf``).
    normalization:
        Free-text label for the normalization regime (part of the hash).
    """

    tokenizer_id: str
    representation: str = "smiles"
    canonicalizer: str = "ligandra.curate.standardize.standardize_smiles"
    rdkit_version: str = ""
    max_length: int = 128
    special_tokens: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_SPECIAL_TOKENS))
    backend: str = "char"
    vocab: Sequence[str] = DEFAULT_SMILES_VOCAB
    normalization: str = "canonical"

    # -- canonicalization ----------------------------------------------------
    def canonicalize(self, smiles: str) -> str | None:
        """Apply the spec's exact standardizer; ``None`` if it can't parse."""
        fn = _resolve_callable(self.canonicalizer)
        return fn(smiles)

    # -- tokenization --------------------------------------------------------
    def _vocab_index(self) -> dict[str, int]:
        return {tok: i for i, tok in enumerate(self.vocab)}

    def _tokenize_text(self, text: str) -> list[str]:
        if self.representation == "selfies":
            try:
                import selfies as sf
            except ImportError as exc:  # pragma: no cover - optional
                raise ImportError("representation='selfies' needs the `selfies` package.") from exc
            return list(sf.split_selfies(text))
        return _SMILES_TOKEN_RE.findall(text)

    def preprocess(self, smiles: str) -> EncodedMol:
        """Canonicalize → tokenize → pad/truncate.  The one true encoding path."""
        canonical = self.canonicalize(smiles)
        if canonical is None:
            raise ValueError(f"Cannot preprocess invalid SMILES: {smiles!r}")

        text = canonical
        if self.representation == "selfies":
            import selfies as sf  # pragma: no cover - optional

            text = sf.encoder(canonical)

        if self.backend == "hf":
            return self._preprocess_hf(canonical, text)
        return self._preprocess_char(canonical, text)

    def _preprocess_char(self, canonical: str, text: str) -> EncodedMol:
        vocab = self._vocab_index()
        pad_id = vocab[self.special_tokens["pad"]]
        unk_id = vocab[self.special_tokens["unk"]]
        bos_id = vocab[self.special_tokens["bos"]]
        eos_id = vocab[self.special_tokens["eos"]]

        core_tokens = self._tokenize_text(text)
        # reserve room for <bos>/<eos>
        core_tokens = core_tokens[: max(0, self.max_length - 2)]
        tokens = [self.special_tokens["bos"]] + core_tokens + [self.special_tokens["eos"]]
        ids = [bos_id] + [vocab.get(t, unk_id) for t in core_tokens] + [eos_id]
        mask = [1] * len(ids)
        # right-pad to max_length
        pad_n = self.max_length - len(ids)
        if pad_n > 0:
            ids += [pad_id] * pad_n
            mask += [0] * pad_n
            tokens += [self.special_tokens["pad"]] * pad_n
        return EncodedMol(
            canonical_smiles=canonical,
            representation=self.representation,
            tokens=tuple(tokens),
            input_ids=tuple(ids),
            attention_mask=tuple(mask),
        )

    def _preprocess_hf(
        self, canonical: str, text: str
    ) -> EncodedMol:  # pragma: no cover - needs transformers
        tok = _load_hf_tokenizer(self.tokenizer_id)
        enc = tok(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        ids = list(enc["input_ids"])
        mask = list(enc["attention_mask"])
        tokens = tuple(tok.convert_ids_to_tokens(ids))
        return EncodedMol(
            canonical_smiles=canonical,
            representation=self.representation,
            tokens=tokens,
            input_ids=tuple(ids),
            attention_mask=tuple(mask),
        )

    # -- identity ------------------------------------------------------------
    def _identity(self) -> dict:
        """The fields that define this pipeline's identity (drive the hash)."""
        return {
            "tokenizer_id": self.tokenizer_id,
            "representation": self.representation,
            "canonicalizer": self.canonicalizer,
            "rdkit_version": self.rdkit_version,
            "max_length": self.max_length,
            "special_tokens": dict(sorted(self.special_tokens.items())),
            "backend": self.backend,
            "vocab": list(self.vocab) if self.backend == "char" else None,
            "normalization": self.normalization,
        }

    def hash(self) -> str:
        """Stable 16-hex ``preprocess_hash`` over the pipeline identity."""
        payload = json.dumps(self._identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def __hash__(self) -> int:  # make specs usable as dict keys / in sets
        return int(self.hash(), 16)
