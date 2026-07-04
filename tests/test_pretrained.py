"""Section 7 — transfer learning & representation-consistency contract.

These tests pin the acceptance criteria that make transfer learning *correct*:

* representation consistency is enforced (shared ``preprocess_hash``);
* a molecule's encoding/embedding is byte-identical via the featurizer and via
  the checkpoint's reference spec;
* a mismatched tokenizer/pipeline is rejected;
* selecting a foundation/transformer checkpoint does **not** assemble a
  multi-million-molecule corpus — only the target set is fetched (mocked source);
* data-leakage / applicability-domain guards behave.

They run fully offline via the self-contained ``reference-char`` checkpoint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ligandra.core.molecule import MoleculeSet
from ligandra.core.types import Column
from ligandra.data.base import DATA_SOURCES, DataSource
from ligandra.featurize import build_featurizer
from ligandra.models import build_model
from ligandra.pretrained import (
    PRETRAINED,
    PreprocessingMismatchError,
    PreprocessingSpec,
    applicability_domain_flags,
    assert_consistent_preprocessing,
    dedupe_against,
    inchikey,
    overlap_with_pretraining,
)
from ligandra.pretrained.checkpoints import HashingEncoder

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


# ---------------------------------------------------------------- registry ---
def test_registry_populated_with_families():
    avail = set(PRETRAINED.available())
    assert {"reference-char", "chemberta", "molformer", "smiles-lm", "diffsbdd"} <= avail
    families = {n: PRETRAINED.create(n).family for n in avail}
    assert families["chemberta"] == "predictive_transformer"
    assert families["smiles-lm"] == "generative_lm"
    assert families["diffsbdd"] == "structure_diffusion"


def test_new_checkpoint_is_one_class_addition():
    from ligandra.pretrained.base import PretrainedCheckpoint

    @PRETRAINED.register("dummy_ckpt_test")
    class _Dummy(PretrainedCheckpoint):
        name = "dummy_ckpt_test"
        family = "predictive_transformer"

        def spec(self):
            return PreprocessingSpec(tokenizer_id="char:dummy")

        def load(self):
            return HashingEncoder(8), self.spec()

    assert "dummy_ckpt_test" in PRETRAINED.available()
    model, spec = PRETRAINED.create("dummy_ckpt_test").load()
    assert spec.hash()


# ------------------------------------------------------------------- hash ----
def test_preprocess_hash_is_stable_and_identity_bearing():
    a = PRETRAINED.create("reference-char").spec()
    b = PRETRAINED.create("reference-char").spec()
    assert a.hash() == b.hash()  # deterministic

    # any change to the pipeline changes the hash
    assert a.hash() != PreprocessingSpec(tokenizer_id="char:smiles-v1", max_length=64).hash()
    assert a.hash() != PreprocessingSpec(tokenizer_id="other").hash()


# --------------------------------------------------- byte-identical encoding -
def test_embedding_byte_identical_via_featurizer_and_reference():
    """A held-out molecule's encoding & embedding must be byte-identical whether
    produced through the model's featurizer or the checkpoint's own spec."""
    feat = build_featurizer("learned")  # default reference-char
    spec = feat.spec

    enc_via_featurizer = feat.encode(ASPIRIN)
    enc_via_reference = spec.preprocess(ASPIRIN)
    assert enc_via_featurizer.input_ids == enc_via_reference.input_ids
    assert enc_via_featurizer.attention_mask == enc_via_reference.attention_mask

    emb_featurizer = feat.transform(MoleculeSet.from_smiles([ASPIRIN]))[0]
    emb_reference = HashingEncoder(64).embed(spec.preprocess(ASPIRIN))
    assert np.array_equal(emb_featurizer, emb_reference)


def test_featurizer_uses_only_checkpoint_preprocessing():
    """The learned featurizer must never reach for a generic tokenizer."""
    feat = build_featurizer("learned")
    # its cache key and hash come from the bound spec
    assert feat.preprocess_hash == feat.spec.hash()
    assert feat.preprocess_hash in feat.cache_key


def test_invalid_smiles_rejected_by_preprocess():
    spec = PRETRAINED.create("reference-char").spec()
    with pytest.raises(ValueError):
        spec.preprocess("this is not a molecule )(")


# ----------------------------------------------------------- mismatch guard --
def test_mismatched_tokenizer_is_rejected():
    ref = build_featurizer("learned")  # reference-char
    other = build_featurizer("learned", max_length=64)  # different pipeline
    assert ref.preprocess_hash != other.preprocess_hash
    with pytest.raises(PreprocessingMismatchError):
        assert_consistent_preprocessing(ref, other)


def test_shared_hash_across_model_and_reusing_featurizer():
    """The fine-tuned model and any featurizer/generator reusing it share one
    preprocess_hash (Section 7.2)."""
    model = build_model("foundation", checkpoint="chemberta")
    reuse = build_featurizer("learned", checkpoint="chemberta")
    shared = assert_consistent_preprocessing(model, reuse, model.spec)
    assert shared == model.preprocess_hash


# ---------------------------------------------------------- no corpus grab ---
_SPY_ROWS = [
    ("CCO", 800.0),
    ("c1ccccc1", 5000.0),
    ("CC(=O)O", 20000.0),
    ("CC(=O)Oc1ccccc1C(=O)O", 300.0),
    ("CC(=O)Nc1ccc(O)cc1", 1200.0),
    ("O=C(O)c1ccccc1", 900.0),
    ("c1ccc(-c2ccncc2)cc1", 400.0),
    ("COc1ccc(CCN)cc1", 2500.0),
    ("CC(C)Cc1ccc(C(C)C(=O)O)cc1", 600.0),
    ("O=C(O)Cc1ccccc1", 15000.0),
]


@DATA_SOURCES.register("spy_source")
class _SpyDataSource(DataSource):
    """Counts fetches so we can prove only the target set is pulled."""

    #: class-level counters (single instance per run)
    n_fetch = 0
    n_search = 0
    fetch_targets: list = []

    def search_targets(self, query: str):
        type(self).n_search += 1
        from ligandra.core.types import Target

        return [Target(target_id="SPY1", name=query, source="spy")]

    def fetch_activities(self, target_id: str, endpoint: str, **filters):
        type(self).n_fetch += 1
        type(self).fetch_targets.append(target_id)
        df = pd.DataFrame(
            {
                Column.MOLECULE_ID: [f"m{i}" for i in range(len(_SPY_ROWS))],
                Column.SMILES: [s for s, _ in _SPY_ROWS],
                Column.TARGET_ID: [target_id] * len(_SPY_ROWS),
                Column.ENDPOINT: [endpoint] * len(_SPY_ROWS),
                Column.VALUE: [v for _, v in _SPY_ROWS],
                Column.UNITS: ["nM"] * len(_SPY_ROWS),
            }
        )
        return df


def test_selecting_checkpoint_does_not_assemble_a_corpus():
    from ligandra.config.schema import (
        CacheConfig,
        ExperimentConfig,
        FeaturizerConfig,
        ModelConfig,
        TargetConfig,
    )
    from ligandra.core.types import Endpoint
    from ligandra.pipeline.runner import load_and_curate

    _SpyDataSource.n_fetch = 0
    _SpyDataSource.n_search = 0
    _SpyDataSource.fetch_targets = []

    cfg = ExperimentConfig(
        name="corpus-guard",
        target=TargetConfig(source="spy_source", target_id="SPY1", endpoint=Endpoint.IC50),
        featurizers=[FeaturizerConfig(name="learned", params={"checkpoint": "reference-char"})],
        models=[ModelConfig(name="foundation", params={"checkpoint": "chemberta"})],
        cache=CacheConfig(enabled=False),  # test fetch behavior, not the CSV cache
    )

    # Constructing the featurizer/model + loading the checkpoint must NOT fetch.
    build_featurizer("learned", checkpoint="reference-char")
    build_model("foundation", checkpoint="chemberta")
    PRETRAINED.create("reference-char").load()
    assert _SpyDataSource.n_fetch == 0

    # The data step fetches ONLY the small target set — exactly one call, for the
    # target id; never a bulk/pretraining pull, and no target search (id given).
    curated, _ = load_and_curate(cfg)
    assert _SpyDataSource.n_fetch == 1
    assert _SpyDataSource.n_search == 0
    assert _SpyDataSource.fetch_targets == ["SPY1"]
    assert len(curated) <= len(_SPY_ROWS)  # target-set sized, not a corpus


# ------------------------------------------------------ transfer stays local -
def test_reference_transfer_consumes_only_target_set():
    ck = PRETRAINED.create("reference-char")
    target = MoleculeSet.from_smiles(
        ["CCO", "c1ccccc1", "CC(=O)O", "CCN"],
        props=[{"y": 5.0}, {"y": 6.5}, {"y": 4.0}, {"y": 7.0}],
    )
    head = ck.transfer(target, mode="frozen_head")
    assert head.n_train == len(target)
    assert head.preprocess_hash == ck.spec().hash()
    preds = head.predict(["CCC"])
    assert preds.shape == (1,)


def test_structure_diffusion_transfer_needs_pocket():
    ck = PRETRAINED.create("diffsbdd")
    assert ck.family == "structure_diffusion"
    with pytest.raises(NotImplementedError):
        ck.transfer(MoleculeSet.from_smiles(["CCO"]), mode="condition")


# ------------------------------------------------------ leakage / AD guards --
def test_overlap_and_dedupe_by_inchikey():
    ik = inchikey("CCO")
    assert ik and ik.count("-") == 2  # standard InChIKey shape
    rep = overlap_with_pretraining(["CCO", "c1ccccc1", "CCN"], {ik})
    assert rep.n_total == 3 and rep.n_overlap == 1
    assert "CCO" in rep.overlapping_smiles
    kept = dedupe_against(["CCO", "c1ccccc1"], {ik})
    assert kept == ["c1ccccc1"]


def test_applicability_domain_flags_outliers():
    rng = np.random.RandomState(0)
    train = rng.randn(30, 8)
    query = np.vstack([train[0], np.ones(8) * 50.0])  # in-domain, far-outlier
    flags = applicability_domain_flags(train, query)
    assert flags.tolist() == [False, True]
