"""Registry wiring and the extensibility contract."""

from __future__ import annotations

import pytest

from ligandra.core.registry import Registry


def test_register_create_and_available():
    reg: Registry = Registry("thing")

    @reg.register("foo")
    class Foo:
        def __init__(self, x=1):
            self.x = x

    assert "foo" in reg
    assert reg.available() == ["foo"]
    obj = reg.create("foo", x=5)
    assert obj.x == 5
    assert Foo.registry_name == "foo"


def test_duplicate_registration_raises():
    reg: Registry = Registry("thing")

    @reg.register("dup")
    class A:
        pass

    with pytest.raises(ValueError):

        @reg.register("dup")
        class B:
            pass


def test_unknown_name_raises_with_hint():
    reg: Registry = Registry("thing")
    with pytest.raises(KeyError):
        reg.create("missing")


def test_all_builtin_registries_populated():
    from ligandra.data import DATA_SOURCES
    from ligandra.dock import DOCKERS
    from ligandra.featurize import FEATURIZERS
    from ligandra.generate import GENERATORS
    from ligandra.models import MODELS
    from ligandra.pretrained import PRETRAINED
    from ligandra.score import SCORERS

    assert {"chembl", "local"} <= set(DATA_SOURCES.available())
    assert {"ecfp", "maccs", "lipinski", "rdkit_descriptors"} <= set(FEATURIZERS.available())
    assert {"linear", "ridge", "lasso", "random_forest", "gnn", "foundation"} <= set(MODELS.available())
    assert {"graph_ga", "smiles_lm", "vae", "transformer", "diffusion"} <= set(GENERATORS.available())
    assert {"potency", "qed", "sa", "novelty", "druglikeness"} <= set(SCORERS.available())
    assert "vina" in DOCKERS.available()
    assert {"reference-char", "chemberta", "molformer"} <= set(PRETRAINED.available())


def test_new_model_is_one_class_addition():
    """A new model shows up everywhere with just a subclass + decorator."""
    from ligandra.models import MODELS
    from ligandra.models.classical import _SklearnModel

    @MODELS.register("dummy_test_model")
    class DummyModel(_SklearnModel):
        def _make_estimator(self):
            from sklearn.dummy import DummyRegressor

            return DummyRegressor()

    assert "dummy_test_model" in MODELS.available()
    m = MODELS.create("dummy_test_model")
    assert hasattr(m, "fit") and hasattr(m, "predict")
