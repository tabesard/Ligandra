"""Generative de novo design layer. Importing registers all generators."""

# Import concrete generators so their @register decorators run.
from ligandra.generate import diffusion as _diffusion  # noqa: F401,E402
from ligandra.generate import genetic as _genetic  # noqa: F401,E402
from ligandra.generate import language_model as _lm  # noqa: F401,E402
from ligandra.generate import transformer as _transformer  # noqa: F401,E402
from ligandra.generate import vae as _vae  # noqa: F401,E402
from ligandra.generate.base import (
    GENERATORS,
    GenerationMetrics,
    Generator,
    generation_metrics,
)


def build_generator(name: str, **params) -> Generator:
    return GENERATORS.create(name, **params)


__all__ = [
    "GENERATORS",
    "Generator",
    "GenerationMetrics",
    "generation_metrics",
    "build_generator",
]
