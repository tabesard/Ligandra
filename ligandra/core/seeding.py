"""Global reproducibility controls.

Call :func:`set_global_seed` once at the start of an experiment.  It seeds the
standard library, NumPy, and (if installed) PyTorch, and returns the seed so it
can be recorded in the run manifest.
"""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED, *, deterministic_torch: bool = True) -> int:
    """Seed every RNG we can reach. Returns the seed for logging.

    Optional heavy libraries (NumPy, PyTorch) are seeded only if importable, so
    this works on a bare interpreter.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
