from __future__ import annotations

import os
import random


def seed_runtime(seed: int) -> None:
    """Seed libraries already available in a host without adding dependencies."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
