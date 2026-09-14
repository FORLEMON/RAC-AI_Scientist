from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Iterator

from .ledger import config_hash


def expand_matrix(config: dict[str, Any], config_path: Path) -> Iterator[dict[str, Any]]:
    """Expand a frozen config into deterministic, model-free episode records."""
    experiment_id = str(config["experiment_id"])
    repeats = int(config.get("repeats", 1))
    seeds = [int(item) for item in config.get("seeds", [0])]
    digest = config_hash(config)
    paths = config.get("paths", {})
    benchmark = (config_path.parent.parent / paths["benchmark"]).resolve()
    ordinal = 0
    for host, condition, task, seed, repeat in itertools.product(
        config["hosts"], config["conditions"], config["tasks"], seeds, range(repeats)
    ):
        task_id = str(task)
        ordinal += 1
        yield {
            "schema_version": 1,
            "ordinal": ordinal,
            "episode_id": f"{experiment_id}__{task_id}__{host}__{condition}__s{seed}__r{repeat}",
            "experiment_id": experiment_id,
            "config_sha256": digest,
            "host": host,
            "condition": condition,
            "task_id": task_id,
            "task_dir": str(benchmark / "tasks" / task_id),
            "seed": seed,
            "repeat": repeat,
            "model": config["model"]["name"],
            "budget": config["budget"],
            "status": "planned",
        }
