from __future__ import annotations

import itertools
import re
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
    benchmark_spec = config.get("benchmark", {"id": "researchclawbench", "split": "default"})
    benchmark_id, split = benchmark_spec["id"], benchmark_spec.get("split", "default")
    prepared_root = (config_path.parent.parent / paths.get("prepared_tasks", "prepared_tasks")).resolve()
    ordinal = 0
    for host, condition, task, seed, repeat in itertools.product(
        config["hosts"], config["conditions"], config["tasks"], seeds, range(repeats)
    ):
        task_id = str(task)
        identity = task_id
        task_dir = benchmark / "tasks" / task_id
        if benchmark_id != "researchclawbench":
            portable_id = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id) + "_" + config_hash(task_id)[:8]
            identity = f"{benchmark_id}__{split}__{portable_id}"
            task_dir = prepared_root / benchmark_id / split / task_id
        ordinal += 1
        yield {
            "schema_version": 1,
            "ordinal": ordinal,
            "episode_id": f"{experiment_id}__{identity}__{host}__{condition}__s{seed}__r{repeat}",
            "experiment_id": experiment_id,
            "config_sha256": digest,
            "host": host,
            "condition": condition,
            "task_id": task_id,
            "task_dir": str(task_dir),
            "benchmark_id": benchmark_id,
            "split": split,
            "profile": benchmark_spec.get("profile", "default"),
            "runtime": config.get("runtime", {}),
            "seed": seed,
            "repeat": repeat,
            "model": config["model"]["name"],
            "budget": config["budget"],
            "status": "planned",
        }
