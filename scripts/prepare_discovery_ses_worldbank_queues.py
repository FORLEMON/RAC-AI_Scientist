"""Prepare four independent DiscoveryBench N0-R3 lanes without starting them."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import stat
import time

from rac_ai_scientist.benchmarks import get_adapter, load_prepared
from rac_ai_scientist.benchmarks.base import digest, write_json
from rac_ai_scientist.benchmarks.upstream import assert_revision
from rac_ai_scientist.config import load_config, validate_config
from rac_ai_scientist.matrix import expand_matrix


TASKS = (
    "nls_ses/metadata_0/0/0",
    "worldbank_education_gdp/metadata_0/0/0",
)
LANES = (
    ("evo_scientist", TASKS[0], "configs/discoverybench.nls_ses.evo_scientist.json"),
    ("evo_scientist", TASKS[1], "configs/discoverybench.worldbank_education_gdp.evo_scientist.json"),
    ("agent_laboratory", TASKS[0], "configs/discoverybench.nls_ses.agent_laboratory.json"),
    ("agent_laboratory", TASKS[1], "configs/discoverybench.worldbank_education_gdp.agent_laboratory.json"),
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def prepare_task(root: Path, work_root: Path, task_id: str) -> Path:
    source = root / "upstreams/discoverybench"
    destination = work_root / "prepared_tasks/discoverybench/test" / task_id
    if destination.exists():
        spec = load_prepared(destination)
        if (spec.benchmark_id, spec.split, spec.task_id) != ("discoverybench", "test", task_id):
            raise ValueError(f"prepared task identity mismatch: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    get_adapter("discoverybench").prepare(source, destination, task_id=task_id, split="test",
                                            dataset=None, capsules=None)
    return destination


def queue_entry(root: Path, row: dict, config_path: Path, sharednet_root: Path) -> dict:
    task_dir = Path(row["task_dir"])
    condition = row["condition"]
    sharednet_file = None
    if condition != "N0":
        sharednet_file = sharednet_root / f"{safe_name(row['episode_id'])}.env"
        sharednet_file.write_text(
            "# Fill all values before queue validation/start. Use a fresh Room per cell.\n"
            "SHAREDNET_ROOM_ID=\n"
            "SHAREDNET_INVITE=\n"
            "SHAREDNET_BASE_URL=https://www.sharednet.ai\n",
            encoding="utf-8", newline="\n")
        sharednet_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return {
        "ordinal": row["ordinal"],
        "episode_id": row["episode_id"],
        "config": os.path.relpath(config_path, root),
        "config_sha256": row["config_sha256"],
        "benchmark_id": "discoverybench",
        "task_id": row["task_id"],
        "task_dir": os.path.relpath(task_dir, root),
        "task_spec_sha256": digest(task_dir / "task_spec.json"),
        "condition": condition,
        "seed": row["seed"],
        "repeat": row["repeat"],
        "sharednet_env_file": str(sharednet_file) if sharednet_file else None,
        "scorer_source": "upstreams/discoverybench",
        "scorer_dataset": None,
        "log_dir": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--work-root", default="/mnt/data0/ldav/src/rac_discovery_work")
    parser.add_argument("--batch", default="discovery-pro-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    args = parser.parse_args()
    if Path(args.batch).name != args.batch or args.batch in {".", ".."}:
        raise ValueError("batch must be one safe path component")
    root, work_root = Path(args.root).resolve(), Path(args.work_root).resolve()
    assert_revision(root / "upstreams/discoverybench", "discoverybench")
    for task_id in TASKS:
        prepare_task(root, work_root, task_id)

    plan_root = work_root / "queue-plans" / args.batch
    if plan_root.exists():
        raise FileExistsError(f"queue plan already exists: {plan_root}")
    sharednet_root = plan_root / "sharednet"
    sharednet_root.mkdir(parents=True)
    queue_files = []
    for host, task_id, config_name in LANES:
        config_path = root / config_name
        config = load_config(config_path)
        blockers = [item.message for item in validate_config(config, root) if item.level in {"ERROR", "BLOCKED"}]
        if blockers:
            raise ValueError(f"invalid lane config {config_name}: {'; '.join(blockers)}")
        rows = list(expand_matrix(config, config_path))
        if [row["condition"] for row in rows] != ["N0", "R1", "R2", "R3"]:
            raise ValueError(f"unexpected condition order in {config_name}")
        queue_id = safe_name(f"{host}-{task_id}")
        queue_path = plan_root / f"{queue_id}.json"
        write_json(queue_path, {
            "schema_version": 1,
            "manifest_kind": "queue_plan_only",
            "queue_id": queue_id,
            "host": host,
            "path_base": "repository_root",
            "environment_file": str(root / ".env"),
            "policy": {
                "max_parallel_episodes": 1,
                "score_after_each_episode": True,
                "continue_on_episode_error": True,
                "continue_on_scoring_error": True,
                "retry_automatically": False,
            },
            "episodes": [queue_entry(root, row, config_path, sharednet_root) for row in rows],
        })
        queue_files.append(str(queue_path))

    execution_path = plan_root / "execution.json"
    write_json(execution_path, {
        "global_parallelism": 4,
        "host_order": ["evo_scientist", "agent_laboratory"],
        "queue_files": queue_files,
        "run_root": str(work_root / "runs"),
        "host_cpus": 6,
        "host_cpuset_cpus": "2-7",
        "host_memory": "4g",
        "task_cpus": 6,
        "task_cpuset_cpus": "2-7",
        "task_memory": "2g",
        "cpu_guard": {
            "sample_seconds": 5, "pause_above_percent": 90, "overload_seconds": 20,
            "resume_below_percent": 70, "recovery_seconds": 10,
            "max_pause_seconds": 20, "cooldown_seconds": 20,
        },
        "score_timeout": 1800,
        "host_images": {
            "evo_scientist": "rac-discoverybench/evo-scientist:0de1937",
            "agent_laboratory": "rac-discoverybench/agent-laboratory:0de1937",
        },
        "scorer_image": "rac-discoverybench/benchmark-scorer:32f7657",
        "task_runtime_image": "rac-discoverybench/task-runtime:32f7657",
        "agent_relay": {
            "container": "rac-ai_scientist-azure-ai-relay-pro-1",
            "expected_model": "DeepSeek-V4-Pro",
            "port": 8000,
        },
        "pricing": {
            "input_usd_per_million": 2.0,
            "output_usd_per_million": 4.0,
            "source": "Existing conservative DeepSeek V4 Pro local accounting; cached input charged as uncached.",
            "accounting": "Local conservative estimate, not a provider billing statement.",
        },
    })

    repo_q, plan_q, batch_q = map(shlex.quote, (str(root), str(plan_root), args.batch))
    common = f"cd {repo_q} && PYTHONPATH=src python3 -u scripts/run_smoke_queue.py --root {repo_q} --batch {batch_q} --execution-config {shlex.quote(str(execution_path))}"
    write_executable(plan_root / "check.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + common + " --check\n")
    write_executable(plan_root / "start.sh", "#!/usr/bin/env bash\nset -euo pipefail\n"
                     f"{plan_q}/check.sh\n"
                     f"session={shlex.quote('rac-' + safe_name(args.batch)[:60])}\n"
                     "if tmux has-session -t \"$session\" 2>/dev/null; then echo \"already running: $session\" >&2; exit 1; fi\n"
                     f"tmux new-session -d -s \"$session\" {shlex.quote(common + ' >> ' + str(plan_root / 'launcher.log') + ' 2>&1')}\n"
                     "echo \"started tmux session: $session\"\n")
    write_executable(plan_root / "status.sh", "#!/usr/bin/env bash\nset -euo pipefail\n"
                     f"status={shlex.quote(str(work_root / 'runs' / args.batch / 'status.json'))}\n"
                     "if [[ -f \"$status\" ]]; then python3 -m json.tool \"$status\"; else echo 'not started'; fi\n")
    write_executable(plan_root / "tail.sh", "#!/usr/bin/env bash\nset -euo pipefail\n"
                     f"touch {shlex.quote(str(plan_root / 'launcher.log'))}\n"
                     f"tail -n 100 -F {shlex.quote(str(plan_root / 'launcher.log'))}\n")
    (plan_root / "README.txt").write_text(
        "Fill every sharednet/*.env file with a unique room id and complete invite.\n"
        "Then run ./check.sh. Only after it reports 16 valid episodes, run ./start.sh.\n"
        "Each of four lanes runs N0, R1, R2, R3 serially; lanes run in parallel.\n"
        "Every cell is scored, and failures are recorded without stopping later cells.\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({"batch": args.batch, "plan_root": str(plan_root), "execution": str(execution_path),
                      "lanes": 4, "episodes": 16, "sharednet_files_to_fill": 12}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
