"""Prepare one EvoScientist DiscoveryBench NLS R1-R3 queue without starting it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import time

from rac_ai_scientist.benchmarks.base import write_json
from rac_ai_scientist.benchmarks.upstream import assert_revision
from rac_ai_scientist.config import load_config, validate_config
from rac_ai_scientist.matrix import expand_matrix

from prepare_discovery_ses_worldbank_queues import (
    prepare_task,
    queue_entry,
    safe_name,
    write_executable,
)


TASK_ID = "nls_ses/metadata_0/0/0"
CONFIG = "configs/discoverybench.nls_ses.evo_scientist.json"
CONDITIONS = ("R1", "R2", "R3")
EVO_IMAGE = "rac-discoverybench/evo-scientist:0fa606d"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--work-root", default="/mnt/data0/ldav/src/rac_discovery_work")
    parser.add_argument(
        "--batch",
        default="discovery-nls-evo-r1-r3-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    )
    args = parser.parse_args()
    if Path(args.batch).name != args.batch or args.batch in {".", ".."}:
        raise ValueError("batch must be one safe path component")

    root = Path(args.root).resolve()
    work_root = Path(args.work_root).resolve()
    assert_revision(root / "upstreams/discoverybench", "discoverybench")
    prepare_task(root, work_root, TASK_ID)

    config_path = root / CONFIG
    config = load_config(config_path)
    blockers = [
        item.message
        for item in validate_config(config, root)
        if item.level in {"ERROR", "BLOCKED"}
    ]
    if blockers:
        raise ValueError("invalid EvoScientist NLS config: " + "; ".join(blockers))

    rows = [row for row in expand_matrix(config, config_path) if row["condition"] in CONDITIONS]
    if [row["condition"] for row in rows] != list(CONDITIONS):
        raise ValueError("the source config must produce exactly R1, R2 and R3 in order")
    if any(row["host"] != "evo_scientist" or row["task_id"] != TASK_ID for row in rows):
        raise ValueError("queue scope expanded beyond EvoScientist NLS")

    plan_root = work_root / "queue-plans" / args.batch
    if plan_root.exists():
        raise FileExistsError(f"queue plan already exists: {plan_root}")
    sharednet_root = plan_root / "sharednet"
    sharednet_root.mkdir(parents=True)

    episodes = []
    for row in rows:
        entry = queue_entry(root, row, config_path, sharednet_root)
        entry["cpuset_cpus"] = "8-13"
        episodes.append(entry)

    queue_path = plan_root / "evo_scientist-nls_ses-r1-r3.json"
    write_json(queue_path, {
        "schema_version": 1,
        "manifest_kind": "queue_plan_only",
        "queue_id": "evo_scientist-nls_ses-r1-r3",
        "host": "evo_scientist",
        "resources": {"cpus": 6, "cpuset_cpus": "8-13"},
        "path_base": "repository_root",
        "environment_file": str(root / ".env"),
        "policy": {
            "max_parallel_episodes": 1,
            "score_after_each_episode": True,
            "continue_on_episode_error": True,
            "continue_on_scoring_error": True,
            "retry_automatically": False,
        },
        "episodes": episodes,
    })

    execution_path = plan_root / "execution.json"
    write_json(execution_path, {
        "global_parallelism": 1,
        "controller_lock_scope": "batch",
        "allow_parallel_controllers": True,
        "host_order": ["evo_scientist"],
        "queue_files": [str(queue_path)],
        "run_root": str(work_root / "runs"),
        "host_cpus": 6,
        "host_memory": "4g",
        "task_cpus": 6,
        "task_memory": "2g",
        "score_timeout": 1800,
        "host_images": {"evo_scientist": EVO_IMAGE},
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

    repo_q = shlex.quote(str(root))
    plan_q = shlex.quote(str(plan_root))
    common = (
        f"cd {repo_q} && PYTHONPATH=src python3 -u scripts/run_smoke_queue.py "
        f"--root {repo_q} --batch {shlex.quote(args.batch)} "
        f"--execution-config {shlex.quote(str(execution_path))}"
    )
    write_executable(
        plan_root / "check.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n" + common + " --check\n",
    )
    write_executable(
        plan_root / "start.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"{plan_q}/check.sh\n"
        f"session={shlex.quote('rac-' + safe_name(args.batch)[:60])}\n"
        "if tmux has-session -t \"$session\" 2>/dev/null; then echo \"already running: $session\" >&2; exit 1; fi\n"
        f"tmux new-session -d -s \"$session\" {shlex.quote(common + ' >> ' + str(plan_root / 'launcher.log') + ' 2>&1')}\n"
        "echo \"started tmux session: $session\"\n",
    )
    write_executable(
        plan_root / "status.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"status={shlex.quote(str(work_root / 'runs' / args.batch / 'status.json'))}\n"
        "if [[ -f \"$status\" ]]; then python3 -m json.tool \"$status\"; else echo 'not started'; fi\n",
    )
    write_executable(
        plan_root / "tail.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"touch {shlex.quote(str(plan_root / 'launcher.log'))}\n"
        f"tail -n 100 -F {shlex.quote(str(plan_root / 'launcher.log'))}\n",
    )
    (plan_root / "README.txt").write_text(
        "Fill the three sharednet/*.env files with distinct Room credentials.\n"
        "Run ./check.sh, then ./start.sh. R1, R2 and R3 run serially and each is scored.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "batch": args.batch,
        "plan_root": str(plan_root),
        "execution": str(execution_path),
        "image": EVO_IMAGE,
        "conditions": list(CONDITIONS),
        "episodes": len(episodes),
        "sharednet_files_to_fill": len(episodes),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
