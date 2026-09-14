from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from .benchmark import assert_no_target_study, materialize_rcb_workspace
from .artifacts import snapshot_workspace
from .config import load_config, validate_config
from .ledger import JsonlLedger, config_hash
from .matrix import expand_matrix
from .runner import EpisodeRunner
from .schemas import Budget, to_jsonable
from .provenance import tree_hash


def _doctor(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    data = load_config(config_path)
    findings = validate_config(data, config_path.parent.parent)
    for finding in findings:
        print(f"[{finding.level}] {finding.message}")
    return 1 if any(item.level == "ERROR" for item in findings) else 2 if any(item.level == "BLOCKED" for item in findings) else 0


def _plan(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    data = load_config(config_path)
    findings = validate_config(data, config_path.parent.parent)
    blockers = [item.message for item in findings if item.level in {"ERROR", "BLOCKED"}]
    if blockers:
        raise ValueError("configuration is not runnable: " + "; ".join(blockers))
    output = Path(args.output).resolve() if args.output else config_path.with_suffix(".plan.jsonl")
    if output.exists() and not args.force:
        raise FileExistsError(f"plan already exists: {output}; pass --force to replace it")
    rows = list(expand_matrix(data, config_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"plan": str(output), "episodes": len(rows)}, indent=2))
    return 0


def _check_workspace(args: argparse.Namespace) -> int:
    assert_no_target_study(Path(args.workspace).resolve())
    print("[OK] no ResearchClawBench target_study material is visible")
    return 0


def _prepare_task(args: argparse.Namespace) -> int:
    source = Path(args.task_dir).resolve()
    destination = Path(args.output).resolve()
    if destination.exists():
        raise FileExistsError(f"prepared task already exists: {destination}")
    materialize_rcb_workspace(source, destination)
    assert_no_target_study(destination)
    print(json.dumps({"prepared_task": str(destination), "source_task": source.name}, indent=2))
    return 0


def _score_episode(args: argparse.Namespace) -> int:
    episode_dir = Path(args.episode_dir).resolve()
    workspace = episode_dir / "workspace"
    metadata_path = episode_dir / "episode.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing episode metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert_no_target_study(workspace)
    report = workspace / "report" / "report.md"
    score_path = episode_dir / "score.json"
    if not report.is_file() or not report.read_text(encoding="utf-8", errors="replace").strip():
        result = {"task_id": metadata.get("task_id"), "total_score": 0.0, "error": "No report found in workspace"}
        score_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 2
    benchmark = Path(args.benchmark).resolve()
    if not (benchmark / "evaluation" / "score.py").is_file():
        raise FileNotFoundError(f"ResearchClawBench checkout not found: {benchmark}")
    (workspace / "_meta.json").write_text(
        json.dumps(
            {
                "run_id": metadata["episode_id"],
                "task_id": metadata["task_id"],
                "agent_name": f"{metadata['host']}+{metadata['condition']}",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    sys.path.insert(0, str(benchmark))
    from evaluation.score import score_workspace

    result = score_workspace(workspace)
    score_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if "error" in result else 0


def _resolve_upstream(root: Path, host: str) -> Path:
    local_names = {
        "ark": "ARK",
        "agent_laboratory": "AgentLaboratory-main",
        "data_to_paper": "data-to-paper-main",
    }
    candidates = (root / local_names[host], root / "upstreams" / host)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no checkout found for {host}; checked: " + ", ".join(str(item) for item in candidates))


def _selected_upstream(root: Path, host: str, explicit: str | None) -> Path:
    selected = explicit or os.environ.get("RAC_HOST_ROOT")
    if selected:
        upstream = Path(selected).resolve()
        if not upstream.exists():
            raise FileNotFoundError(f"explicit host checkout does not exist: {upstream}")
        return upstream
    return _resolve_upstream(root, host)


def _run_one(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    task_dir = Path(args.task_dir).resolve()
    raw_run_root = Path(args.run_root)
    run_root = (root / raw_run_root).resolve() if not raw_run_root.is_absolute() else raw_run_root.resolve()
    model = args.model or os.environ.get("AGENT_MODEL_NAME")
    api_key = os.environ.get("AGENT_API_KEY")
    if not model or not api_key:
        raise ValueError("set AGENT_MODEL_NAME and AGENT_API_KEY (or pass --model) before a live run")
    budget = Budget(
        args.max_cost_usd,
        args.max_input_tokens,
        args.max_output_tokens,
        args.max_agent_calls,
        args.max_wall_seconds,
        args.max_hops,
    )
    if budget.exhausted():
        raise ValueError("all lifecycle budget limits must be positive")
    manifest = root / "configs" / "hosts" / f"{args.host}.json"
    upstream = _selected_upstream(root, args.host, args.upstream)
    info_path = task_dir / "task_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing ResearchClawBench task_info.json: {info_path}")
    preview = json.loads(info_path.read_text(encoding="utf-8"))
    task_id = str(preview.get("task_id") or task_dir.name)
    episode_id = args.episode_id or f"{task_id}_{args.host}_{args.condition}_s{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    episode_dir = run_root / episode_id
    workspace = episode_dir / "workspace"
    if episode_dir.exists():
        raise FileExistsError(f"episode directory already exists: {episode_dir}")
    objective = str(preview.get("task", "")).strip()
    if not objective:
        raise ValueError("ResearchClawBench task has an empty objective")
    episode_dir.mkdir(parents=True)
    materialize_rcb_workspace(task_dir, workspace)
    assert_no_target_study(workspace)
    if args.host == "ark":
        from .hosts.ark import ArkBridge

        bridge = ArkBridge(upstream, manifest, budget, model, api_key)
    elif args.host == "agent_laboratory":
        from .hosts.agent_laboratory import AgentLaboratoryBridge

        bridge = AgentLaboratoryBridge(upstream, manifest, budget, model, api_key)
    else:
        from .hosts.data_to_paper import DataToPaperBridge

        bridge = DataToPaperBridge(upstream, manifest, budget, model, api_key)
    run_config = {
        "host": args.host,
        "condition": args.condition,
        "task_id": task_id,
        "seed": args.seed,
        "model": model,
        "budget": to_jsonable(budget),
        "review_score_threshold": args.review_score_threshold,
    }
    metadata = {
        "schema_version": 1,
        "episode_id": episode_id,
        "task_id": task_id,
        "host": args.host,
        "condition": args.condition,
        "seed": args.seed,
        "model": model,
        "budget": to_jsonable(budget),
        "run_config_sha256": config_hash(run_config),
        "workspace_input_sha256": config_hash(to_jsonable(snapshot_workspace(workspace))),
        "manifest_sha256": config_hash(json.loads(manifest.read_text(encoding="utf-8"))),
        "status": "initializing",
    }
    metadata_path = episode_dir / "episode.json"
    lock_path = root / "upstream.lock.json"
    source_mismatch = False
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        spec = lock.get("upstreams", {}).get(args.host, {})
        actual_tree, file_count, byte_count = tree_hash(upstream)
        expected_tree = spec.get("tree_sha256")
        source_mismatch = bool(expected_tree and actual_tree != expected_tree)
        metadata["upstream"] = {
            "revision": spec.get("revision"),
            "expected_tree_sha256": expected_tree,
            "actual_tree_sha256": actual_tree,
            "file_count": file_count,
            "byte_count": byte_count,
        }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        if source_mismatch:
            raise ValueError(f"{args.host} checkout does not match upstream.lock.json")
        bridge.initialize(episode_id=episode_id, workspace=workspace, objective=objective, seed=args.seed)
        outcome = EpisodeRunner(
            bridge,
            args.condition,
            JsonlLedger(episode_dir / "coordination.jsonl"),
            review_score_threshold=args.review_score_threshold,
        ).run(hard_hop_limit=args.max_hops)
        metadata.update({"status": outcome.status, "hops": outcome.hops, "reason": outcome.reason})
    except Exception as exc:
        metadata.update({"status": "failed", "error_type": type(exc).__name__, "reason": str(exc)})
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0 if outcome.status in {"completed", "stop"} else 2


def _verify_upstreams(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    lock = json.loads((root / "upstream.lock.json").read_text(encoding="utf-8"))
    failed = False
    for name, spec in lock["upstreams"].items():
        if args.only and name not in args.only:
            continue
        candidates = (root / spec["local_snapshot"], root / "upstreams" / name)
        source = next((item for item in candidates if item.exists()), None)
        if source is None:
            print(f"[MISSING] {name}")
            failed = True
            continue
        digest, count, size = tree_hash(source)
        expected = spec.get("tree_sha256")
        state = "OK" if expected == digest else "MISMATCH"
        print(f"[{state}] {name}: {digest} files={count} bytes={size}")
        failed = failed or state != "OK"
    return 1 if failed else 0


def _doctor_host(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    upstream = _selected_upstream(root, args.host, args.upstream)
    requirements = {
        "ark": (("yaml", "litellm"), ("openhands",)),
        "agent_laboratory": (("openai", "torch", "yaml", "pypdf"), ()),
        "data_to_paper": (("openai", "pandas", "PySide6"), ("pdflatex",)),
    }
    import shutil

    missing_modules = [name for name in requirements[args.host][0] if importlib.util.find_spec(name) is None]
    missing_commands = [name for name in requirements[args.host][1] if shutil.which(name) is None]
    print(f"[OK] upstream: {upstream}")
    for name in missing_modules:
        print(f"[BLOCKED] missing Python module: {name}")
    for name in missing_commands:
        print(f"[BLOCKED] missing executable: {name}")
    if missing_modules or missing_commands:
        return 2
    print("[OK] host runtime prerequisites are discoverable")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rac-ai-scientist")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="validate configuration without model or network calls")
    doctor.add_argument("--config", required=True)
    doctor.set_defaults(func=_doctor)
    plan = sub.add_parser("plan", help="expand a validated experiment matrix without model calls")
    plan.add_argument("--config", required=True)
    plan.add_argument("--output")
    plan.add_argument("--force", action="store_true")
    plan.set_defaults(func=_plan)
    check = sub.add_parser("check-workspace", help="fail if target_study leaked into a host workspace")
    check.add_argument("workspace")
    check.set_defaults(func=_check_workspace)
    prepare = sub.add_parser("prepare-task", help="create a host-safe task bundle without target_study")
    prepare.add_argument("--task-dir", required=True)
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(func=_prepare_task)
    run = sub.add_parser("run-one", help="run one host/condition/task episode")
    run.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    run.add_argument("--host", required=True, choices=("ark", "agent_laboratory", "data_to_paper"))
    run.add_argument("--upstream", help="explicit host checkout (or set RAC_HOST_ROOT)")
    run.add_argument("--condition", required=True, choices=("N0", "R1", "R2", "R3", "R4", "R5"))
    run.add_argument("--task-dir", required=True)
    run.add_argument("--run-root", default="runs")
    run.add_argument("--episode-id")
    run.add_argument("--model")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--max-cost-usd", type=float, required=True)
    run.add_argument("--max-input-tokens", type=int, required=True)
    run.add_argument("--max-output-tokens", type=int, required=True)
    run.add_argument("--max-agent-calls", type=int, required=True)
    run.add_argument("--max-wall-seconds", type=float, required=True)
    run.add_argument("--max-hops", type=int, required=True)
    run.add_argument("--review-score-threshold", type=float, default=8.0)
    run.set_defaults(func=_run_one)
    verify = sub.add_parser("verify-upstreams", help="verify local source snapshots against frozen tree hashes")
    verify.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    verify.add_argument("--only", action="append", default=[])
    verify.set_defaults(func=_verify_upstreams)
    host = sub.add_parser("doctor-host", help="check one host environment without model calls")
    host.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    host.add_argument("--host", required=True, choices=("ark", "agent_laboratory", "data_to_paper"))
    host.add_argument("--upstream", help="explicit host checkout (or set RAC_HOST_ROOT)")
    host.set_defaults(func=_doctor_host)
    score = sub.add_parser("score-episode", help="score a finished episode outside the evaluated host")
    score.add_argument("--episode-dir", required=True)
    score.add_argument("--benchmark", required=True)
    score.set_defaults(func=_score_episode)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
