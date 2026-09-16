from __future__ import annotations

import json
import shutil
from pathlib import Path


class BenchmarkBoundaryError(RuntimeError):
    pass


def _assert_safe_input_tree(source: Path) -> None:
    for path in source.rglob("*"):
        if path.is_symlink():
            raise BenchmarkBoundaryError(f"symbolic links are not allowed in host-visible benchmark input: {path}")
        try:
            path.resolve().relative_to(source.resolve())
        except ValueError as exc:
            raise BenchmarkBoundaryError(f"benchmark input escapes its declared directory: {path}") from exc


def materialize_rcb_workspace(task_dir: Path, destination: Path) -> dict:
    """Copy only host-visible ResearchClawBench inputs into a fresh workspace."""
    task_dir = task_dir.resolve()
    destination = destination.resolve()
    target = task_dir / "target_study"
    if destination == task_dir or task_dir in destination.parents:
        raise BenchmarkBoundaryError("episode workspace cannot be inside the benchmark task directory")
    if not (task_dir / "task_info.json").is_file():
        raise BenchmarkBoundaryError(f"missing task_info.json under {task_dir}")

    info = json.loads((task_dir / "task_info.json").read_text(encoding="utf-8"))
    task_id = str(info.get("task_id") or task_dir.name)
    sources: list[tuple[str, Path]] = []
    for name in ("data", "related_work"):
        source = task_dir / name
        if source.exists():
            _assert_safe_input_tree(source)
            sources.append((name, source))
    destination.mkdir(parents=True, exist_ok=False)
    for name, source in sources:
        shutil.copytree(source, destination / name)
    for name in ("code", "outputs", "report", "report/images", ".rac"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    (destination / "task.json").write_text(
        json.dumps({"task_id": task_id, "task": info.get("task", ""), "data": info.get("data", [])}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (destination / "task_info.json").write_text(
        json.dumps({"task_id": task_id, "task": info.get("task", ""), "data": info.get("data", [])}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    data_items = "\n".join(f"- {item}" for item in info.get("data", [])) or "- See data/ for supplied inputs."
    (destination / "INSTRUCTIONS.md").write_text(
        "# Research task\n\n"
        + str(info.get("task", "")).strip()
        + "\n\n# Available inputs\n\n"
        + data_items
        + "\n\nUse only the visible task files, data/, and related_work/. "
        "Persist executable analysis in code/, measurements in outputs/, and the final report in report/report.md.\n",
        encoding="utf-8",
    )
    if target.resolve() == destination or target.resolve() in destination.parents:
        raise BenchmarkBoundaryError("target_study leaked into episode workspace")
    return info


def assert_no_target_study(workspace: Path) -> None:
    leaked = [path for path in workspace.rglob("*") if "target_study" in path.parts]
    if leaked:
        raise BenchmarkBoundaryError(f"benchmark target material leaked into workspace: {leaked[0]}")
