from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


class BenchmarkBoundaryError(RuntimeError):
    pass


def _sequence_profile(path: Path) -> dict | None:
    """Return factual summary data for the benchmark's MOT-style JSON input."""
    if path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list) or not payload:
        return None
    if not all(isinstance(row, dict) for row in payload):
        return None
    required = {"frame", "gt_ids", "detections"}
    if not all(required.issubset(row) for row in payload):
        return None

    frame_count = len({row["frame"] for row in payload})
    gt_counts = [len(row["gt_ids"]) for row in payload if isinstance(row["gt_ids"], list)]
    detection_counts = [len(row["detections"]) for row in payload if isinstance(row["detections"], list)]
    if len(gt_counts) != len(payload) or len(detection_counts) != len(payload):
        return None
    gt_instances = sum(gt_counts)
    detections = sum(detection_counts)
    distinct_gt_ids = len({item for row in payload for item in row["gt_ids"]})
    return {
        "frames": frame_count,
        "distinct_gt_ids": distinct_gt_ids,
        "gt_instances": gt_instances,
        "gt_per_frame_min": min(gt_counts),
        "gt_per_frame_max": max(gt_counts),
        "detections": detections,
        "detection_presence_percent": round(100.0 * detections / gt_instances, 3) if gt_instances else None,
    }


def _profile_summary(profile: dict) -> str:
    per_frame = (
        str(profile["gt_per_frame_min"])
        if profile["gt_per_frame_min"] == profile["gt_per_frame_max"]
        else f'{profile["gt_per_frame_min"]}-{profile["gt_per_frame_max"]}'
    )
    presence = profile["detection_presence_percent"]
    presence_text = f"{presence:.3f}%" if presence is not None else "unavailable"
    return (
        f'{profile["frames"]} frames; {profile["distinct_gt_ids"]} distinct GT IDs; '
        f'{per_frame} GT objects/frame; {profile["gt_instances"]} GT instances; '
        f'{profile["detections"]} detections; detection presence {presence_text}'
    )


def _claimed_numeric_profile(description: str) -> dict[str, float]:
    claims: dict[str, float] = {}
    patterns = {
        "frames": r"\b(\d+)\s*frames?\b",
        "objects": r"\b(\d+)\s*objects?\b",
        "detection_rate": r"\b(\d+(?:\.\d+)?)\s*%\s*detection(?:\s+rate|\s+presence)?\b",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, description, flags=re.IGNORECASE)
        if match:
            claims[name] = float(match.group(1))
    return claims


def _profile_mismatches(description: str, profile: dict) -> list[str]:
    claims = _claimed_numeric_profile(description)
    mismatches: list[str] = []
    if "frames" in claims and int(claims["frames"]) != profile["frames"]:
        mismatches.append(f'metadata frames={int(claims["frames"])}; file frames={profile["frames"]}')
    if "objects" in claims and not (
        profile["gt_per_frame_min"] == profile["gt_per_frame_max"] == int(claims["objects"])
    ):
        actual_objects = (
            str(profile["gt_per_frame_min"])
            if profile["gt_per_frame_min"] == profile["gt_per_frame_max"]
            else f'{profile["gt_per_frame_min"]}-{profile["gt_per_frame_max"]}'
        )
        mismatches.append(
            f'metadata objects/frame={int(claims["objects"])}; '
            f"file objects/frame={actual_objects}"
        )
    if "detection_rate" in claims and profile["detection_presence_percent"] is not None:
        if abs(claims["detection_rate"] - profile["detection_presence_percent"]) > 0.5:
            mismatches.append(
                f'metadata detection rate={claims["detection_rate"]:g}%; '
                f'file detection presence={profile["detection_presence_percent"]:.3f}%'
            )
    return mismatches


def _verified_data_items(task_dir: Path, raw_items: object) -> tuple[list, list[dict]]:
    items = list(raw_items) if isinstance(raw_items, list) else []
    findings: list[dict] = []
    verified: list = []
    for item in items:
        if not isinstance(item, dict):
            verified.append(item)
            continue
        updated = dict(item)
        relative = updated.get("path") or updated.get("name")
        candidate = task_dir / str(relative) if relative else None
        if candidate is not None and not candidate.is_file() and updated.get("name"):
            candidate = task_dir / "data" / str(updated["name"])
        if candidate is not None:
            try:
                relative_candidate = candidate.resolve().relative_to(task_dir.resolve())
            except (OSError, ValueError):
                candidate = None
            else:
                if not relative_candidate.parts or relative_candidate.parts[0] not in {"data", "related_work"}:
                    candidate = None
        profile = _sequence_profile(candidate) if candidate is not None and candidate.is_file() else None
        if profile is not None:
            summary = _profile_summary(profile)
            description = str(updated.get("description", ""))
            mismatches = _profile_mismatches(description, profile)
            updated["verified_profile"] = profile
            updated["description"] = f"Verified file contents (authoritative): {summary}."
            findings.append({
                "path": str(candidate.relative_to(task_dir)).replace("\\", "/"),
                "verified_profile": profile,
                "metadata_mismatches": mismatches,
            })
        verified.append(updated)
    return verified, findings


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
    data_items, validation_findings = _verified_data_items(task_dir, info.get("data", []))
    destination.mkdir(parents=True, exist_ok=False)
    for name, source in sources:
        shutil.copytree(source, destination / name)
    for name in ("code", "outputs", "report", "report/images", ".rac"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    (destination / "task.json").write_text(
        json.dumps({"task_id": task_id, "task": info.get("task", ""), "data": data_items}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (destination / "task_info.json").write_text(
        json.dumps({"task_id": task_id, "task": info.get("task", ""), "data": data_items}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    data_lines = "\n".join(f"- {item}" for item in data_items) or "- See data/ for supplied inputs."
    (destination / "INSTRUCTIONS.md").write_text(
        "# Research task\n\n"
        + str(info.get("task", "")).strip()
        + "\n\n# Available inputs\n\n"
        + data_lines
        + "\n\nUse only the visible task files, data/, and related_work/. "
        "Persist executable analysis in code/, measurements in outputs/, and the final report in report/report.md.\n",
        encoding="utf-8",
    )
    (destination / ".rac" / "input_validation.json").write_text(
        json.dumps({"task_id": task_id, "data": validation_findings}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (destination / ".rac" / "benchmark.json").write_text(
        json.dumps(
            {"schema_version": 1, "benchmark": "researchclawbench"},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if target.resolve() == destination or target.resolve() in destination.parents:
        raise BenchmarkBoundaryError("target_study leaked into episode workspace")
    return {**info, "data": data_items, "input_validation": validation_findings}


def assert_no_target_study(workspace: Path) -> None:
    leaked = [path for path in workspace.rglob("*") if "target_study" in path.parts]
    if leaked:
        raise BenchmarkBoundaryError(f"benchmark target material leaked into workspace: {leaked[0]}")
