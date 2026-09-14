from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .conditions import Condition


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DoctorFinding:
    level: str
    message: str


def load_config(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON experiment config: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("experiment config must be a JSON object")
    return data


def validate_config(data: dict[str, Any], root: Path) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    try:
        conditions = [Condition.parse(item) for item in data.get("conditions", [])]
    except ValueError as exc:
        findings.append(DoctorFinding("ERROR", str(exc)))
        conditions = []
    if not conditions:
        findings.append(DoctorFinding("ERROR", "at least one condition is required"))

    hosts = data.get("hosts", [])
    allowed_hosts = {"ark", "agent_laboratory", "data_to_paper"}
    unknown = sorted(set(hosts) - allowed_hosts)
    if unknown:
        findings.append(DoctorFinding("ERROR", f"unknown hosts: {', '.join(unknown)}"))

    tasks = data.get("tasks", [])
    if not tasks:
        findings.append(DoctorFinding("BLOCKED", "task list is empty; freeze CPU-feasible tasks before live runs"))
    budget = data.get("budget", {})
    nonpositive = [key for key, value in budget.items() if not isinstance(value, (int, float)) or value <= 0]
    if nonpositive:
        findings.append(DoctorFinding("BLOCKED", "live budget is unset: " + ", ".join(sorted(nonpositive))))
    model = data.get("model", {})
    if model.get("name") in (None, "", "SET_ME"):
        findings.append(DoctorFinding("BLOCKED", "evaluated model is not configured"))

    paths = data.get("paths", {})
    for key in ("benchmark", "upstream_root"):
        raw = paths.get(key)
        if raw and not (root / raw).exists():
            findings.append(DoctorFinding("BLOCKED", f"{key} path does not exist: {raw}"))
    if not findings:
        findings.append(DoctorFinding("OK", "configuration is ready for a dry-run"))
    return findings


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
