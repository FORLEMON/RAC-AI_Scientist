from __future__ import annotations

import hashlib
from pathlib import Path

from .schemas import ArtifactRecord


IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv"}
PRIVATE_TOP_LEVEL = {
    ".ark",
    ".rac",
    "agent_laboratory",
    "state_saves",
    "d2p_native",
    "auto_research",
    "ai_researcher_native",
    "evo_scientist_native",
    "auto_research_claw_native",
}


def artifact_kind(relative: Path) -> str:
    posix = relative.as_posix().lower()
    name = relative.name.lower()
    if posix == "report/report.md" or (relative.parts and relative.parts[0].lower() == "report" and relative.suffix.lower() in {".tex", ".pdf"}):
        return "terminal_report"
    if "review" in name:
        return "review"
    if "literature" in name or "citation" in name:
        return "literature"
    if "interpretation" in name or "analysis" in name:
        return "analysis"
    if "plan" in name or "idea" in name or "goal" in name:
        return "plan"
    if relative.parts and relative.parts[0].lower() == "code":
        return "code"
    if relative.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"} and "report" in [part.lower() for part in relative.parts]:
        return "figure"
    if relative.parts and relative.parts[0].lower() == "outputs":
        return "result"
    return "state"


def snapshot_workspace(workspace: Path) -> list[ArtifactRecord]:
    workspace = workspace.resolve()
    records: list[ArtifactRecord] = []
    for path in sorted(workspace.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace)
        if IGNORED_PARTS.intersection(relative.parts) or (relative.parts and relative.parts[0] in PRIVATE_TOP_LEVEL):
            continue
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            size = path.stat().st_size
        except OSError:
            continue
        relative_path = relative.as_posix()
        records.append(
            ArtifactRecord(
                artifact_id=hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:20],
                relative_path=relative_path,
                kind=artifact_kind(relative),
                sha256=digest.hexdigest(),
                size_bytes=size,
            )
        )
    return records
