from __future__ import annotations

import hashlib
import shutil
import tempfile
from fnmatch import fnmatch
from pathlib import Path

from .schemas import ArtifactRecord


IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", ".conda_env"}
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
SYSTEM_SIDE_EFFECTS = {
    "results/credentials_needed.json",
    "results/environment_setup.json",
    "results/setup_commands.log",
}


def _ignored_artifact(relative: Path) -> bool:
    return bool(
        IGNORED_PARTS.intersection(relative.parts)
        or (relative.parts and relative.parts[0] in PRIVATE_TOP_LEVEL)
        or relative.as_posix() in SYSTEM_SIDE_EFFECTS
    )


def artifact_kind(relative: Path) -> str:
    posix = relative.as_posix().lower()
    name = relative.name.lower()
    if posix == "report/report.md" or (relative.parts and relative.parts[0].lower() == "report" and relative.suffix.lower() in {".tex", ".pdf"}):
        return "terminal_report"
    if "literature_review" in name:
        return "literature"
    if "review" in name:
        return "review"
    if "literature" in name or "citation" in name or "survey" in name:
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
        if _ignored_artifact(relative):
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


class WorkspaceTransaction:
    """File-level transaction for one capability invocation.

    Generated environments and append-only host logs are deliberately outside
    the transaction. Scientific artifacts and host state are restored when a
    verifier rejects an invocation, so rejected work cannot contaminate later
    phases.
    """

    _PERSISTENT_PATTERNS = (".conda_env/**", "auto_research/logs/**")

    def __init__(self, workspace: Path | None):
        self.workspace = workspace.resolve() if workspace is not None else None
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self.backup: Path | None = None
        if self.workspace is not None:
            self._temporary = tempfile.TemporaryDirectory(prefix="rac-workspace-")
            self.backup = Path(self._temporary.name) / "before"
            self.backup.mkdir()
            self._copy_tree(self.workspace, self.backup, skip_persistent=True)

    @staticmethod
    def _matches(relative: Path, patterns: tuple[str, ...]) -> bool:
        posix = relative.as_posix()
        return any(fnmatch(posix, pattern) for pattern in patterns)

    def _copy_tree(self, source: Path, destination: Path, *, skip_persistent: bool = False) -> None:
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if skip_persistent and self._matches(relative, self._PERSISTENT_PATTERNS):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    def rollback(self, *, preserve_patterns: tuple[str, ...] = ()) -> None:
        if self.workspace is None or self.backup is None:
            return
        preserved_root = Path(self._temporary.name) / "preserved"  # type: ignore[union-attr]
        preserved_root.mkdir()
        for path in self.workspace.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.workspace)
            if self._matches(relative, preserve_patterns):
                target = preserved_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

        for path in sorted(self.workspace.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            relative = path.relative_to(self.workspace)
            if self._matches(relative, self._PERSISTENT_PATTERNS):
                continue
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass

        self._copy_tree(self.backup, self.workspace)
        self._copy_tree(preserved_root, self.workspace)

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
