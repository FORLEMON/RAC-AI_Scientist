from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
from typing import Any


class BoundaryError(ValueError):
    pass


class InvalidSubmission(ValueError):
    pass


def relative_path(value: str) -> str:
    """Portable relative paths, including when preparing Linux input on Windows."""
    path = PurePosixPath(value)
    if (not value or "\\" in value or path.is_absolute() or PureWindowsPath(value).drive
            or any(part in {"..", ".", ""} for part in value.split("/")) or ":" in value):
        raise BoundaryError(f"unsafe relative path: {value!r}")
    return path.as_posix()


def safe_file(root: Path, name: str) -> Path:
    name = relative_path(name)
    root = root.resolve()
    result = root / name
    for parent in (result, *result.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise BoundaryError(f"symlink in task input: {name}")
    if not result.resolve().is_relative_to(root):
        raise BoundaryError(f"path escapes task root: {name}")
    return result


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_json(path: Path) -> Any:
    def bad_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("JSON number is outside finite float range")
        return parsed
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=bad_constant,
                      parse_float=finite_float, object_pairs_hook=unique_pairs)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class TaskSpec:
    benchmark_id: str
    task_id: str
    split: str
    objective: str
    source_revision: str
    output_file: str
    public_files: dict[str, str] = field(default_factory=dict)
    questions: tuple[str, ...] = ()
    profile: str = "default"
    schema_version: int = 1

    def __post_init__(self):
        from .registry import BENCHMARK_IDS
        if self.benchmark_id not in BENCHMARK_IDS or self.schema_version != 1:
            raise ValueError("unsupported benchmark or TaskSpec version")
        if not self.task_id or not self.objective.strip() or not self.split:
            raise ValueError("task id, split and objective must be nonempty")
        expected = {"researchclawbench": "report/report.md", "discoverybench": "discovery_result.json", "corebench": "report.json"}
        if self.output_file != expected[self.benchmark_id]:
            raise ValueError("submission path differs from the benchmark protocol")
        for name, sha in self.public_files.items():
            relative_path(name)
            if any(part.lower() == ".env" or part.lower().startswith(".env.") for part in Path(name).parts) or name == "task_spec.json" or name.startswith(".git/"):
                raise BoundaryError(f"reserved private input path: {name}")
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("invalid input SHA256")

    @classmethod
    def from_dict(cls, payload: dict) -> "TaskSpec":
        payload = dict(payload)
        payload["questions"] = tuple(payload.get("questions", ()))
        return cls(**payload)

    def to_dict(self) -> dict:
        return asdict(self)

    def instructions(self) -> str:
        if self.benchmark_id == "discoverybench":
            delivery = (
                'Write discovery_result.json with nonempty string fields "hypothesis" and "workflow". '
                'An optional "evidence" list may contain {"claim": "...", "artifact": "outputs/..."}. '
                "Derive the hypothesis from the supplied data. State context, variables, relationships and uncertainty. "
                "Describe the analysis actually performed. Persist executable analysis in code/ and results in outputs/."
            )
        elif self.benchmark_id == "corebench":
            delivery = (
                "Install the capsule's dependencies in the task runtime, execute its code, and answer the task questions. "
                "Write report.json as a JSON object whose keys are EXACTLY the question strings below, and values are answers. "
                "Preserve capsule-relative paths. Do not seek reference answers or precomputed benchmark outputs.\n"
                + json.dumps(self.questions, ensure_ascii=False)
            )
        else:
            delivery = "Persist executable analysis in code/, measurements in outputs/, and the final report in report/report.md."
        return f"# Task\n\n{self.objective}\n\n# Delivery\n\n{delivery}\n\nA Markdown report is supplementary for JSON submissions. Never access hidden reference data.\n"


@dataclass
class ScoreResult:
    benchmark_id: str
    task_id: str
    status: str
    total_score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class BenchmarkAdapter(ABC):
    benchmark_id: str

    @abstractmethod
    def prepare(self, source: Path, destination: Path, **options) -> TaskSpec:
        """Build a new workspace from explicitly public source fields/files."""

    def read_submission(self, workspace: Path, spec: TaskSpec) -> Any:
        try:
            path = safe_file(workspace, spec.output_file)
            if not path.is_file() or path.stat().st_size == 0:
                raise InvalidSubmission(f"missing or empty {spec.output_file}")
            if path.stat().st_size > 8 * 1024 * 1024:
                raise InvalidSubmission("submission exceeds 8 MiB")
            value = read_json(path)
            if not isinstance(value, dict) or not value:
                raise InvalidSubmission("submission must be a nonempty JSON object")
            return value
        except (ValueError, OSError) as exc:
            raise InvalidSubmission(str(exc)) from exc


def finish_preparation(destination: Path, spec: TaskSpec) -> TaskSpec:
    for directory in ("code", "outputs", "report", "state", ".rac"):
        (destination / directory).mkdir(parents=True, exist_ok=True)
    (destination / "INSTRUCTIONS.md").write_text(spec.instructions(), encoding="utf-8")
    write_json(destination / ".rac" / "benchmark.json", {"schema_version": 1, "benchmark": spec.benchmark_id})
    files = {p.relative_to(destination).as_posix(): digest(p) for p in destination.rglob("*") if p.is_file()}
    spec = TaskSpec.from_dict({**spec.to_dict(), "public_files": files})
    write_json(destination / "task_spec.json", spec.to_dict())
    return spec


def load_prepared(workspace: Path, *, verify: bool = True) -> TaskSpec:
    spec = TaskSpec.from_dict(read_json(safe_file(workspace, "task_spec.json")))
    if verify:
        allowed = {*spec.public_files, "task_spec.json", ".env"}
        for path in workspace.rglob("*"):
            name = path.relative_to(workspace).as_posix()
            if path.is_symlink():
                raise BoundaryError(f"symlink in prepared workspace: {name}")
            if path.is_file() and name not in allowed:
                raise BoundaryError(f"undeclared file in prepared workspace: {name}")
        for name, sha in spec.public_files.items():
            if digest(safe_file(workspace, name)) != sha:
                raise BoundaryError(f"prepared input hash mismatch: {name}")
    return spec


def copy_prepared(source: Path, destination: Path, *, allow_empty: bool = False,
                  allow_empty_mounts: tuple[str, ...] = ()) -> TaskSpec:
    spec = load_prepared(source)
    if destination.resolve() == source.resolve() or destination.resolve().is_relative_to(source.resolve()):
        raise BoundaryError("episode workspace cannot be inside prepared inputs")
    if destination.exists():
        entries = list(destination.iterdir())
        pristine = all(p.name in allow_empty_mounts and not p.is_symlink() and p.is_dir()
                       and p.is_mount() and not any(p.iterdir()) for p in entries)
        if not allow_empty or not pristine:
            raise FileExistsError(f"episode workspace already exists or is nonempty: {destination}")
    if any(Path(name).parts[0] in allow_empty_mounts for name in spec.public_files):
        raise BoundaryError("prepared inputs cannot overwrite a host dependency mount")
    destination.mkdir(parents=True, exist_ok=allow_empty)
    for name in spec.public_files:
        target = safe_file(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_file(source, name), target)
    write_json(destination / "task_spec.json", spec.to_dict())
    for name in ("code", "outputs", "report", "state"):
        (destination / name).mkdir(exist_ok=True)
    return spec
