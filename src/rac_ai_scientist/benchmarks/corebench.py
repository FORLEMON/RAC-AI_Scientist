from __future__ import annotations

from pathlib import Path
import shutil

from .base import BenchmarkAdapter, BoundaryError, InvalidSubmission, TaskSpec, finish_preparation, read_json, safe_file
from .upstream import core_harness

REVISION = "16bb03ebc11577fb5ea6dc8bb6c968387085e6aa"


def reference(dataset: Path, task_id: str) -> dict:
    data = read_json(dataset)
    if not isinstance(data, list):
        raise ValueError("CORE dataset must be a JSON array")
    matches = [item for item in data if item.get("capsule_id") == task_id]
    if len(matches) != 1:
        raise ValueError("capsule id missing or duplicated in private dataset")
    task = matches[0]
    results = task.get("results")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict) or not results[0]:
        raise ValueError("CORE reference has no questions")
    keys = set(results[0])
    for row in results:
        if not isinstance(row, dict) or set(row) != keys:
            raise ValueError("inconsistent reference question keys")
        for key, value in row.items():
            if not isinstance(value, (int, float, str, list)) or isinstance(value, bool):
                raise ValueError(f"unsupported reference type for {key}")
            if type(value) != type(results[0][key]) and not (isinstance(value, (int, float)) and isinstance(results[0][key], (int, float))):
                raise ValueError("inconsistent reference types")
    if len(results) < 2 and any(isinstance(v, (int, float)) for v in results[0].values()):
        raise ValueError("numeric CORE scoring requires at least two reference runs")
    return task


class CoreBenchAdapter(BenchmarkAdapter):
    benchmark_id = "corebench"

    def prepare(self, source: Path, destination: Path, **options) -> TaskSpec:
        task_id = options["task_id"]
        # split is a local, frozen partition of HAL's test set, not an upstream
        # training split. Report it explicitly in provenance.
        split = options["split"]
        if split not in {"dev", "eval"}:
            raise ValueError("CORE local partition must be dev or eval")
        task = reference(Path(options["dataset"]), task_id)
        if "gpu" in task:
            raise ValueError("initial CORE profile is CPU-only; GPU capsule rejected")
        capsule = safe_file(Path(options["capsules"]), task_id)
        if not capsule.is_dir():
            raise FileNotFoundError(capsule)
        for path in capsule.rglob("*"):
            if path.is_symlink():
                raise BoundaryError("capsules containing symlinks are not supported")
        harness = core_harness(source)
        files = harness._get_capsule_files_dict(str(capsule))
        if not files:
            raise ValueError("capsule has no public files after official Hard filtering")
        destination.mkdir(parents=True, exist_ok=False)
        reserved = {"task_spec.json", "INSTRUCTIONS.md", "report.json", ".env"}
        for source_name in files.values():
            name = Path(source_name).relative_to(capsule).as_posix()
            if name in reserved or name.startswith((".rac/", ".git/")):
                raise BoundaryError(f"capsule collides with harness metadata: {name}")
            target = safe_file(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(safe_file(capsule, name), target)
        spec = TaskSpec(self.benchmark_id, task_id, split, harness._construct_prompt(task),
                        REVISION, "report.json", questions=tuple(task["results"][0]), profile="hal-hard-cpu")
        return finish_preparation(destination, spec)

    def read_submission(self, workspace: Path, spec: TaskSpec):
        value = super().read_submission(workspace, spec)
        if set(value) != set(spec.questions):
            raise InvalidSubmission("report.json keys must match the exact CORE question strings")
        if any(v is None or isinstance(v, (dict, bool)) for v in value.values()):
            raise InvalidSubmission("CORE answers must be numbers, strings, or lists")
        return value
