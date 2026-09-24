from pathlib import Path
from .base import BenchmarkAdapter, TaskSpec, InvalidSubmission
from ..benchmark import materialize_rcb_workspace, assert_no_target_study


class ResearchClawBenchAdapter(BenchmarkAdapter):
    benchmark_id = "researchclawbench"

    def prepare(self, source: Path, destination: Path, **options) -> TaskSpec:
        info = materialize_rcb_workspace(source, destination)
        assert_no_target_study(destination)
        return TaskSpec(self.benchmark_id, str(info.get("task_id") or source.name),
                        options.get("split") or "default", str(info["task"]),
                        options.get("revision") or "legacy", "report/report.md")

    def read_submission(self, workspace: Path, spec: TaskSpec):
        path = workspace / spec.output_file
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            raise InvalidSubmission("missing or empty report/report.md")
        return path.read_text(encoding="utf-8")
