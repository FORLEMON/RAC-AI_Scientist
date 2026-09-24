from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .base import TaskSpec, ScoreResult, InvalidSubmission, digest, read_json, write_json
from .registry import get_adapter


def score_episode(episode: Path, source: Path, *, dataset: Path | None = None, timeout: float = 1800) -> int:
    metadata = read_json(episode / "episode.json")
    spec_path = episode / "task_spec.json"  # immutable controller copy, never workspace copy
    result = ScoreResult(metadata.get("benchmark_id", "unknown"), metadata.get("task_id", "unknown"), "scorer_failed")
    score_path = episode / "score.json"
    def persist_status():
        score = read_json(score_path)
        metadata["scoring_status"] = score["status"]
        write_json(episode / "episode.json", metadata)
    try:
        spec = TaskSpec.from_dict(read_json(spec_path))
        if metadata.get("task_spec_sha256") != digest(spec_path):
            raise ValueError("frozen task spec does not match episode metadata")
        if metadata.get("benchmark_id") != spec.benchmark_id or metadata.get("task_id") != spec.task_id:
            raise ValueError("episode benchmark identity mismatch")
        submission = get_adapter(spec.benchmark_id).read_submission(episode / "workspace", spec)
    except InvalidSubmission as exc:
        result.status, result.error = "invalid_submission", str(exc)
        write_json(score_path, result.to_dict())
        persist_status()
        return 2
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        write_json(score_path, result.to_dict())
        persist_status()
        return 2
    try:
        from .upstream import assert_revision
        assert_revision(source, spec.benchmark_id)
        if spec.benchmark_id == "corebench" and dataset is None:
            raise ValueError("CORE scoring requires --dataset pointing to private reference JSON")
        scoring_dir = episode / "scoring"
        scoring_dir.mkdir(exist_ok=True)
        # Parsed JSON is copied out once; workers never read agent-controlled
        # paths, imports, Python files, or a mutable task_spec.json.
        write_json(scoring_dir / "submission.json", submission)
        write_json(score_path, result.to_dict())
        command = [sys.executable, "-B", "-m", "rac_ai_scientist.benchmarks.scoring",
            "--spec", str(spec_path), "--submission", str(scoring_dir / "submission.json"),
            "--source", str(source), "--output", str(score_path)]
        if dataset:
            command += ["--dataset", str(dataset)]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("AGENT_", "SHAREDNET_", "RAC_TASK_RUNTIME_"))}
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        with tempfile.TemporaryDirectory(prefix="rac-scorer-") as temporary:
            try:
                process = subprocess.run(command, cwd=temporary, env=env, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                for name, value in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", "replace")
                    (scoring_dir / f"worker.{name}.log").write_text(value or "", encoding="utf-8")
                raise
            (scoring_dir / "worker.stdout.log").write_text(process.stdout or "", encoding="utf-8")
            (scoring_dir / "worker.stderr.log").write_text(process.stderr or "", encoding="utf-8")
        if not score_path.is_file():
            raise RuntimeError(f"scorer worker exited {process.returncode} without a score: {process.stderr[-1000:]}")
        scored = read_json(score_path)
        if "wall_seconds" not in scored.get("provenance", {}):
            raise RuntimeError(f"scorer worker did not finish ({process.returncode}): {process.stderr[-1000:]}")
        persist_status()
        return 0 if process.returncode == 0 and scored.get("status") == "scored" else 2
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        write_json(score_path, result.to_dict())
        persist_status()
        return 2
