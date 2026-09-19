import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from rac_ai_scientist.cli import _score_episode, _score_preflight


class ScorePreflightTests(unittest.TestCase):
    def test_requires_report_instructions_and_requested_evidence(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            (workspace / "report").mkdir()
            (workspace / "code").mkdir()
            (workspace / "outputs").mkdir()
            (workspace / "INSTRUCTIONS.md").write_text(
                "Persist executable analysis in code/, measurements in outputs/.",
                encoding="utf-8",
            )
            errors = _score_preflight(workspace)
            self.assertTrue(any("report" in item for item in errors))
            self.assertTrue(any("analysis artifact" in item for item in errors))

    def test_accepts_nonempty_report_and_analysis_artifact(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            (workspace / "report").mkdir()
            (workspace / "code").mkdir()
            (workspace / "report" / "report.md").write_text("# Result\n", encoding="utf-8")
            (workspace / "INSTRUCTIONS.md").write_text(
                "Persist executable analysis in code/, measurements in outputs/.",
                encoding="utf-8",
            )
            (workspace / "code" / "analysis.py").write_text("print('ok')\n", encoding="utf-8")
            self.assertEqual(_score_preflight(workspace), [])

    def test_preflight_failure_writes_null_score_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as raw:
            episode = Path(raw) / "episode"
            workspace = episode / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "INSTRUCTIONS.md").write_text("Research task\n", encoding="utf-8")
            (episode / "episode.json").write_text(
                json.dumps(
                    {
                        "episode_id": "ep",
                        "task_id": "Math_000",
                        "host": "auto_research_claw",
                        "condition": "N0",
                    }
                ),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                code = _score_episode(
                    argparse.Namespace(episode_dir=str(episode), benchmark=str(Path(raw) / "missing"))
                )
            result = json.loads((episode / "score.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 2)
            self.assertIsNone(result["total_score"])
            self.assertIn("missing or empty report", result["error"])


if __name__ == "__main__":
    unittest.main()
