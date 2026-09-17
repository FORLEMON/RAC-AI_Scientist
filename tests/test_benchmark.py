import json
import tempfile
import unittest
from pathlib import Path

from rac_ai_scientist.benchmark import BenchmarkBoundaryError, assert_no_target_study, materialize_rcb_workspace


class BenchmarkTests(unittest.TestCase):
    def test_materializer_excludes_target_study(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            task = tmp_path / "tasks" / "Demo_000"
            (task / "data").mkdir(parents=True)
            (task / "related_work").mkdir()
            (task / "target_study").mkdir()
            (task / "data" / "x.csv").write_text("x\n1\n", encoding="utf-8")
            (task / "target_study" / "paper.pdf").write_bytes(b"secret")
            (task / ".env").write_text("SHAREDNET_INVITE=private", encoding="utf-8")
            (task / "task_info.json").write_text(json.dumps({"task": "demo", "data": []}), encoding="utf-8")
            workspace = tmp_path / "runs" / "ep"
            materialize_rcb_workspace(task, workspace)
            self.assertTrue((workspace / "data" / "x.csv").is_file())
            sanitized = json.loads((workspace / "task_info.json").read_text(encoding="utf-8"))
            self.assertEqual(sanitized["task_id"], "Demo_000")
            instructions = (workspace / "INSTRUCTIONS.md").read_text(encoding="utf-8")
            self.assertIn("demo", instructions)
            self.assertNotIn("target_study", instructions)
            self.assertFalse((workspace / "target_study").exists())
            self.assertFalse((workspace / ".env").exists())
            assert_no_target_study(workspace)

    def test_leak_check_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            (tmp_path / "target_study").mkdir()
            with self.assertRaises(BenchmarkBoundaryError):
                assert_no_target_study(tmp_path)

    def test_materializer_rejects_symlinked_inputs(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            task = tmp_path / "tasks" / "Demo_001"
            (task / "data").mkdir(parents=True)
            secret = task / "target_study"
            secret.mkdir()
            (secret / "answer.txt").write_text("hidden", encoding="utf-8")
            (task / "task_info.json").write_text(json.dumps({"task": "demo"}), encoding="utf-8")
            try:
                (task / "data" / "escape.txt").symlink_to(secret / "answer.txt")
            except OSError:
                self.skipTest("symbolic links are unavailable on this platform")
            with self.assertRaises(BenchmarkBoundaryError):
                materialize_rcb_workspace(task, tmp_path / "runs" / "ep")
