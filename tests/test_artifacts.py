import tempfile
import unittest
from pathlib import Path

from rac_ai_scientist.artifacts import WorkspaceTransaction, snapshot_workspace


class ArtifactTests(unittest.TestCase):
    def test_snapshot_ignores_generated_conda_environment(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            generated = root / ".conda_env" / "lib" / "python" / "site-packages"
            generated.mkdir(parents=True)
            (generated / "numpy.py").write_text("generated dependency", encoding="utf-8")
            outputs = root / "outputs"
            outputs.mkdir()
            (outputs / "result.json").write_text("{}", encoding="utf-8")

            records = snapshot_workspace(root)

            self.assertEqual([item.relative_path for item in records], ["outputs/result.json"])

    def test_snapshot_classifies_terminal_report_and_is_stable(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "report").mkdir()
            (root / "report" / "report.md").write_text("result", encoding="utf-8")
            first = snapshot_workspace(root)
            second = snapshot_workspace(root)
            self.assertEqual(first, second)
            self.assertEqual(first[0].kind, "terminal_report")

    def test_snapshot_classifies_survey_as_literature(self):
        with tempfile.TemporaryDirectory() as raw:
            survey = Path(raw) / "state" / "ai_researcher" / "survey.md"
            survey.parent.mkdir(parents=True)
            survey.write_text("literature review", encoding="utf-8")

            records = snapshot_workspace(Path(raw))

            self.assertEqual(records[0].kind, "literature")

    def test_snapshot_ignores_openhands_environment_bookkeeping(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            results = root / "results"
            results.mkdir()
            (results / "environment_setup.json").write_text("{}", encoding="utf-8")
            (results / "credentials_needed.json").write_text("{}", encoding="utf-8")
            (results / "setup_commands.log").write_text("setup", encoding="utf-8")
            (results / "scientific_result.json").write_text("{}", encoding="utf-8")

            records = snapshot_workspace(root)

            self.assertEqual([item.relative_path for item in records], ["results/scientific_result.json"])

    def test_snapshot_ignores_runtime_cache_and_provision_log(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cache = root / ".cache" / "pip"
            cache.mkdir(parents=True)
            (cache / "download").write_text("generated", encoding="utf-8")
            (root / ".env_provision.log").write_text("generated", encoding="utf-8")
            (root / "AGENTS.md").write_text("runtime instructions", encoding="utf-8")
            results = root / "results"
            results.mkdir()
            (results / "measurement.json").write_text("{}", encoding="utf-8")

            records = snapshot_workspace(root)

            self.assertEqual([item.relative_path for item in records], ["results/measurement.json"])
            self.assertEqual(records[0].kind, "result")

    def test_workspace_transaction_rolls_back_rejected_changes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "state").mkdir()
            original = root / "state" / "plan.md"
            original.write_text("before", encoding="utf-8")
            transaction = WorkspaceTransaction(root)
            original.write_text("after", encoding="utf-8")
            (root / "unauthorized").write_text("bad", encoding="utf-8")

            transaction.rollback()
            transaction.close()

            self.assertEqual(original.read_text(encoding="utf-8"), "before")
            self.assertFalse((root / "unauthorized").exists())

    def test_workspace_transaction_recovery_preserves_only_declared_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            root.mkdir(exist_ok=True)
            transaction = WorkspaceTransaction(root)
            (root / "outputs").mkdir()
            (root / "outputs" / "partial.json").write_text("{}", encoding="utf-8")
            (root / "outside.txt").write_text("bad", encoding="utf-8")

            transaction.rollback(preserve_patterns=("outputs/**",))
            transaction.close()

            self.assertTrue((root / "outputs" / "partial.json").is_file())
            self.assertFalse((root / "outside.txt").exists())
