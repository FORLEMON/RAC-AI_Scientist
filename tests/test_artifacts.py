import tempfile
import unittest
from pathlib import Path

from rac_ai_scientist.artifacts import snapshot_workspace


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
