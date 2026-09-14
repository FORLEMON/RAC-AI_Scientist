import tempfile
import unittest
from pathlib import Path

from rac_ai_scientist.artifacts import snapshot_workspace


class ArtifactTests(unittest.TestCase):
    def test_snapshot_classifies_terminal_report_and_is_stable(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "report").mkdir()
            (root / "report" / "report.md").write_text("result", encoding="utf-8")
            first = snapshot_workspace(root)
            second = snapshot_workspace(root)
            self.assertEqual(first, second)
            self.assertEqual(first[0].kind, "terminal_report")
