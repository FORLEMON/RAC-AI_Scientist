import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rac_ai_scientist.hosts.ark import ArkBridge


class ArkReportTests(unittest.TestCase):
    def test_placeholder_tex_does_not_become_a_terminal_report(self):
        with tempfile.TemporaryDirectory() as raw:
            report = Path(raw) / "report"
            report.mkdir()
            (report / "main.tex").write_text(
                "\\documentclass{article}\n\\begin{document}\nWork in progress.\n\\end{document}\n",
                encoding="utf-8",
            )
            bridge = object.__new__(ArkBridge)
            bridge.workspace = Path(raw)
            bridge._normalize_report()
            self.assertFalse((report / "report.md").exists())

    def test_latex_source_is_converted_to_markdown(self):
        with tempfile.TemporaryDirectory() as raw:
            report = Path(raw) / "report"
            report.mkdir()
            (report / "main.tex").write_text(
                "\\documentclass{article}\n\\begin{document}\n\\section{Results}\nMeasured tracking results.\n\\end{document}\n",
                encoding="utf-8",
            )
            bridge = object.__new__(ArkBridge)
            bridge.workspace = Path(raw)

            def convert(command, **kwargs):
                output = next(item.split("=", 1)[1] for item in command if item.startswith("--output="))
                (report / output).write_text("# Results\n\nMeasured tracking results.\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            with patch("subprocess.run", side_effect=convert):
                bridge._normalize_report()
            text = (report / "report.md").read_text(encoding="utf-8")
            self.assertIn("# Results", text)
            self.assertNotIn("\\documentclass", text)

    def test_incomplete_markdown_is_preserved_as_a_draft(self):
        with tempfile.TemporaryDirectory() as raw:
            report = Path(raw) / "report"
            report.mkdir()
            (report / "main.tex").write_text("Work in progress.", encoding="utf-8")
            (report / "report.md").write_text("# Research\n[TO BE WRITTEN by writer]\n", encoding="utf-8")
            bridge = object.__new__(ArkBridge)
            bridge.workspace = Path(raw)
            bridge.hop = 0
            bridge._normalize_report()
            self.assertFalse((report / "report.md").exists())
            draft = Path(raw) / "state" / "ark" / "report_draft_0.md"
            self.assertIn("TO BE WRITTEN", draft.read_text(encoding="utf-8"))

    def test_failed_conversion_preserves_existing_report_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as raw:
            report = Path(raw) / "report"
            report.mkdir()
            (report / "main.tex").write_text("\\section{Updated results}", encoding="utf-8")
            existing = report / "report.md"
            existing.write_text("# Previously valid report\n", encoding="utf-8")
            bridge = object.__new__(ArkBridge)
            bridge.workspace = Path(raw)

            def failed_convert(command, **kwargs):
                (report / ".report.md.tmp").write_text("partial", encoding="utf-8")
                raise subprocess.CalledProcessError(1, command)

            with patch("subprocess.run", side_effect=failed_convert):
                with self.assertRaises(subprocess.CalledProcessError):
                    bridge._normalize_report()

            self.assertEqual(existing.read_text(encoding="utf-8"), "# Previously valid report\n")
            self.assertFalse((report / ".report.md.tmp").exists())


if __name__ == "__main__":
    unittest.main()
