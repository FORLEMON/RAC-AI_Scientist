import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rac_ai_scientist.hosts.ark import ArkBridge
from rac_ai_scientist.schemas import Budget, CapabilityCard, CoordinationDecision, Action


class ArkReportTests(unittest.TestCase):
    def test_markdown_export_removes_only_ark_page_count_probe(self):
        marker = (r"\makeatletter\pdfsavepos\write\@auxout{\string\gdef\string\arkBodyEndY{\the\pdflastypos}"
                  r"\string\gdef\string\arkPageH{\number\pdfpageheight}"
                  r"\string\gdef\string\arkBodyEndPage{\arabic{page}}}\makeatother")
        original = "\\section{Results}\nMeasured result.\n" + marker + "\n\\clearpage\n\\bibliography{references}\n"
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw) / "main.tex"
            source.write_text(original, encoding="utf-8")
            converted = ArkBridge._markdown_latex_source(source)
            self.assertEqual(converted, original.replace(marker, ""))
            self.assertEqual(source.read_text(encoding="utf-8"), original)

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
            self.assertFalse((report / ".report.source.tex").exists())

    def test_markdown_source_preserves_citations_and_manual_bibliography(self):
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw) / "main.tex"
            source.write_text(
                "\\begin{document}\nEvidence~\\cite{paper_a,paper_b}.\n"
                "\\begin{thebibliography}{9}\n"
                "\\bibitem{paper_a} Author A. First paper.\n"
                "\\bibitem{paper_b} Author B. Second paper.\n"
                "\\end{thebibliography}\n\\end{document}\n",
                encoding="utf-8",
            )

            converted = ArkBridge._markdown_latex_source(source)

            self.assertIn("[1, 2]", converted)
            self.assertIn("\\section*{References}", converted)
            self.assertIn("\\item  Author A", converted)
            self.assertNotIn("\\cite", converted)

    def test_reviewer_reads_full_persisted_review(self):
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw) / "auto_research" / "state"
            state.mkdir(parents=True)
            (state / "latest_review.md").write_text("Total = 7.7 / 10", encoding="utf-8")
            bridge = object.__new__(ArkBridge)
            bridge.workspace = Path(raw)
            self.assertEqual(bridge._review_text("summary only"), "Total = 7.7 / 10")
            self.assertEqual(
                bridge._review_text("new summary", previous="Total = 7.7 / 10"),
                "new summary",
            )

    def test_stale_rendered_pages_are_removed_before_review(self):
        with tempfile.TemporaryDirectory() as raw:
            report = Path(raw) / "report"
            report.mkdir()
            (report / "page_01.png").write_bytes(b"one")
            (report / "page_14.png").write_bytes(b"stale")
            bridge = object.__new__(ArkBridge)
            bridge.workspace = Path(raw)
            bridge._clear_rendered_pages()
            self.assertEqual(list(report.glob("page_*.png")), [])

    def test_reviewer_transition_is_committed_only_after_acceptance(self):
        class Orchestrator:
            _agent_stats = []

            def compile_latex(self):
                pass

            def run_agent(self, capability_id, prompt, timeout):
                review = workspace / "auto_research" / "state" / "latest_review.md"
                review.parent.mkdir(parents=True, exist_ok=True)
                review.write_text(
                    "Total = 7.7 / 10\nMajor issue: missing robustness analysis.",
                    encoding="utf-8",
                )
                return "Saved full review to disk."

        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            bridge = object.__new__(ArkBridge)
            bridge.workspace = workspace
            bridge.orchestrator = Orchestrator()
            bridge.cards = [CapabilityCard("reviewer", "review", ("terminal_review",), (), ("state/**",), ("review",))]
            bridge.initial_budget = Budget(10, 10000, 10000, 10, 100, 10)
            bridge.started = time.monotonic()
            bridge.objective = "review"
            bridge.hop = 0
            bridge.native_capability = "reviewer"
            bridge.open_issues = []
            bridge._pending_transition = None

            result = bridge.invoke("reviewer", None)

            self.assertEqual(result.metrics["review_score"], 7.7)
            self.assertEqual(bridge.native_capability, "reviewer")
            bridge.accept_invocation(result, CoordinationDecision(Action.REVERIFY, None, "accepted"))
            self.assertEqual(bridge.native_capability, "planner")
            self.assertTrue(bridge.open_issues)
            self.assertNotIn("review:score_missing", [item.issue_id for item in bridge.open_issues])


if __name__ == "__main__":
    unittest.main()
