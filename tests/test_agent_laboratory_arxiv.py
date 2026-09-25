import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rac_ai_scientist.hosts.agent_laboratory import (
    _install_arxiv_transport,
    _install_researchclawbench_literature_guard,
)


def mark_researchclawbench(workspace: Path) -> None:
    marker = workspace / ".rac" / "benchmark.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        '{"schema_version": 1, "benchmark": "researchclawbench"}',
        encoding="utf-8",
    )


class FakeSession:
    def __init__(self):
        self.timeout = None

    def get(self, url, **kwargs):
        self.timeout = kwargs.get("timeout")
        return url


class FakeClient:
    def __init__(self):
        self.num_retries = 3
        self._session = FakeSession()


class FakeArxivSearch:
    result = None

    def __init__(self):
        self.full_text_requests = []

    def find_papers_by_str(self, query, N=20):
        return self.result

    def retrieve_full_paper_text(self, query, MAX_LEN=50000):
        self.full_text_requests.append(query)
        return "remote paper"


class GuardPhD:
    def __init__(self):
        self.lit_review = []

    def inference(self, *args, **kwargs):
        return "```SUMMARY\nkeep searching forever\n```"


class AgentLaboratoryArxivTests(unittest.TestCase):
    def test_arxiv_api_call_has_deadline_without_nested_retries(self):
        arxiv = types.ModuleType("arxiv")
        arxiv.Client = type("Client", (FakeClient,), {})
        tools = types.ModuleType("tools")
        tools.ArxivSearch = type("ArxivSearch", (FakeArxivSearch,), {})
        with patch.dict(sys.modules, {"arxiv": arxiv, "tools": tools}):
            _install_arxiv_transport()
            client = arxiv.Client()
            self.assertEqual(client.num_retries, 0)
            self.assertEqual(client._session.get("https://export.arxiv.org/api/query"), "https://export.arxiv.org/api/query")
            self.assertEqual(client._session.inner.timeout, (5, 30))

    def test_failed_search_is_skipped_as_an_empty_result(self):
        arxiv = types.ModuleType("arxiv")
        arxiv.Client = type("Client", (FakeClient,), {})
        tools = types.ModuleType("tools")
        tools.ArxivSearch = type("ArxivSearch", (FakeArxivSearch,), {})
        with patch.dict(sys.modules, {"arxiv": arxiv, "tools": tools}):
            _install_arxiv_transport()
            search = tools.ArxivSearch()
            with self.assertWarnsRegex(RuntimeWarning, "continuing Agent Laboratory"):
                self.assertEqual(search.find_papers_by_str("query"), "")
            search.result = ""
            self.assertEqual(search.find_papers_by_str("query"), "")

    def test_supplied_pdf_precedes_network_search_and_supports_full_text(self):
        arxiv = types.ModuleType("arxiv")
        arxiv.Client = type("Client", (FakeClient,), {})
        tools = types.ModuleType("tools")
        tools.ArxivSearch = type("ArxivSearch", (FakeArxivSearch,), {})
        pypdf = types.ModuleType("pypdf")

        class Page:
            def extract_text(self):
                return "Local benchmark evidence with methods and results."

        class Reader:
            metadata = types.SimpleNamespace(title="Supplied Study")
            pages = [Page()]

            def __init__(self, path):
                self.path = path

        pypdf.PdfReader = Reader
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            mark_researchclawbench(workspace)
            related = workspace / "related_work"
            related.mkdir()
            (related / "paper_000.pdf").write_bytes(b"placeholder")
            with patch.dict(sys.modules, {"arxiv": arxiv, "tools": tools, "pypdf": pypdf}):
                self.assertEqual(_install_arxiv_transport(workspace), 1)
                search = tools.ArxivSearch()
                summaries = search.find_papers_by_str("unrelated query")
                self.assertIn("Supplied Study", summaries)
                self.assertIn("arXiv paper ID: local-paper-000", summaries)
                self.assertIn("Local benchmark evidence", search.retrieve_full_paper_text("local-paper-000"))
                self.assertEqual(search.full_text_requests, [])

    def test_real_arxiv_id_aliases_resolve_to_supplied_pdf(self):
        arxiv = types.ModuleType("arxiv")
        arxiv.Client = type("Client", (FakeClient,), {})
        tools = types.ModuleType("tools")
        tools.ArxivSearch = type("ArxivSearch", (FakeArxivSearch,), {})
        pypdf = types.ModuleType("pypdf")

        class Page:
            def extract_text(self):
                return "arXiv:1503.01243v2 Local accelerated-gradient evidence."

        class Reader:
            metadata = types.SimpleNamespace(
                title="arXiv:1503.01243v2 [stat.ML]",
                subject="",
                keywords="",
            )
            pages = [Page()]

            def __init__(self, path):
                self.path = path

        pypdf.PdfReader = Reader

        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            mark_researchclawbench(workspace)
            related = workspace / "related_work"
            related.mkdir()
            (related / "paper_001.pdf").write_bytes(b"placeholder")
            with patch.dict(sys.modules, {"arxiv": arxiv, "tools": tools, "pypdf": pypdf}):
                self.assertEqual(_install_arxiv_transport(workspace), 1)
                search = tools.ArxivSearch()
                summaries = search.find_papers_by_str("accelerated gradient")
                self.assertIn("arXiv paper ID: 1503.01243", summaries)
                for paper_id in (
                    "1503.01243",
                    "1503.01243v2",
                    "arXiv:1503.01243v2",
                    "https://arxiv.org/abs/1503.01243v2",
                    "local-paper-000",
                ):
                    self.assertIn(
                        "Local accelerated-gradient evidence",
                        search.retrieve_full_paper_text(paper_id),
                    )
                self.assertEqual(search.full_text_requests, [])

    def test_unknown_id_falls_back_to_native_arxiv_with_local_papers(self):
        arxiv = types.ModuleType("arxiv")
        arxiv.Client = type("Client", (FakeClient,), {})
        tools = types.ModuleType("tools")
        tools.ArxivSearch = type("ArxivSearch", (FakeArxivSearch,), {})
        pypdf = types.ModuleType("pypdf")

        class Page:
            def extract_text(self):
                return "Supplied benchmark evidence."

        class Reader:
            metadata = types.SimpleNamespace(title="Supplied Study", subject="", keywords="")
            pages = [Page()]

            def __init__(self, path):
                self.path = path

        pypdf.PdfReader = Reader

        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            mark_researchclawbench(workspace)
            related = workspace / "related_work"
            related.mkdir()
            (related / "paper_000.pdf").write_bytes(b"placeholder")
            with patch.dict(sys.modules, {"arxiv": arxiv, "tools": tools, "pypdf": pypdf}):
                self.assertEqual(_install_arxiv_transport(workspace), 1)
                search = tools.ArxivSearch()
                self.assertEqual(search.retrieve_full_paper_text("2501.04227"), "remote paper")
                self.assertEqual(search.full_text_requests, ["2501.04227"])

    def test_external_search_remains_fallback_without_supplied_pdfs(self):
        arxiv = types.ModuleType("arxiv")
        arxiv.Client = type("Client", (FakeClient,), {})
        tools = types.ModuleType("tools")
        tools.ArxivSearch = type("ArxivSearch", (FakeArxivSearch,), {})
        with tempfile.TemporaryDirectory() as raw, patch.dict(sys.modules, {"arxiv": arxiv, "tools": tools}):
            self.assertEqual(_install_arxiv_transport(Path(raw)), 0)
            search = tools.ArxivSearch()
            search.result = "remote results"
            self.assertEqual(search.find_papers_by_str("query"), "remote results")
            self.assertEqual(search.retrieve_full_paper_text("2501.04227"), "remote paper")

    def test_non_rcb_workspace_does_not_activate_local_pdf_transport(self):
        arxiv = types.ModuleType("arxiv")
        arxiv.Client = type("Client", (FakeClient,), {})
        tools = types.ModuleType("tools")
        tools.ArxivSearch = type("ArxivSearch", (FakeArxivSearch,), {})
        with tempfile.TemporaryDirectory() as raw, patch.dict(
            sys.modules, {"arxiv": arxiv, "tools": tools}
        ):
            workspace = Path(raw)
            related = workspace / "related_work"
            related.mkdir()
            (related / "paper.pdf").write_bytes(b"paperbench-like primary PDF")
            self.assertEqual(_install_arxiv_transport(workspace), 0)
            search = tools.ArxivSearch()
            search.result = "native benchmark search"
            self.assertEqual(search.find_papers_by_str("query"), "native benchmark search")

    def test_rcb_guard_bounds_each_local_paper_to_full_text_then_add(self):
        pypdf = types.ModuleType("pypdf")

        class Page:
            def __init__(self, text):
                self.text = text

            def extract_text(self):
                return self.text

        class Reader:
            def __init__(self, path):
                stem = Path(path).stem
                self.metadata = types.SimpleNamespace(
                    title=f"Study {stem}", subject="", keywords=""
                )
                self.pages = [Page(f"Evidence from {stem}.")]

        workflow = types.SimpleNamespace(phd=GuardPhD())
        pypdf.PdfReader = Reader
        with tempfile.TemporaryDirectory() as raw, patch.dict(sys.modules, {"pypdf": pypdf}):
            workspace = Path(raw)
            mark_researchclawbench(workspace)
            related = workspace / "related_work"
            related.mkdir()
            (related / "paper_000.pdf").write_bytes(b"first")
            (related / "paper_001.pdf").write_bytes(b"second")

            self.assertEqual(
                _install_researchclawbench_literature_guard(workflow, workspace),
                2,
            )
            first_full = workflow.phd.inference("topic", "literature review")
            self.assertIn("```FULL_TEXT\nlocal-paper-000", first_full)
            first_add = workflow.phd.inference("topic", "literature review", feedback="full")
            self.assertIn("```ADD_PAPER\nlocal-paper-000", first_add)
            workflow.phd.lit_review.append({"arxiv_id": "local-paper-000"})

            second_full = workflow.phd.inference("topic", "literature review")
            self.assertIn("```FULL_TEXT\nlocal-paper-001", second_full)
            second_add = workflow.phd.inference("topic", "literature review", feedback="full")
            self.assertIn("```ADD_PAPER\nlocal-paper-001", second_add)

            untouched = workflow.phd.inference("topic", "plan formulation")
            self.assertIn("```SUMMARY", untouched)

            restored = pickle.loads(pickle.dumps(workflow.phd))
            self.assertIn("```FULL_TEXT", restored.inference("topic", "literature review"))


if __name__ == "__main__":
    unittest.main()
