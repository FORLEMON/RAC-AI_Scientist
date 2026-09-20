import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rac_ai_scientist.hosts.agent_laboratory import _install_arxiv_transport


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

    def test_failed_search_surfaces_error_but_empty_results_are_valid(self):
        arxiv = types.ModuleType("arxiv")
        arxiv.Client = type("Client", (FakeClient,), {})
        tools = types.ModuleType("tools")
        tools.ArxivSearch = type("ArxivSearch", (FakeArxivSearch,), {})
        with patch.dict(sys.modules, {"arxiv": arxiv, "tools": tools}):
            _install_arxiv_transport()
            search = tools.ArxivSearch()
            with self.assertRaisesRegex(RuntimeError, "arXiv API search failed"):
                search.find_papers_by_str("query")
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


if __name__ == "__main__":
    unittest.main()
