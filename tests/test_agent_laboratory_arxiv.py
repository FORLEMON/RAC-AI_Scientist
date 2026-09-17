import sys
import types
import unittest
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

    def find_papers_by_str(self, query, N=20):
        return self.result


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


if __name__ == "__main__":
    unittest.main()
