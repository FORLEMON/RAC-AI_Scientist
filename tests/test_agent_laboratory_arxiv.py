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

    def results(self, search):
        return iter(())


class FakeArxivSearch:
    result = None

    def __init__(self):
        self.sch_engine = types.SimpleNamespace(_rac_search_error=None)

    def find_papers_by_str(self, query, N=20):
        return self.result


class AgentLaboratoryArxivTests(unittest.TestCase):
    def test_native_retry_exhaustion_preserves_the_external_failure(self):
        class Client(FakeClient):
            def results(self, query):
                yield from ()
                raise TimeoutError("arXiv transport timed out")

        class Search:
            def __init__(self):
                self.sch_engine = Client()

            def find_papers_by_str(self, query, N=20):
                if query == "fails_before_client":
                    return None
                for _ in range(3):
                    try:
                        return "\n".join(self.sch_engine.results(query))
                    except Exception:
                        pass
                return None

        arxiv, tools = types.ModuleType("arxiv"), types.ModuleType("tools")
        arxiv.Client, tools.ArxivSearch = Client, Search
        with patch.dict(sys.modules, {"arxiv": arxiv, "tools": tools}):
            _install_arxiv_transport()
            _install_arxiv_transport()
            search = tools.ArxivSearch()
            with self.assertRaisesRegex(RuntimeError, "TimeoutError") as raised:
                search.find_papers_by_str("query")
            self.assertIsInstance(raised.exception.__cause__, TimeoutError)
            with self.assertRaises(RuntimeError) as unknown:
                search.find_papers_by_str("fails_before_client")
            self.assertNotIn("TimeoutError", str(unknown.exception))
            self.assertIsNone(unknown.exception.__cause__)

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
