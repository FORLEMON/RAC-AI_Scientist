import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rac_ai_scientist.hosts.agent_laboratory import _install_hf_data_search


class FakeHFDataSearch:
    created = 0

    def __init__(self):
        type(self).created += 1
        raise RuntimeError("external dataset catalog must not be loaded")


class AgentLaboratoryHFDataTests(unittest.TestCase):
    def test_explicit_hf_search_returns_local_input_notice_without_remote_catalog(self):
        FakeHFDataSearch.created = 0
        ai_lab_repo = types.ModuleType("ai_lab_repo")
        ai_lab_repo.HFDataSearch = FakeHFDataSearch
        with tempfile.TemporaryDirectory() as raw, patch.dict(sys.modules, {"ai_lab_repo": ai_lab_repo}):
            workspace = Path(raw)
            data = workspace / "data"
            data.mkdir()
            (data / "records.csv").write_text("year,count\n2000,1\n", encoding="utf-8")
            _install_hf_data_search(workspace)
            search = ai_lab_repo.HFDataSearch()
            notice = "\n".join(search.results_str(search.retrieve_ds("external records")))
            self.assertIn("External Hugging Face dataset search is disabled", notice)
            self.assertIn("data/records.csv", notice)
            self.assertNotIn(str(workspace), notice)
            self.assertNotIn("external records", notice)
            self.assertEqual(FakeHFDataSearch.created, 0)

    def test_new_episode_notice_uses_its_own_local_input_path(self):
        ai_lab_repo = types.ModuleType("ai_lab_repo")
        ai_lab_repo.HFDataSearch = FakeHFDataSearch
        with tempfile.TemporaryDirectory() as raw, patch.dict(sys.modules, {"ai_lab_repo": ai_lab_repo}):
            first = Path(raw) / "first"
            second = Path(raw) / "second"
            (first / "data").mkdir(parents=True)
            (second / "data").mkdir(parents=True)
            (first / "data" / "first.csv").write_text("a\n", encoding="utf-8")
            (second / "data" / "second.csv").write_text("b\n", encoding="utf-8")
            _install_hf_data_search(first)
            _install_hf_data_search(second)
            search = ai_lab_repo.HFDataSearch()
            notice = "\n".join(search.results_str(search.retrieve_ds("external")))
            self.assertIn("data/second.csv", notice)
            self.assertNotIn("first.csv", notice)


if __name__ == "__main__":
    unittest.main()
