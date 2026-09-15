import sys
import types
import unittest
from unittest.mock import patch

from rac_ai_scientist.hosts.agent_laboratory import _install_hf_data_search


class FakeHFDataSearch:
    created = 0

    def __init__(self):
        type(self).created += 1

    def retrieve_ds(self, query):
        return [query]

    def results_str(self, datasets):
        return ["found: " + item for item in datasets]


class AgentLaboratoryHFDataTests(unittest.TestCase):
    def test_dataset_catalog_is_loaded_only_for_explicit_hf_search(self):
        FakeHFDataSearch.created = 0
        ai_lab_repo = types.ModuleType("ai_lab_repo")
        ai_lab_repo.HFDataSearch = FakeHFDataSearch
        with patch.dict(sys.modules, {"ai_lab_repo": ai_lab_repo}):
            _install_hf_data_search()
            _install_hf_data_search()
            search = ai_lab_repo.HFDataSearch()
            self.assertEqual(FakeHFDataSearch.created, 0)
            datasets = search.retrieve_ds("tracking")
            self.assertEqual(search.results_str(datasets), ["found: tracking"])
            self.assertEqual(FakeHFDataSearch.created, 1)


if __name__ == "__main__":
    unittest.main()
