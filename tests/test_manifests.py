import unittest
from pathlib import Path

from rac_ai_scientist.manifest import capability_cards, load_host_manifest


class ManifestTests(unittest.TestCase):
    def test_all_host_manifests_are_policy_free_and_parseable(self):
        root = Path(__file__).resolve().parents[1] / "configs" / "hosts"
        manifests = [load_host_manifest(path) for path in sorted(root.glob("*.json"))]
        self.assertEqual(
            {item["host_id"] for item in manifests},
            {
                "ark",
                "agent_laboratory",
                "data_to_paper",
                "ai_researcher",
                "evo_scientist",
                "auto_research_claw",
            },
        )
        for manifest in manifests:
            self.assertGreater(len(capability_cards(manifest)), 0)
