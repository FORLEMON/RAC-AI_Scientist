import unittest

from rac_ai_scientist.hosts.registry import HOST_IDS, SHAREDNET_HOST_IDS, local_snapshot_name


class HostRegistryTests(unittest.TestCase):
    def test_all_six_hosts_have_unique_snapshots(self):
        self.assertEqual(len(HOST_IDS), 6)
        snapshots = [local_snapshot_name(host) for host in HOST_IDS]
        self.assertEqual(len(snapshots), len(set(snapshots)))
        self.assertEqual(local_snapshot_name("ai_researcher"), "AI-Researcher-main")
        self.assertEqual(local_snapshot_name("evo_scientist"), "EvoScientist-main")
        self.assertEqual(local_snapshot_name("auto_research_claw"), "AutoResearchClaw-main")

    def test_sharednet_matrix_excludes_retired_data_to_paper_host(self):
        self.assertEqual(
            SHAREDNET_HOST_IDS,
            ("ark", "agent_laboratory", "ai_researcher", "evo_scientist", "auto_research_claw"),
        )
        self.assertNotIn("data_to_paper", SHAREDNET_HOST_IDS)
