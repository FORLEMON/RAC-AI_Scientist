import unittest

from rac_ai_scientist.hosts.registry import HOST_IDS, SHAREDNET_HOST_IDS, local_snapshot_name


class HostRegistryTests(unittest.TestCase):
    def test_active_hosts_have_unique_snapshots(self):
        self.assertEqual(len(HOST_IDS), 3)
        snapshots = [local_snapshot_name(host) for host in HOST_IDS]
        self.assertEqual(len(snapshots), len(set(snapshots)))
        self.assertEqual(local_snapshot_name("evo_scientist"), "EvoScientist-main")

    def test_sharednet_matrix_contains_only_active_hosts(self):
        self.assertEqual(
            SHAREDNET_HOST_IDS,
            ("ark", "agent_laboratory", "evo_scientist"),
        )
        self.assertNotIn("data_to_paper", SHAREDNET_HOST_IDS)
        self.assertNotIn("ai_researcher", SHAREDNET_HOST_IDS)
        self.assertNotIn("auto_research_claw", SHAREDNET_HOST_IDS)
