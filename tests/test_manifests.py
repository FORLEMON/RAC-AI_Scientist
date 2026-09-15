import unittest
from pathlib import Path

from rac_ai_scientist.manifest import capability_cards, load_host_manifest
from rac_ai_scientist.policy import verify_result
from rac_ai_scientist.schemas import ArtifactRecord, EvidenceRequirement, InvocationResult, WorkContract


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

    def test_agent_laboratory_data_preparation_can_write_derived_data_not_source_data(self):
        manifest = load_host_manifest(
            Path(__file__).resolve().parents[1] / "configs" / "hosts" / "agent_laboratory.json"
        )
        card = next(
            item for item in capability_cards(manifest)
            if item.capability_id == "data_preparation"
        )
        contract = WorkContract(
            "ep:data_preparation", card.capability_id, "prepare local data",
            card.readable_artifacts, card.writable_artifacts,
            (
                EvidenceRequirement("artifact_changed", card.produces),
                EvidenceRequirement("nonempty_output"),
            ),
        )
        loader = ArtifactRecord("loader", "code/agent_laboratory/load_data.py", "code", "new", 200)
        processed = ArtifactRecord(
            "processed", "data/processed/cloud_seeding_prepared.csv", "result", "new", 200
        )
        produced = InvocationResult(card.capability_id, "prepared", [], [loader, processed])
        self.assertEqual(verify_result(contract, produced).verdict.value, "supported")

        source = ArtifactRecord(
            "source", "data/dataset1_cloud_seeding_records/source.csv", "result", "new", 200
        )
        overwritten = InvocationResult(card.capability_id, "prepared", [], [loader, source])
        self.assertEqual(verify_result(contract, overwritten).verdict.value, "refuted")
