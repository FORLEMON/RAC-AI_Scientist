import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rac_ai_scientist.hosts.agent_laboratory import AgentLaboratoryBridge
from rac_ai_scientist.schemas import Budget, EvidenceRequirement, WorkContract


class AgentLaboratoryContractNotesTests(unittest.TestCase):
    def test_benchmark_note_requires_portable_paths_for_all_policies(self):
        bridge = object.__new__(AgentLaboratoryBridge)
        bridge.workspace = Path("/workspace")

        note = bridge._benchmark_note()

        self.assertIn("workspace-relative paths", note)

    def test_native_agents_receive_current_contract_without_accumulating_retry_notes(self):
        class FakeWorkflow:
            def __init__(self):
                self.notes = [{"phases": ["data preparation"], "note": "existing native note"}]
                self.ml_engineer = SimpleNamespace(notes=self.notes)
                self.phase_status = {"literature review": True, "plan formulation": True}
                self.seen = []

            def data_preparation(self):
                self.seen.append([
                    item["note"] for item in self.ml_engineer.notes
                    if item["phases"] == ["data preparation"]
                ])
                return True

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "agent_laboratory.json"
            manifest.write_text(json.dumps({
                "host_id": "agent_laboratory",
                "capabilities": [{
                    "capability_id": "data_preparation", "description": "prepare data",
                    "tags": ["code"], "readable_artifacts": ["data/**"],
                    "writable_artifacts": ["code/**", "outputs/**", "data/processed/**"],
                    "produces": ["code", "result"],
                }],
            }), encoding="utf-8")
            workspace = root / "workspace"
            workspace.mkdir()
            bridge = AgentLaboratoryBridge(
                root, manifest, Budget(20, 1000, 1000, 10, 100, 10),
                "DeepSeek-V4-Pro", "fake-only"
            )
            bridge.workspace = workspace
            bridge.workflow = FakeWorkflow()
            bridge.workflow.notes[0]["note"] = bridge._benchmark_note()
            bridge.started = time.monotonic()
            def contract(contract_id, objective):
                return WorkContract(
                    contract_id, "data_preparation", objective,
                    ("data/**",), ("code/**", "outputs/**", "data/processed/**"),
                    (EvidenceRequirement("artifact_changed", ("code", "result")),),
                )

            with (
                patch.object(bridge, "_persist_products"),
                patch.object(bridge, "_normalize_report"),
                patch.object(bridge, "_save"),
            ):
                self.assertIsNone(bridge.invoke("data_preparation", None).error)
                self.assertIsNone(bridge.invoke("data_preparation", contract("hop1", "first pass")).error)
                self.assertIsNone(bridge.invoke("data_preparation", contract("hop2", "second pass")).error)

            native, first, second = bridge.workflow.seen
            self.assertEqual(len(native), 1)
            self.assertIn("workspace-relative paths", native[0])
            self.assertEqual(len(first), 2)
            self.assertIn("first pass", first[-1])
            self.assertIn("data/processed/**", first[-1])
            self.assertNotIn("workspace-relative paths", first[-1])
            self.assertEqual(len(second), 2)
            self.assertIn("second pass", second[-1])
            self.assertNotIn("first pass", second[-1])
