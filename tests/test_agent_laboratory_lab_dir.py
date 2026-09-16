import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rac_ai_scientist.hosts.agent_laboratory import AgentLaboratoryBridge
from rac_ai_scientist.schemas import Budget


class AgentLaboratoryLabDirTests(unittest.TestCase):
    def test_native_save_path_resolves_inside_the_workspace(self):
        class FakeWorkflow:
            received_lab_dir = None

            def __init__(self, **kwargs):
                type(self).received_lab_dir = kwargs["lab_dir"]
                native_path = Path(f"./{kwargs['lab_dir']}/src") / "probe.py"
                native_path.write_text("saved by pinned native path", encoding="utf-8")

        fake_upstream = types.ModuleType("ai_lab_repo")
        fake_upstream.LaboratoryWorkflow = FakeWorkflow
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            upstream = root / "upstream"
            upstream.mkdir()
            (upstream / "ai_lab_repo.py").write_text("# test-only upstream\n", encoding="utf-8")
            manifest = root / "agent_laboratory.json"
            manifest.write_text(
                json.dumps({"host_id": "agent_laboratory", "capabilities": []}), encoding="utf-8"
            )
            workspace = root / "workspace"
            bridge = AgentLaboratoryBridge(
                upstream, manifest, Budget(20, 1000, 1000, 10, 100, 10),
                "DeepSeek-V4-Pro", "fake-only"
            )
            with (
                patch.dict(sys.modules, {"ai_lab_repo": fake_upstream}),
                patch.dict(os.environ, {"OPENAI_API_KEY": "fake-only"}),
                patch.object(sys, "path", list(sys.path)),
                patch("rac_ai_scientist.hosts.agent_laboratory.seed_runtime"),
                patch("rac_ai_scientist.hosts.agent_laboratory._install_arxiv_transport"),
                patch("rac_ai_scientist.hosts.agent_laboratory._install_hf_data_search"),
                patch("rac_ai_scientist.hosts.agent_laboratory._install_report_writing_scope"),
                patch.object(AgentLaboratoryBridge, "_install_model_adapter"),
                patch.object(AgentLaboratoryBridge, "_save"),
            ):
                bridge.initialize_native(episode_id="probe", workspace=workspace, objective="test topic", seed=0)

            self.assertEqual(FakeWorkflow.received_lab_dir, "agent_laboratory")
            self.assertEqual((workspace / "agent_laboratory" / "src" / "probe.py").read_text(encoding="utf-8"),
                             "saved by pinned native path")


if __name__ == "__main__":
    unittest.main()
