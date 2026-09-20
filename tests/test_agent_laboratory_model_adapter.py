import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rac_ai_scientist.hosts.agent_laboratory import (
    MAX_COMPLETION_TOKENS_PER_REQUEST,
    AgentLaboratoryBridge,
    _request_completion_limit,
)
from rac_ai_scientist.schemas import Budget


class AgentLaboratoryModelAdapterTests(unittest.TestCase):
    def test_lifecycle_output_budget_is_capped_per_request(self):
        self.assertEqual(_request_completion_limit(1_300_000), MAX_COMPLETION_TOKENS_PER_REQUEST)
        self.assertEqual(_request_completion_limit(12_345), 12_345)
        with self.assertRaisesRegex(RuntimeError, "output-token budget exhausted"):
            _request_completion_limit(0)

    def test_bad_request_is_terminal_after_one_invocation(self):
        class ProviderValidationError(Exception):
            status_code = 400

        class FakeWorkflow:
            notes = []
            phase_status = {"literature review": False}

            @staticmethod
            def literature_review():
                raise ProviderValidationError("request exceeds model context")

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "agent_laboratory.json"
            manifest.write_text(json.dumps({
                "host_id": "agent_laboratory",
                "capabilities": [{
                    "capability_id": "literature_review",
                    "description": "review supplied literature",
                    "tags": ["literature"],
                    "readable_artifacts": ["related_work/**"],
                    "writable_artifacts": ["state/agent_laboratory/**"],
                    "produces": ["literature"],
                }],
            }), encoding="utf-8")
            workspace = root / "workspace"
            workspace.mkdir()
            bridge = AgentLaboratoryBridge(
                root,
                manifest,
                Budget(25, 60_000_000, 1_300_000, 900, 21_600, 14),
                "openai/DeepSeek-V4-Flash-0731",
                "fake-only",
            )
            bridge.workspace = workspace
            bridge.workflow = FakeWorkflow()
            bridge.started = time.monotonic()

            with patch.object(bridge, "_save"):
                result = bridge.invoke("literature_review", None)

        self.assertIn("request exceeds model context", result.error or "")
        self.assertTrue(result.terminal_error)
        self.assertEqual(bridge.hop, 1)


if __name__ == "__main__":
    unittest.main()
