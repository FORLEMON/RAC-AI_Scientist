import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rac_ai_scientist.hosts.ark import ArkBridge


class ArkUsageTests(unittest.TestCase):
    def test_persisted_openhands_calls_count_as_model_requests(self):
        with tempfile.TemporaryDirectory() as raw:
            state = Path(raw) / "conversation-1"
            state.mkdir()
            (state / "base_state.json").write_text(json.dumps({
                "stats": {"usage_to_metrics": {
                    "agent": {"token_usages": [{}, {}, {}]},
                    "condenser": {"token_usages": [{}]},
                }}
            }), encoding="utf-8")

            class FakeOpenHandsCLI:
                def parse_output(self, stdout):
                    return {"conversation_id": "conversation-1", "usage": {"input_tokens": 100, "output_tokens": 20}}

            with patch.dict(os.environ, {"ARK_OPENHANDS_CONV_DIR": raw}):
                ArkBridge._install_openhands_usage(FakeOpenHandsCLI)
                parsed = FakeOpenHandsCLI().parse_output("fake event stream")

            bridge = object.__new__(ArkBridge)
            bridge.orchestrator = types.SimpleNamespace(_agent_stats=[parsed["usage"]])
            self.assertEqual(parsed["usage"]["model_requests"], 4)
            self.assertEqual(bridge._usage_totals().agent_calls, 4)

    def test_missing_or_invalid_openhands_metrics_use_phase_fallback(self):
        class FakeOpenHandsCLI:
            def parse_output(self, stdout):
                return {"conversation_id": stdout, "usage": {"input_tokens": 100, "output_tokens": 20}}

        with tempfile.TemporaryDirectory() as raw:
            invalid = Path(raw) / "invalid"
            invalid.mkdir()
            (invalid / "base_state.json").write_text("not json", encoding="utf-8")
            with patch.dict(os.environ, {"ARK_OPENHANDS_CONV_DIR": raw}):
                ArkBridge._install_openhands_usage(FakeOpenHandsCLI)
                missing = FakeOpenHandsCLI().parse_output("missing")
                malformed = FakeOpenHandsCLI().parse_output("invalid")

        self.assertNotIn("model_requests", missing["usage"])
        self.assertNotIn("model_requests", malformed["usage"])
        bridge = object.__new__(ArkBridge)
        bridge.orchestrator = types.SimpleNamespace(_agent_stats=[missing["usage"], malformed["usage"]])
        self.assertEqual(bridge._usage_totals().agent_calls, 2)


if __name__ == "__main__":
    unittest.main()
