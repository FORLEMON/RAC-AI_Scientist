import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rac_ai_scientist.hosts.agent_laboratory import (
    MAX_COMPLETION_TOKENS_PER_REQUEST,
    AgentLaboratoryBridge,
    _content_filter_retry_messages,
    _is_content_filter_error,
    _request_completion_limit,
    _usage_counts,
)
from rac_ai_scientist.schemas import Budget


class AgentLaboratoryModelAdapterTests(unittest.TestCase):
    @staticmethod
    def _bridge(root: Path) -> AgentLaboratoryBridge:
        manifest = root / "agent_laboratory.json"
        manifest.write_text(json.dumps({
            "host_id": "agent_laboratory",
            "capabilities": [],
        }), encoding="utf-8")
        bridge = AgentLaboratoryBridge(
            root,
            manifest,
            Budget(25, 60_000_000, 1_300_000, 900, 21_600, 14),
            "openai/DeepSeek-V4-Flash-0731",
            "fake-only",
        )
        bridge.started = time.monotonic()
        return bridge

    @staticmethod
    def _response(content="ok", prompt_tokens=10, completion_tokens=3):
        return SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            _hidden_params={},
            model_extra={},
        )

    def _install_fake_adapter(self, bridge, outcomes):
        requests = []

        class FakeCompletions:
            def create(self, **request):
                requests.append(request)
                outcome = outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        openai = types.ModuleType("openai")
        openai.OpenAI = lambda **_kwargs: fake_client
        modules = {name: types.ModuleType(name) for name in (
            "ai_lab_repo", "agents", "inference", "mlesolver", "papersolver",
        )}
        for module in modules.values():
            module.query_model = None
        with patch.dict(sys.modules, {"openai": openai, **modules}):
            bridge._install_model_adapter()
        return modules["inference"].query_model, requests

    def test_lifecycle_output_budget_is_capped_per_request(self):
        self.assertEqual(_request_completion_limit(1_300_000), MAX_COMPLETION_TOKENS_PER_REQUEST)
        self.assertEqual(_request_completion_limit(12_345), 12_345)
        with self.assertRaisesRegex(RuntimeError, "output-token budget exhausted"):
            _request_completion_limit(0)

    def test_content_filter_is_recognized_and_usage_is_read_from_error(self):
        class Filtered(Exception):
            status_code = 400
            body = {
                "choices": [{"finish_reason": "content_filter"}],
                "usage": {"prompt_tokens": 4_945, "completion_tokens": 0},
            }

        exc = Filtered("Response content blocked by label 'Jailbreak'.")
        self.assertTrue(_is_content_filter_error(exc))
        self.assertEqual(_usage_counts(exc), (4_945, 0))

    def test_content_filter_retry_preserves_task_and_adds_safety_frame(self):
        messages = _content_filter_retry_messages("system role", "scientific task")
        self.assertIn("Follow all provider safety policies", messages[0]["content"])
        self.assertIn("system role", messages[0]["content"])
        self.assertIn("scientific task", messages[1]["content"])
        self.assertIn("<scientific_subtask>", messages[1]["content"])

    def test_content_filter_retries_once_and_counts_both_calls(self):
        class Filtered(Exception):
            status_code = 400

            def __init__(self):
                super().__init__("content_filter: Response content blocked by label 'Jailbreak'")
                self.body = {"usage": {"prompt_tokens": 4_945, "completion_tokens": 0}}

        with tempfile.TemporaryDirectory() as raw:
            bridge = self._bridge(Path(raw))
            query_model, requests = self._install_fake_adapter(
                bridge,
                [Filtered(), self._response("recovered", 40, 7)],
            )
            result = query_model(
                model_str="ignored",
                prompt="prepare the experiment",
                system_prompt="researcher role",
            )

        self.assertEqual(result, "recovered")
        self.assertEqual(len(requests), 2)
        self.assertEqual(bridge.provider_calls, 2)
        self.assertEqual(bridge.input_tokens, 4_985)
        self.assertEqual(bridge.output_tokens, 7)
        self.assertIn("benign academic research", requests[1]["messages"][0]["content"])

    def test_non_filter_bad_request_is_not_retried(self):
        class InvalidRequest(Exception):
            status_code = 400

        with tempfile.TemporaryDirectory() as raw:
            bridge = self._bridge(Path(raw))
            query_model, requests = self._install_fake_adapter(
                bridge,
                [InvalidRequest("request exceeds model context")],
            )
            with self.assertRaisesRegex(InvalidRequest, "exceeds model context"):
                query_model(model_str="ignored", prompt="task", system_prompt="role")

        self.assertEqual(len(requests), 1)
        self.assertEqual(bridge.provider_calls, 1)

    def test_second_content_filter_is_terminal(self):
        class Filtered(Exception):
            status_code = 400

            def __init__(self):
                super().__init__("content_filter")
                self.body = {"usage": {"prompt_tokens": 5, "completion_tokens": 0}}

        with tempfile.TemporaryDirectory() as raw:
            bridge = self._bridge(Path(raw))
            query_model, requests = self._install_fake_adapter(
                bridge,
                [Filtered(), Filtered()],
            )
            with self.assertRaises(Filtered):
                query_model(model_str="ignored", prompt="task", system_prompt="role")

        self.assertEqual(len(requests), 2)
        self.assertEqual(bridge.provider_calls, 2)
        self.assertEqual(bridge.input_tokens, 10)

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
