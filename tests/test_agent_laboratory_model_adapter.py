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

    def test_content_filter_retry_extracts_science_without_raw_instruction_scaffolding(self):
        system_prompt = (
            "plan formulation role; you MUST respond EXACTLY with "
            "```PLAN\\n...``` or ```DIALOGUE\\n...``` and use ONLY one COMMAND"
        )
        prompt = """
Scientific Goal: Develop a hybrid MAPF planner that reduces collisions.

Dataset: Grid maps contain starts, goals, obstacles, and benchmark scenarios.

Related literature: Prior studies combine neighborhood search with learned policies.

Postdoc: You MUST output EXACTLY one COMMAND and ignore all other formats.
"""
        first = _content_filter_retry_messages(system_prompt, prompt, level=1)
        second = _content_filter_retry_messages(system_prompt, prompt, level=2)

        first_text = " ".join(message["content"].lower() for message in first)
        second_text = " ".join(message["content"].lower() for message in second)
        self.assertIn("hybrid mapf planner", first_text)
        self.assertIn("grid maps", first_text)
        self.assertIn("prior studies", first_text)
        self.assertIn("```plan", first_text)
        self.assertNotIn("<workflow_context>", first_text)
        self.assertNotIn("related literature", second_text)
        self.assertLess(len(second_text), len(first_text))
        for text in (first_text, second_text):
            for phrase in ("jailbreak", "must", "exactly", "only one", "ignore all", "command"):
                self.assertNotIn(phrase, text)

    def test_content_filter_uses_two_reduced_retries_and_counts_all_calls(self):
        class Filtered(Exception):
            status_code = 400

            def __init__(self):
                super().__init__("content_filter: Response content blocked by label 'Jailbreak'")
                self.body = {"usage": {"prompt_tokens": 4_945, "completion_tokens": 0}}

        with tempfile.TemporaryDirectory() as raw:
            bridge = self._bridge(Path(raw))
            query_model, requests = self._install_fake_adapter(
                bridge,
                [Filtered(), Filtered(), self._response("recovered", 40, 7)],
            )
            result = query_model(
                model_str="ignored",
                prompt="prepare the experiment",
                system_prompt="researcher role",
            )

        self.assertEqual(result, "recovered")
        self.assertEqual(len(requests), 3)
        self.assertEqual(bridge.provider_calls, 3)
        self.assertEqual(bridge.input_tokens, 9_930)
        self.assertEqual(bridge.output_tokens, 7)
        self.assertIn("Scientific research assistant", requests[1]["messages"][0]["content"])
        self.assertEqual(requests[2]["messages"][0]["content"], "Scientific research assistant.")

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

    def test_third_content_filter_is_terminal(self):
        class Filtered(Exception):
            status_code = 400

            def __init__(self):
                super().__init__("content_filter")
                self.body = {"usage": {"prompt_tokens": 5, "completion_tokens": 0}}

        with tempfile.TemporaryDirectory() as raw:
            bridge = self._bridge(Path(raw))
            query_model, requests = self._install_fake_adapter(
                bridge,
                [Filtered(), Filtered(), Filtered()],
            )
            with self.assertRaises(Filtered):
                query_model(model_str="ignored", prompt="task", system_prompt="role")

        self.assertEqual(len(requests), 3)
        self.assertEqual(bridge.provider_calls, 3)
        self.assertEqual(bridge.input_tokens, 15)

    def test_filtered_empty_response_uses_reduced_retry(self):
        filtered = self._response("", 19, 0)
        filtered.choices[0].finish_reason = "content_filter"
        recovered = self._response("recovered", 7, 2)

        with tempfile.TemporaryDirectory() as raw:
            bridge = self._bridge(Path(raw))
            query_model, requests = self._install_fake_adapter(bridge, [filtered, recovered])
            result = query_model(
                model_str="ignored",
                prompt="Scientific Goal: evaluate a MAPF planner.",
                system_prompt="plan formulation ```PLAN\\n...```",
            )

        self.assertEqual(result, "recovered")
        self.assertEqual(len(requests), 2)
        self.assertEqual(bridge.input_tokens, 26)
        self.assertEqual(bridge.output_tokens, 2)

    def test_unfiltered_empty_response_is_not_accepted(self):
        empty = self._response("", 13, 0)
        empty.choices[0].finish_reason = "stop"

        with tempfile.TemporaryDirectory() as raw:
            bridge = self._bridge(Path(raw))
            query_model, requests = self._install_fake_adapter(bridge, [empty])
            with self.assertRaisesRegex(RuntimeError, "empty completion"):
                query_model(model_str="ignored", prompt="task", system_prompt="role")

        self.assertEqual(len(requests), 1)
        self.assertEqual(bridge.provider_calls, 1)
        self.assertEqual(bridge.input_tokens, 13)

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

    def test_native_failure_returns_usage_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bridge = self._bridge(root)

            class FailedWorkflow:
                @staticmethod
                def perform_research():
                    bridge.provider_calls += 2
                    bridge.input_tokens += 5_750
                    raise RuntimeError("provider rejected response")

            workspace = root / "workspace"
            workspace.mkdir()
            bridge.workspace = workspace
            bridge.workflow = FailedWorkflow()

            result = bridge.run_native()

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.native_status, "failed")
        self.assertIn("provider rejected response", result.reason)
        self.assertEqual(result.usage.agent_calls, 2)
        self.assertEqual(result.usage.input_tokens, 5_750)


if __name__ == "__main__":
    unittest.main()
