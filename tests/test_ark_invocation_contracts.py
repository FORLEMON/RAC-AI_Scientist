import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rac_ai_scientist.hosts.ark import ArkBridge
from rac_ai_scientist.policy import SharedPolicy
from rac_ai_scientist.schemas import Budget, Issue


class ArkInvocationContractTests(unittest.TestCase):
    def make_bridge(self, root, run_agent):
        manifest = Path(__file__).parents[1] / "configs/hosts/ark.json"
        bridge = ArkBridge(root, manifest, Budget(25, 10000, 10000, 10, 100, 10), "fake", "FAKE")
        self.assertEqual(bridge.model, "openai/fake")
        bridge.workspace = root
        bridge.started = time.monotonic()
        bridge.orchestrator = types.SimpleNamespace(run_agent=run_agent, _terminal_error=None, _agent_stats=[])
        bridge.episode_id = "episode"
        bridge.objective = "Analyze the supplied measurements"
        return bridge

    def test_native_terminal_error_is_propagated_before_persisting_output(self):
        with tempfile.TemporaryDirectory() as directory:
            def run_agent(*args, **kwargs):
                bridge.orchestrator._terminal_error = "APIError: provider terminated the run"
                return ""
            bridge = self.make_bridge(Path(directory), run_agent)
            with patch.object(bridge, "_persist_output") as persist, patch.object(bridge, "_normalize_report") as normalize:
                result = bridge.invoke("writer", None)
            self.assertTrue(result.terminal_error)
            self.assertIn("APIError", result.error)
            self.assertIsNone(bridge._pending_transition)
            persist.assert_not_called()
            normalize.assert_not_called()

    def test_content_filter_retries_once_with_compact_native_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def run_agent(role, prompt, **kwargs):
                calls.append((role, prompt, kwargs))
                if len(calls) == 1:
                    bridge.orchestrator._terminal_error = (
                        "LLMBadRequestError: Error code: 400; "
                        "finish_reason': 'content_filter'"
                    )
                    return ""
                return "Writer recovered and persisted the report."

            bridge = self.make_bridge(Path(directory), run_agent)
            result = bridge.invoke("writer", None)

            self.assertEqual(len(calls), 2)
            self.assertIsNone(result.error)
            self.assertFalse(result.terminal_error)
            self.assertEqual(result.metrics["provider_filter_retried"], 1.0)
            self.assertIn("Use neutral technical language", calls[1][1])
            self.assertIn("Analyze the supplied measurements", calls[1][1])

    def test_second_content_filter_remains_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def run_agent(*args, **kwargs):
                calls.append((args, kwargs))
                bridge.orchestrator._terminal_error = "content_filter: Jailbreak"
                return ""

            bridge = self.make_bridge(Path(directory), run_agent)
            result = bridge.invoke("writer", None)

            self.assertEqual(len(calls), 2)
            self.assertTrue(result.terminal_error)
            self.assertIn("content_filter", result.error)

    def test_unrelated_bad_request_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def run_agent(*args, **kwargs):
                calls.append((args, kwargs))
                bridge.orchestrator._terminal_error = (
                    "LLMBadRequestError: Error code: 400 invalid_parameter"
                )
                return ""

            bridge = self.make_bridge(Path(directory), run_agent)
            result = bridge.invoke("writer", None)

            self.assertEqual(len(calls), 1)
            self.assertTrue(result.terminal_error)
            self.assertNotIn("provider_filter_retried", result.metrics)

    def test_writer_acceptance_retains_unassigned_execution_issue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def run_agent(role, prompt, **kwargs):
                (root / "report").mkdir()
                (root / "report/report.md").write_text("# Results\n" + "Measured result. " * 30)
                return "Updated only the discussion."
            bridge = self.make_bridge(root, run_agent)
            bridge.open_issues = [
                Issue("review:writing:one", "writing", "Clarify the discussion", required_tags=("writing",)),
                Issue("review:execution:two", "execution", "Repair the experiment", required_tags=("code", "debug")),
            ]
            policy = SharedPolicy("R3")
            checkpoint = bridge.checkpoint()
            decision = policy.decide(checkpoint)
            self.assertEqual(decision.capability_id, "writer")
            result = bridge.invoke("writer", decision.contract)
            evaluated = policy.evaluate(checkpoint, decision, result)
            self.assertEqual(evaluated.verification.verdict.value, "supported")
            bridge.accept_invocation(result, evaluated)
            self.assertEqual([i.issue_id for i in bridge.open_issues], ["review:execution:two"])
            self.assertNotEqual(policy.decide(bridge.checkpoint()).capability_id, "writer")


if __name__ == "__main__":
    unittest.main()
