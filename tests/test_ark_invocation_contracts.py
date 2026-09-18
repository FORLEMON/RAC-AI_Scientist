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
            policy = SharedPolicy("R5")
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
