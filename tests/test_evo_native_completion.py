import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from rac_ai_scientist.hosts.evo_scientist import EvoScientistBridge
from rac_ai_scientist.schemas import Budget, Usage


def _message(identifier: str, content: str, *, input_tokens: int = 10, output_tokens: int = 5):
    return SimpleNamespace(
        id=identifier,
        content=content,
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens},
    )


def _bridge(workspace: Path) -> EvoScientistBridge:
    bridge = object.__new__(EvoScientistBridge)
    bridge.workspace = workspace
    bridge.agent = object()
    bridge.episode_id = "evo-native-test"
    bridge.objective = "derive and test an optimization method"
    bridge.initial_budget = Budget(50, 1_000_000, 100_000, 100, 3_600, 14)
    bridge.usage = Usage()
    bridge._seen_messages = set()
    bridge.terminal = False
    bridge.started = time.monotonic()
    return bridge


class EvoNativeCompletionTests(unittest.TestCase):
    def test_successful_first_pass_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            bridge = _bridge(workspace)

            def invoke(_prompt, *, rubric, retry_prompt):
                self.assertIn("report/report.md", rubric)
                self.assertIn("Objective:", retry_prompt)
                self.assertIn("preselected as Lite", retry_prompt)
                self.assertIn("already set to Lite", _prompt)
                report = workspace / "report" / "report.md"
                report.parent.mkdir(parents=True)
                report.write_text("finished report", encoding="utf-8")
                return {"messages": [_message("first", "done")]}

            bridge._invoke_agent = Mock(side_effect=invoke)
            result = bridge.run_native()

            self.assertEqual(bridge._invoke_agent.call_count, 1)
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.native_iterations, 1)
            self.assertEqual(result.usage.agent_calls, 1)

    def test_native_rubric_middleware_owns_revision_without_a_second_rac_invoke(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            bridge = _bridge(workspace)
            def invoke(_prompt, *, rubric, retry_prompt):
                # Simulate EvoScientist's internal grade-and-revise loop: the
                # outer RAC adapter observes only one completed invocation.
                self.assertIn("analysis artifact", rubric)
                self.assertIn("Objective:", retry_prompt)
                report = workspace / "report" / "report.md"
                report.parent.mkdir(parents=True)
                report.write_text("evidence-grounded report", encoding="utf-8")
                return {"messages": [_message("final", "Report written after native revision.")]}

            bridge._invoke_agent = Mock(side_effect=invoke)
            result = bridge.run_native()

            self.assertEqual(bridge._invoke_agent.call_count, 1)
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.native_iterations, 1)
            self.assertEqual(result.usage.agent_calls, 1)
            self.assertTrue((workspace / "state/evo_scientist/native_run.md").is_file())

    def test_missing_report_triggers_one_bounded_continuation(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            bridge = _bridge(workspace)

            def invoke(prompt, *, rubric, retry_prompt):
                if "Continue the supplied" in prompt:
                    report = workspace / "report" / "report.md"
                    report.parent.mkdir(parents=True)
                    report.write_text("continued report", encoding="utf-8")
                    return {"messages": [_message("continued", "report completed")]}
                return {"messages": [_message("first", "waiting for mode selection")]}

            bridge._invoke_agent = Mock(side_effect=invoke)

            result = bridge.run_native()

            self.assertEqual(bridge._invoke_agent.call_count, 2)
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.native_status, "completed")
            self.assertEqual(result.native_iterations, 2)
            transcript = (workspace / "state/evo_scientist/native_run.md").read_text()
            self.assertIn("Native invocation 1", transcript)
            self.assertIn("Native invocation 2", transcript)

    def test_missing_report_stops_after_bounded_continuation(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            bridge = _bridge(workspace)
            bridge._invoke_agent = Mock(return_value={
                "messages": [_message("final", "native rubric loop exhausted")],
            })

            result = bridge.run_native()

            self.assertEqual(bridge._invoke_agent.call_count, 2)
            self.assertEqual(result.status, "stop")
            self.assertEqual(result.native_status, "missing_report")
            self.assertEqual(result.native_iterations, 2)
            self.assertIn("native rubric loop", result.reason)


if __name__ == "__main__":
    unittest.main()
