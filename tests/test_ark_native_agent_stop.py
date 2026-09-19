import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rac_ai_scientist.hosts.ark import ArkBridge


class ArkNativeAgentStopTests(unittest.TestCase):
    def bridge(self, workspace):
        bridge = object.__new__(ArkBridge)
        bridge.workspace = workspace
        bridge.native_mode = True
        bridge.orchestrator = types.SimpleNamespace(
            _terminal_error=None, _run_fatal=None, _agent_stats=[], iteration=0,
            load_paper_state=lambda: {"status": "in_progress", "current_score": 0})
        return bridge

    def test_terminal_native_error_cannot_reenter_writer_or_compile(self):
        for field in ("_terminal_error", "_run_fatal"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                bridge = self.bridge(Path(raw))
                events = []
                def agent(*args, **kwargs):
                    events.append("writer")
                    setattr(bridge.orchestrator, field, "APIError: request budget exhausted")
                    return ""
                bridge.orchestrator.run_agent = agent
                def run():
                    try:
                        for _ in range(10):
                            try:
                                bridge.orchestrator.run_agent("writer", "fix compilation")
                            except Exception:
                                events.append("native retry")
                            events.append("compile")
                        events.append("review phase")
                    finally:
                        events.append("native cleanup")
                bridge.orchestrator.run = run
                with patch.object(bridge, "_normalize_report") as normalize:
                    result = bridge.run_native()
                self.assertEqual(events, ["writer", "native cleanup"])
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.reason, "APIError: request budget exhausted")
                self.assertIs(bridge.orchestrator.run_agent, agent)
                normalize.assert_not_called()

    def test_normal_native_agents_continue_and_wrapper_is_restored(self):
        with tempfile.TemporaryDirectory() as raw:
            bridge = self.bridge(Path(raw))
            agent = Mock(return_value="native output")
            bridge.orchestrator.run_agent = agent
            def run():
                self.assertEqual(bridge.orchestrator.run_agent("writer", "task"), "native output")
                self.assertEqual(bridge.orchestrator.run_agent("reviewer", "review"), "native output")
            bridge.orchestrator.run = run
            with patch.object(bridge, "_normalize_report") as normalize:
                bridge.run_native()
            self.assertEqual(agent.call_count, 2)
            self.assertIs(bridge.orchestrator.run_agent, agent)
            normalize.assert_called_once()

    def test_original_baseexception_is_not_swallowed(self):
        class NativeAbort(BaseException):
            pass
        with tempfile.TemporaryDirectory() as raw:
            bridge = self.bridge(Path(raw))
            agent = Mock(side_effect=NativeAbort("original cancellation"))
            bridge.orchestrator.run_agent = agent
            bridge.orchestrator.run = lambda: bridge.orchestrator.run_agent("writer", "task")
            with self.assertRaisesRegex(NativeAbort, "original cancellation"):
                bridge.run_native()
            self.assertIs(bridge.orchestrator.run_agent, agent)


if __name__ == "__main__":
    unittest.main()
