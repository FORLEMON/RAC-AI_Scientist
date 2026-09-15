import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from rac_ai_scientist.hosts.ai_researcher import AIResearcherBridge
from rac_ai_scientist.schemas import Usage


class ToolArgumentTests(unittest.TestCase):
    def test_empty_encoding_is_normalized_only_for_zero_input_schema(self):
        calls = [SimpleNamespace(function=SimpleNamespace(name=name, arguments=raw))
                 for name, raw in [('page_down', ''), ('read', ''), ('unknown', ''), ('page_down', '{')]]
        result = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=calls))])
        tools = [
            {'type': 'function', 'function': {'name': 'page_down', 'parameters': {'type': 'object', 'properties': {}}}},
            {'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}},
        ]
        AIResearcherBridge._normalize_empty_tool_arguments(result, tools)
        self.assertEqual([call.function.arguments for call in calls], ['{}', '', '', '{'])

    def test_native_error_is_failed_even_when_report_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = object.__new__(AIResearcherBridge)
            bridge.workspace = Path(directory)
            bridge.client = object()
            bridge.usage = Usage()
            bridge.objective = "test task"
            bridge.completed = set()
            bridge.invoke = lambda *args: SimpleNamespace(error="bad tool response")
            (bridge.workspace / "report").mkdir()
            (bridge.workspace / "report/report.md").write_text("partial draft")
            result = bridge.run_native()
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.native_status, "failed")
            self.assertEqual(result.native_iterations, 1)

    def test_tool_request_is_not_a_completed_research_report(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = object.__new__(AIResearcherBridge)
            bridge.workspace = Path(directory)
            bridge.client = object()
            bridge.usage = Usage()
            bridge.objective = "test task"
            bridge.completed = set()
            bridge.invoke = lambda *args: SimpleNamespace(error=None)
            (bridge.workspace / "report").mkdir()
            (bridge.workspace / "report/report.md").write_text('```json\n{"action":"read","path":"/tmp/workspace/plans"}\n```')
            result = bridge.run_native()
            self.assertEqual(result.status, "stop")
            self.assertIn("tool request", result.reason)
            self.assertEqual(result.metrics["report_is_tool_request"], 1)

    def test_report_and_experiment_artifacts_are_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = object.__new__(AIResearcherBridge)
            bridge.workspace = Path(directory)
            bridge.client = object()
            bridge.usage = Usage()
            bridge.objective = "test task"
            bridge.completed = set()
            bridge.invoke = lambda *args: SimpleNamespace(error=None)
            for folder, name, contents in (("report", "report.md", "# Methods\nExperiment and evidence."),
                                           ("code", "test.py", "print(1)"),
                                           ("outputs", "metric.csv", "score,1")):
                (bridge.workspace / folder).mkdir()
                (bridge.workspace / folder / name).write_text(contents)
            result = bridge.run_native()
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.metrics["code_files"], 1)
            self.assertEqual(result.metrics["result_files"], 1)

    def test_invalid_arguments_are_returned_without_executing_tool(self):
        executed = []
        def native_handle(calls, *args, **kwargs):
            for call in calls:
                executed.append(json.loads(call.function.arguments))
            return SimpleNamespace(messages=[], context_variables={"retained": True})
        logger = SimpleNamespace(_warp_args=lambda raw: str(json.loads(raw).items()))
        client = SimpleNamespace(logger=logger, handle_tool_calls=native_handle)
        AIResearcherBridge._install_tool_argument_adapter(client)
        for raw in ("", "{", "[]", "null"):
            with self.subTest(raw=raw):
                call = SimpleNamespace(id="bad", function=SimpleNamespace(name="open_local_file", arguments=raw))
                self.assertIn("rejected", client.logger._warp_args(raw))
                result = client.handle_tool_calls([call], [], {})
                self.assertEqual(result.messages[0]["tool_call_id"], "bad")
                self.assertIn("[Tool Call Error]", result.messages[0]["content"])
                self.assertEqual(executed, [])
        corrected = SimpleNamespace(id="good", function=SimpleNamespace(name="open_local_file", arguments='{"path":"data.json"}'))
        result = client.handle_tool_calls([corrected], [], {})
        self.assertEqual(executed, [{"path": "data.json"}])
        self.assertEqual(result.messages, [])
        self.assertTrue(result.context_variables["retained"])


if __name__ == '__main__':
    unittest.main()
