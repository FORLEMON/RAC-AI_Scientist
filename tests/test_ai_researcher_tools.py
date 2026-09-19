import json
import unittest
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from rac_ai_scientist.hosts.ai_researcher import AIResearcherBridge
from rac_ai_scientist.manifest import capability_cards, load_host_manifest
from rac_ai_scientist.schemas import Budget, Usage


class ToolArgumentTests(unittest.TestCase):
    def test_planner_is_unavailable_before_survey_handoff(self):
        bridge = object.__new__(AIResearcherBridge)
        bridge.completed = {'idea'}
        bridge.terminal = False
        bridge.cards = capability_cards(load_host_manifest(Path(__file__).parents[1] / 'configs/hosts/ai_researcher.json'))
        available = {card.capability_id for card in bridge._available_cards() if card.available}
        self.assertEqual(available, {'idea', 'survey'})
        bridge.completed.add('survey')
        available = {card.capability_id for card in bridge._available_cards() if card.available}
        self.assertIn('implementation_plan', available)
        self.assertNotIn('implementation', available)

    def test_survey_hands_native_notes_to_planner_without_tool_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = self.phase_bridge(Path(directory))
            notes = [{'definition': 'Gaussian process', 'math_formula': 'K',
                      'code_implementation': 'fit(X, y)', 'reference_papers': ['paper.pdf'],
                      'reference_codebases': []}]
            survey = '# Survey\nGaussian-process calibration with measured data.'
            async def run(agent, messages, **kwargs):
                self.assertNotIn('max_turns', kwargs)  # Native max_turns counts messages, not API calls.
                if agent == 'survey':
                    self.assertEqual(kwargs['context_variables']['notes'], [])
                    self.assertIn('Research proposal', messages[0]['content'])
                    return SimpleNamespace(context_variables={'notes': notes}, messages=[
                        {'role': 'tool', 'name': 'read_file', 'content': 'raw file transcript'},
                        {'role': 'tool', 'name': 'case_resolved', 'content': survey}])
                self.assertEqual(kwargs['context_variables']['model_survey'], survey)
                self.assertIn(survey, messages[0]['content'])
                self.assertEqual(messages[0]['content'].count(survey), 1)
                return SimpleNamespace(context_variables={}, messages=[
                    {'role': 'tool', 'name': 'case_resolved', 'content': '# Dataset, model, training and testing plan\n' + survey}])
            bridge.client.run_async = run
            self.assertIsNone(bridge.invoke('survey', None).error)
            self.assertEqual((bridge.workspace / 'state/ai_researcher/survey.md').read_text(), survey)
            self.assertIsNone(bridge.invoke('implementation_plan', None).error)
            self.assertIn('implementation_plan', bridge.completed)
            self.assertIsNone(bridge.invoke('implementation', None).error)

    def test_empty_native_survey_receipt_cannot_complete_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = self.phase_bridge(Path(directory))
            bridge.client.run_async = AsyncMock(return_value=SimpleNamespace(
                context_variables={'notes': []}, messages=[{'role': 'tool',
                'name': 'case_resolved', 'content': 'I have merged the notes for the innovation.\nThe notes are as follows:\n'}]))
            result = bridge.invoke('survey', None)
            self.assertIn('without research notes', result.error)
            self.assertNotIn('survey', bridge.completed)
            self.assertNotIn('model_survey', bridge.context)
            self.assertFalse((bridge.workspace / 'state/ai_researcher/survey.md').exists())

    def test_native_phase_requires_successful_resolution(self):
        for message in (
            {'role': 'assistant', 'content': 'Still planning'},
            {'role': 'tool', 'name': 'case_resolved', 'content': "[Tool Call Error] 'model_survey'"},
            {'role': 'tool', 'name': 'case_not_resolved', 'content': 'No usable dataset'},
        ):
            with self.subTest(message=message), self.assertRaises(RuntimeError):
                AIResearcherBridge._response_output('implementation_plan', [message])

    @staticmethod
    def phase_bridge(workspace):
        bridge = object.__new__(AIResearcherBridge)
        bridge.workspace = workspace
        bridge.client = SimpleNamespace()
        bridge.agents = {name: name for name in ('survey', 'implementation_plan', 'implementation')}
        bridge.context = {'notes': []}
        bridge.completed = {'idea'}
        bridge.usage = Usage()
        bridge.initial_budget = Budget(25, 60_000_000, 1_300_000, 900, 21600, 33)
        bridge.model = 'fake'
        bridge.objective = 'Fit calibration to the supplied measurements'
        bridge.started = time.monotonic()
        bridge.hop = 0
        bridge._available_cards = lambda: [SimpleNamespace(capability_id=name, available=True) for name in bridge.agents]
        idea = workspace / 'state/ai_researcher/idea.md'
        idea.parent.mkdir(parents=True)
        idea.write_text('Research proposal', encoding='utf-8')
        return bridge

    def test_paper_output_keeps_only_the_final_assistant_report(self):
        messages = [
            {'role': 'tool', 'content': 'outputs/metrics.json: score=0.81'},
            {'role': 'assistant', 'content': '# Report\n\nThe measured score was 0.81.'},
        ]

        output = AIResearcherBridge._response_output('paper_writing', messages)

        self.assertEqual(output, '# Report\n\nThe measured score was 0.81.')
        self.assertNotIn('metrics.json:', output)

    def test_interrupted_writer_does_not_publish_earlier_text_or_none(self):
        for terminal in ({'role': 'tool', 'name': 'read_file', 'content': 'metrics'},
                         {'role': 'assistant', 'content': None},
                         {'role': 'assistant', 'content': 'reading', 'tool_calls': ['read_file']}):
            with self.subTest(terminal=terminal), self.assertRaises(RuntimeError):
                AIResearcherBridge._response_output('paper_writing', [
                    {'role': 'assistant', 'content': 'I will inspect the results'}, terminal])

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
            self.assertIn("survey: bad tool response", result.reason)

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
