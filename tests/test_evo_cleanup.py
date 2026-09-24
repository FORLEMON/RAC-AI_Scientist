import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from rac_ai_scientist.hosts.evo_scientist import (
    EvoScientistBridge,
    _content_filter_retry_prompt,
    _is_content_filter_error,
    _provider_error_usage,
)
from rac_ai_scientist.schemas import Budget, Usage


class EvoCleanupTests(unittest.TestCase):
    def test_evo_image_contains_analysis_dependencies_used_by_native_runs(self):
        dockerfile = (Path(__file__).parents[1] / 'docker/Dockerfile.evo-scientist').read_text()
        install = ' '.join(line for line in dockerfile.splitlines() if 'pip install' in line)
        self.assertIn('matplotlib', install.split())
        self.assertIn('pandas', install.split())
        self.assertIn('python -m pip freeze > /opt/integration/evo-environment.txt', dockerfile)

    def test_azure_deepseek_v4_models_declare_non_text_inputs_unsupported(self):
        for model in ('DeepSeek-V4-Pro', 'openai/DeepSeek-V4-Flash-0731'):
            with self.subTest(model=model):
                self._assert_model_profile(model, {'image_inputs': False, 'pdf_inputs': False})

    def test_other_deepseek_models_keep_their_native_profile(self):
        self._assert_model_profile('deepseek-v4.1-flash', None)

    def _assert_model_profile(self, model, expected_profile):
        captured = {}
        package = ModuleType('EvoScientist')
        package.__path__ = []
        config = ModuleType('EvoScientist.config')
        def make_config(**kwargs):
            captured['config'] = kwargs
            return SimpleNamespace(sandbox_execute_timeout=120, **kwargs)

        config.EvoScientistConfig = make_config
        llm = ModuleType('EvoScientist.llm')
        llm.get_chat_model = lambda **kwargs: captured.update(kwargs) or object()
        api = ModuleType('EvoScientist.EvoScientist')
        api._get_default_middleware = lambda *args, **kwargs: ['base-middleware']

        def create_cli_agent(**kwargs):
            captured['middleware'] = api._get_default_middleware(
                workspace_dir=kwargs['workspace_dir'],
                cfg=kwargs['config'],
                chat_model=kwargs['chat_model'],
            )
            return object()

        api.create_cli_agent = create_cli_agent
        backends = ModuleType('EvoScientist.backends')
        backends.CustomSandboxBackend = lambda **kwargs: SimpleNamespace(**kwargs)
        subagents = ModuleType('EvoScientist.subagents')
        subagents.__path__ = []
        factory = ModuleType('EvoScientist.subagents._factory')
        factory._scheduler_rubric_middleware = (
            lambda **kwargs: ('native-rubric', kwargs['model'], kwargs['backend'])
        )

        with tempfile.TemporaryDirectory() as raw, patch.dict(sys.modules, {
            'EvoScientist': package,
            'EvoScientist.config': config,
            'EvoScientist.llm': llm,
            'EvoScientist.EvoScientist': api,
            'EvoScientist.backends': backends,
            'EvoScientist.subagents': subagents,
            'EvoScientist.subagents._factory': factory,
        }):
            root = Path(raw)
            (root / 'upstream/EvoScientist').mkdir(parents=True)
            (root / 'upstream/EvoScientist/EvoScientist.py').write_text('')
            bridge = EvoScientistBridge(
                root / 'upstream',
                Path(__file__).parents[1] / 'configs/hosts/evo_scientist.json',
                Budget(20, 1_000_000, 100_000, 10, 100, 10),
                model,
                'fake',
            )
            bridge.initialize(episode_id='test', workspace=root / 'workspace', objective='test', seed=0)

        self.assertEqual(captured.get('profile'), expected_profile)
        self.assertTrue(captured['config']['auto_mode'])
        self.assertFalse(captured['config']['enable_ask_user'])
        self.assertEqual(captured['middleware'][0], 'base-middleware')
        self.assertEqual(captured['middleware'][-1][0], 'native-rubric')
        self.assertEqual(api._get_default_middleware.__name__, '<lambda>')

    def test_interpreters_close_on_success_and_provider_error(self):
        module = ModuleType('EvoScientist.middleware.code_interpreter')
        for failure in (None, RuntimeError('provider budget exhausted')):
            with self.subTest(failure=failure):
                module.aclose_code_interpreters = AsyncMock()
                bridge = object.__new__(EvoScientistBridge)
                bridge.episode_id = 'test-episode'
                bridge.agent = SimpleNamespace(invoke=Mock(return_value={'messages': []}, side_effect=failure))
                with patch.dict(sys.modules, {'EvoScientist.middleware.code_interpreter': module}):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, 'provider budget exhausted'):
                            bridge._invoke_agent('test')
                    else:
                        self.assertEqual(bridge._invoke_agent('test'), {'messages': []})
                module.aclose_code_interpreters.assert_awaited_once_with()

    def test_native_rubric_is_forwarded_as_top_level_graph_state(self):
        module = ModuleType('EvoScientist.middleware.code_interpreter')
        module.aclose_code_interpreters = AsyncMock()
        invoke = Mock(return_value={'messages': []})
        bridge = object.__new__(EvoScientistBridge)
        bridge.episode_id = 'native-rubric-test'
        bridge.agent = SimpleNamespace(invoke=invoke)

        with patch.dict(sys.modules, {'EvoScientist.middleware.code_interpreter': module}):
            bridge._invoke_agent('finish the study', rubric='report exists')

        payload = invoke.call_args.args[0]
        self.assertEqual(payload['rubric'], 'report exists')
        self.assertEqual(payload['messages'][0]['content'], 'finish the study')
        self.assertEqual(
            invoke.call_args.kwargs['config']['configurable']['thread_id'],
            'native-rubric-test',
        )

    def test_content_filter_retry_prompt_keeps_objective_without_room_history(self):
        prompt = _content_filter_retry_prompt(
            'derive a robust estimator',
            'research',
            None,
        )
        self.assertIn('derive a robust estimator', prompt)
        self.assertIn('research step', prompt)
        self.assertIn('files already present', prompt)
        self.assertNotIn('SharedNet', prompt)
        self.assertNotIn('jailbreak', prompt.lower())

    def test_content_filter_is_detected_and_serialized_usage_is_read(self):
        class Filtered(Exception):
            status_code = 400

        exc = Filtered(
            "ProviderStreamError: {'finish_reason': 'content_filter', "
            "'usage': {'prompt_tokens': 27010, 'completion_tokens': 0}}"
        )
        self.assertTrue(_is_content_filter_error(exc))
        self.assertEqual(_provider_error_usage(exc), (27010, 0))

    def test_content_filter_retries_once_with_fresh_thread_and_counts_failure(self):
        class Filtered(Exception):
            status_code = 400
            body = {
                'choices': [{'finish_reason': 'content_filter'}],
                'usage': {'prompt_tokens': 27_010, 'completion_tokens': 0},
            }

        module = ModuleType('EvoScientist.middleware.code_interpreter')
        module.aclose_code_interpreters = AsyncMock()
        invoke = Mock(side_effect=[Filtered('response content blocked'), {'messages': []}])
        bridge = object.__new__(EvoScientistBridge)
        bridge.episode_id = 'evo-filter-test'
        bridge.hop = 1
        bridge.usage = Usage()
        bridge.agent = SimpleNamespace(invoke=invoke)

        with patch.dict(sys.modules, {'EvoScientist.middleware.code_interpreter': module}):
            result = bridge._invoke_agent(
                'full Room-augmented prompt',
                retry_prompt='compact academic prompt',
            )

        self.assertEqual(result, {'messages': []})
        self.assertEqual(invoke.call_count, 2)
        first_payload = invoke.call_args_list[0].args[0]
        second_payload = invoke.call_args_list[1].args[0]
        self.assertEqual(first_payload['messages'][0]['content'], 'full Room-augmented prompt')
        self.assertEqual(second_payload['messages'][0]['content'], 'compact academic prompt')
        self.assertEqual(
            invoke.call_args_list[1].kwargs['config']['configurable']['thread_id'],
            'evo-filter-test-provider-retry-h1',
        )
        self.assertEqual(bridge.usage.input_tokens, 27_010)
        self.assertEqual(bridge.usage.output_tokens, 0)
        self.assertEqual(bridge.usage.agent_calls, 1)
        module.aclose_code_interpreters.assert_awaited_once_with()

    def test_second_content_filter_remains_terminal(self):
        class Filtered(Exception):
            status_code = 400

            def __init__(self):
                super().__init__('content_filter')
                self.body = {'usage': {'prompt_tokens': 5, 'completion_tokens': 0}}

        module = ModuleType('EvoScientist.middleware.code_interpreter')
        module.aclose_code_interpreters = AsyncMock()
        bridge = object.__new__(EvoScientistBridge)
        bridge.episode_id = 'evo-filter-terminal'
        bridge.hop = 2
        bridge.usage = Usage()
        bridge.agent = SimpleNamespace(invoke=Mock(side_effect=[Filtered(), Filtered()]))

        with patch.dict(sys.modules, {'EvoScientist.middleware.code_interpreter': module}):
            with self.assertRaises(Filtered):
                bridge._invoke_agent('full prompt', retry_prompt='compact prompt')

        self.assertEqual(bridge.agent.invoke.call_count, 2)
        self.assertEqual(bridge.usage.input_tokens, 5)
        self.assertEqual(bridge.usage.agent_calls, 1)
        module.aclose_code_interpreters.assert_awaited_once_with()
