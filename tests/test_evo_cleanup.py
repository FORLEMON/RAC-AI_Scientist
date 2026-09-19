import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from rac_ai_scientist.hosts.evo_scientist import EvoScientistBridge
from rac_ai_scientist.schemas import Budget


class EvoCleanupTests(unittest.TestCase):
    def test_evo_image_contains_analysis_dependencies_used_by_native_runs(self):
        dockerfile = (Path(__file__).parents[1] / 'docker/Dockerfile.evo-scientist').read_text()
        install = next(line for line in dockerfile.splitlines() if 'pip install' in line)
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
        config.EvoScientistConfig = lambda **kwargs: SimpleNamespace(**kwargs)
        llm = ModuleType('EvoScientist.llm')
        llm.get_chat_model = lambda **kwargs: captured.update(kwargs) or object()
        api = ModuleType('EvoScientist.EvoScientist')
        api.create_cli_agent = lambda **kwargs: object()

        with tempfile.TemporaryDirectory() as raw, patch.dict(sys.modules, {
            'EvoScientist': package,
            'EvoScientist.config': config,
            'EvoScientist.llm': llm,
            'EvoScientist.EvoScientist': api,
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
