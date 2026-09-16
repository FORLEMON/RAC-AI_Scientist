import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from rac_ai_scientist.hosts.evo_scientist import EvoScientistBridge


class EvoCleanupTests(unittest.TestCase):
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
