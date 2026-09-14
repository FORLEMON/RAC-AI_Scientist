import unittest

from rac_ai_scientist.hosts.agent_laboratory import _response_cost


class Response:
    _hidden_params = {"response_cost": 0.125}
    model_extra = None


class UsageTests(unittest.TestCase):
    def test_provider_cost_is_extracted_when_available(self):
        self.assertEqual(_response_cost(Response()), 0.125)
