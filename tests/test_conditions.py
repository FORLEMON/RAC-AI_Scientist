import unittest

from rac_ai_scientist.conditions import Condition


class ConditionTests(unittest.TestCase):
    def test_conditions_are_cumulative(self):
        self.assertFalse(Condition.R1.enables("work_contracts"))
        self.assertTrue(Condition.R2.enables("runtime_routing"))
        self.assertTrue(Condition.R5.enables("issue_aware_control"))

    def test_parse_rejects_unknown_condition(self):
        with self.assertRaises(ValueError):
            Condition.parse("R6")
