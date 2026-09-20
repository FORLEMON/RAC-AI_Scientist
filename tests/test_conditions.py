import unittest

from rac_ai_scientist.conditions import Condition


class ConditionTests(unittest.TestCase):
    def test_conditions_are_cumulative(self):
        self.assertTrue(Condition.R1.enables("runtime_communication"))
        self.assertFalse(Condition.R1.enables("runtime_routing"))
        self.assertTrue(Condition.R2.enables("runtime_routing"))
        self.assertFalse(Condition.R2.enables("work_contracts"))
        self.assertTrue(Condition.R3.enables("work_contracts"))
        self.assertTrue(Condition.R3.enables("verifier"))

    def test_parse_rejects_unknown_condition(self):
        with self.assertRaises(ValueError):
            Condition.parse("R4")
