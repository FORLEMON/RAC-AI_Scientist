import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from rac_ai_scientist.hosts import agent_laboratory


def edit_module():
    module = ModuleType('mlesolver')

    class Edit:
        def execute_command(self, *args):
            start, end, lines, replacement, dataset = args[0]
            for index in reversed(range(start, end + 1)):
                lines.pop(index)
            for line in reversed(replacement):
                lines.insert(start, line)
            return True, lines, 'native execution result'

    module.Edit = Edit
    return module


class EditRangeTests(unittest.TestCase):
    def test_invalid_indices_do_not_mutate_existing_code(self):
        module = edit_module()
        with patch.dict(sys.modules, {'mlesolver': module}):
            agent_laboratory._install_edit_range_validation()
            for start, end, initial in ((-1, 1, ['a', 'b']), (0, 2, ['a', 'b']),
                                         (1, 0, ['a', 'b']), (0, 0, [])):
                with self.subTest(start=start, end=end, initial=initial):
                    lines = list(initial)
                    success, updated, feedback = module.Edit().execute_command((start, end, lines, ['replacement'], ''))
                    self.assertFalse(success)
                    self.assertIsNone(updated)
                    self.assertIn('EDIT range', feedback)
                    self.assertEqual(lines, initial)

    def test_valid_edit_keeps_native_inclusive_line_replacement(self):
        module = edit_module()
        with patch.dict(sys.modules, {'mlesolver': module}):
            agent_laboratory._install_edit_range_validation()
            agent_laboratory._install_edit_range_validation()
            result = module.Edit().execute_command((1, 2, ['a', 'b', 'c'], ['replacement'], ''))
            self.assertEqual(result, (True, ['a', 'replacement'], 'native execution result'))


if __name__ == '__main__':
    unittest.main()
