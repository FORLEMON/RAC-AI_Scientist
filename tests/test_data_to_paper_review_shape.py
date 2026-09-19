import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from rac_ai_scientist.hosts import data_to_paper


class ReviewCorrection(Exception):
    pass


def review_module():
    module = ModuleType('data_to_paper.base_steps.request_code')

    class RequestIssuesToSolutions:
        def _check_response_value(self, value):
            # Pinned native unpacking after Dict[str, List[str]] validation.
            for key, (kind, feedback) in value.items():
                value[key] = (kind.upper(), feedback)
            return value

        def _raise_self_response_error(self, *, title, error_message):
            raise ReviewCorrection(error_message)

    module.RequestIssuesToSolutions = RequestIssuesToSolutions
    return module


class ReviewShapeTests(unittest.TestCase):
    def test_wrong_length_uses_native_correction_instead_of_unpacking(self):
        module = review_module()
        with patch.dict(sys.modules, {'data_to_paper.base_steps.request_code': module}):
            data_to_paper._install_review_shape_validation()
            checker = module.RequestIssuesToSolutions()
            for assessment in ([], ['OK'], ['CONCERN', 'First issue', 'CONCERN', 'Second issue']):
                with self.subTest(assessment=assessment), self.assertRaisesRegex(ReviewCorrection, '2'):
                    checker._check_response_value({'Numeric values': assessment})

    def test_valid_pair_keeps_native_normalization(self):
        module = review_module()
        with patch.dict(sys.modules, {'data_to_paper.base_steps.request_code': module}):
            data_to_paper._install_review_shape_validation()
            data_to_paper._install_review_shape_validation()
            self.assertEqual(module.RequestIssuesToSolutions()._check_response_value(
                {'Uncertainty': ['ok', 'Confidence intervals are reported.']}),
                {'Uncertainty': ('OK', 'Confidence intervals are reported.')})


if __name__ == '__main__':
    unittest.main()
