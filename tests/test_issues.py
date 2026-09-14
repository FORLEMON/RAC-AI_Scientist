import unittest

from rac_ai_scientist.issues import extract_review_issues, parse_review_score


class IssueTests(unittest.TestCase):
    def test_review_parser_uses_stable_categories(self):
        text = "Overall Score: 6/10\nMajor issue: missing experiment baseline and confidence interval."
        issues = extract_review_issues(text)
        self.assertTrue(any(item.kind == "experiment" for item in issues))
        self.assertTrue(any(item.kind == "baseline_fairness" for item in issues))
        self.assertEqual(parse_review_score(text), 6.0)
