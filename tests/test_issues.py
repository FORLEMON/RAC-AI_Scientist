import unittest

from rac_ai_scientist.issues import extract_review_issues, parse_review_score


class ReviewScoreTests(unittest.TestCase):
    def test_review_parser_uses_stable_categories(self):
        text = "Overall Score: 6/10\nMajor issue: missing experiment baseline and confidence interval."
        issues = extract_review_issues(text)
        self.assertTrue(any(item.kind == "experiment" for item in issues))
        self.assertTrue(any(item.kind == "baseline_fairness" for item in issues))
        self.assertEqual(parse_review_score(text), 6.0)

    def test_parses_total_equals_format_from_full_review(self):
        self.assertEqual(parse_review_score("Total = 7.7 / 10"), 7.7)

    def test_parses_markdown_score_table(self):
        self.assertEqual(parse_review_score("| **Total** | **8.1/10** | Weighted |"), 8.1)


if __name__ == "__main__":
    unittest.main()
