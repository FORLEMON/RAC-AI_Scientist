import os
import unittest

from rac_ai_scientist.judge import assert_complete_score, configure_researchclawbench_scorer


class _ScoreModule:
    LLMAgent = object

    @staticmethod
    def multi_thread(*args, **kwargs):
        raise AssertionError("the Azure adapter should replace this")


class JudgeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.old_provider = os.environ.get("JUDGE_PROVIDER")
        self.old_workers = os.environ.get("JUDGE_MAX_WORKERS")

    def tearDown(self):
        for key, value in {
            "JUDGE_PROVIDER": self.old_provider,
            "JUDGE_MAX_WORKERS": self.old_workers,
        }.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_azure_adapter_replaces_agent_and_serializes(self):
        os.environ["JUDGE_PROVIDER"] = "azure"
        os.environ["JUDGE_MAX_WORKERS"] = "1"
        module = _ScoreModule()
        configure_researchclawbench_scorer(module)
        self.assertEqual(module.LLMAgent.__name__, "AzureChatCompletionsJudge")
        self.assertEqual(module.multi_thread([{"value": 1}], lambda value: value + 1), [2])

    def test_azure_adapter_rejects_parallel_workers(self):
        os.environ["JUDGE_PROVIDER"] = "azure"
        os.environ["JUDGE_MAX_WORKERS"] = "2"
        with self.assertRaisesRegex(ValueError, "JUDGE_MAX_WORKERS=1"):
            configure_researchclawbench_scorer(_ScoreModule())

    def test_failed_provider_response_is_not_a_zero_score(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete judge result"):
            assert_complete_score(
                {"items": [{"score": 0, "reasoning": "Failed to parse scoring response."}]}
            )

    def test_valid_zero_score_remains_valid(self):
        assert_complete_score({"items": [{"score": 0, "reasoning": "The report does not establish this claim."}]})

    def test_top_level_error_is_not_a_score(self):
        with self.assertRaisesRegex(RuntimeError, "top-level error"):
            assert_complete_score({"error": "HTTP 429", "items": []})

    def test_empty_checklist_is_not_a_complete_score(self):
        with self.assertRaisesRegex(RuntimeError, "no checklist items"):
            assert_complete_score({"items": []})
