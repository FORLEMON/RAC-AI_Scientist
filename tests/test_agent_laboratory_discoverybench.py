import types
import unittest

from rac_ai_scientist.hosts.agent_laboratory import (
    AgentLaboratoryBridge,
    DISCOVERYBENCH_LITERATURE_SKIP_NOTE,
)


class AgentLaboratoryDiscoveryBenchTests(unittest.TestCase):
    @staticmethod
    def _bridge(benchmark_id: str):
        bridge = object.__new__(AgentLaboratoryBridge)
        bridge.task_spec = types.SimpleNamespace(benchmark_id=benchmark_id)
        bridge.workflow = types.SimpleNamespace(
            phase_status={
                "literature review": False,
                "plan formulation": False,
            },
            phd=types.SimpleNamespace(lit_review=""),
        )
        bridge.terminal = False
        return bridge

    def test_discoverybench_starts_at_plan_formulation(self):
        bridge = self._bridge("discoverybench")

        bridge._apply_benchmark_phase_overrides()

        self.assertTrue(bridge.workflow.phase_status["literature review"])
        self.assertEqual(bridge.workflow.phd.lit_review, DISCOVERYBENCH_LITERATURE_SKIP_NOTE)
        self.assertEqual(bridge._next_native(), "plan_formulation")

    def test_other_benchmarks_keep_literature_review(self):
        bridge = self._bridge("researchclawbench")

        bridge._apply_benchmark_phase_overrides()

        self.assertFalse(bridge.workflow.phase_status["literature review"])
        self.assertEqual(bridge.workflow.phd.lit_review, "")
        self.assertEqual(bridge._next_native(), "literature_review")


if __name__ == "__main__":
    unittest.main()
