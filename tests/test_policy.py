import unittest
from pathlib import Path

from rac_ai_scientist.conditions import Condition
from rac_ai_scientist.manifest import capability_cards, load_host_manifest
from rac_ai_scientist.policy import SharedPolicy, verify_result
from rac_ai_scientist.schemas import (
    Action,
    ArtifactRecord,
    Budget,
    CapabilityCard,
    Checkpoint,
    EvidenceRequirement,
    InvocationResult,
    Issue,
    Usage,
)


def budget():
    return Budget(10.0, 1000, 1000, 10, 100.0, 10)


def cards():
    return [
        CapabilityCard("plan", "decompose", ("planning",), (), ("state",), ("plan",)),
        CapabilityCard("run", "experiment", ("experiment", "analysis"), ("data",), ("results",), ("result",)),
        CapabilityCard("write", "write", ("writing", "finalize"), ("results",), ("report",), ("terminal_report",)),
    ]


def checkpoint(issues=None):
    return Checkpoint("ep", 1, "answer", "native", [], issues or [], budget(), cards())


class PolicyTests(unittest.TestCase):
    def test_n0_preserves_native_successor(self):
        decision = SharedPolicy(Condition.N0).decide(checkpoint(), native_next="write")
        self.assertEqual(decision.capability_id, "write")
        self.assertIsNone(decision.contract)

    def test_native_proposed_completion_stops_without_shared_verifier(self):
        policy = SharedPolicy(Condition.N0)
        decision = policy.decide(checkpoint(), native_next="write")
        result = InvocationResult("write", "done", (), (), Usage(), proposed_done=True)
        self.assertIs(policy.evaluate(checkpoint(), decision, result).action, Action.STOP)

    def test_r2_routes_by_issue_and_creates_contract(self):
        issue = Issue("I-1", "experiment", "missing baseline")
        decision = SharedPolicy(Condition.R2).decide(checkpoint([issue]))
        self.assertEqual(decision.capability_id, "run")
        self.assertIsNotNone(decision.contract)
        self.assertEqual(decision.contract.issue_ids, ("I-1",))

    def test_native_requirement_prefers_named_capability_over_tag_tiebreak(self):
        capabilities = [
            CapabilityCard("researcher", "frame research", ("planning", "literature"), (), (), ("plan",)),
            CapabilityCard("planner", "decompose review", ("planning",), (), (), ("plan",)),
        ]
        issue = Issue(
            "native:researcher",
            "native_requirement",
            "initial research framing is incomplete",
            required_tags=("planning",),
        )
        cp = Checkpoint("ep", 0, "answer", "researcher", [], [issue], budget(), capabilities)

        for condition in (Condition.R1, Condition.R2, Condition.R3, Condition.R4, Condition.R5):
            with self.subTest(condition=condition.value):
                decision = SharedPolicy(condition).decide(cp)
                self.assertEqual(decision.capability_id, "researcher")

    def test_native_requirement_falls_back_to_tags_when_named_capability_is_unavailable(self):
        issue = Issue(
            "native:researcher",
            "native_requirement",
            "research capability is unavailable",
            required_tags=("planning",),
        )
        cp = Checkpoint("ep", 0, "answer", "researcher", [], [issue], budget(), cards())
        self.assertEqual(SharedPolicy(Condition.R1).decide(cp).capability_id, "plan")

    def test_every_host_native_stage_keeps_its_named_capability_for_r1_through_r5(self):
        host_configs = Path(__file__).resolve().parents[1] / "configs" / "hosts"
        conditions = (Condition.R1, Condition.R2, Condition.R3, Condition.R4, Condition.R5)

        for config_path in sorted(host_configs.glob("*.json")):
            capabilities = capability_cards(load_host_manifest(config_path))
            for card in capabilities:
                issue = Issue(
                    f"native:{card.capability_id}",
                    "native_requirement",
                    f"{card.capability_id} remains incomplete",
                    required_tags=card.tags[:1],
                )
                cp = Checkpoint(
                    "ep",
                    0,
                    "answer",
                    card.capability_id,
                    [],
                    [issue],
                    budget(),
                    capabilities,
                )
                for condition in conditions:
                    with self.subTest(
                        host=config_path.stem,
                        capability=card.capability_id,
                        condition=condition.value,
                    ):
                        decision = SharedPolicy(condition).decide(cp)
                        self.assertEqual(decision.capability_id, card.capability_id)

    def test_r4_recovers_changed_artifact_after_timeout(self):
        cp = checkpoint([Issue("I-1", "experiment", "missing baseline")])
        policy = SharedPolicy(Condition.R4)
        decision = policy.decide(cp)
        before = ArtifactRecord("r", "outputs/result.json", "result", "old", 20)
        after = ArtifactRecord("r", "outputs/result.json", "result", "new", 20)
        result = InvocationResult("run", "", [before], [after], timed_out=True)
        evaluated = policy.evaluate(cp, decision, result)
        self.assertIs(evaluated.action, Action.RECOVER)
        self.assertEqual(evaluated.verification.verdict.value, "refuted")

    def test_error_cannot_be_overridden_by_satisfied_artifact_evidence(self):
        cp = checkpoint()
        policy = SharedPolicy(Condition.R3)
        decision = policy.decide(cp)
        report = ArtifactRecord("r", "report/report.md", "terminal_report", "new", 300)
        result = InvocationResult("write", "apparently done", [], [report], error="provider failed", proposed_done=True)
        verification = verify_result(decision.contract, result)
        self.assertEqual(verification.verdict.value, "refuted")
        self.assertIn("invocation_error", [item.name for item in verification.checks])

    def test_verifier_rejects_undeclared_canonical_write(self):
        contract = SharedPolicy(Condition.R3).decide(checkpoint()).contract
        self.assertIsNotNone(contract)
        report = ArtifactRecord("r", "report/report.md", "terminal_report", "new", 300)
        secret = ArtifactRecord("s", "data/input.csv", "state", "changed", 20)
        result = InvocationResult("write", "done", [], [report, secret])
        verification = verify_result(contract, result)
        self.assertEqual(verification.verdict.value, "refuted")
        self.assertFalse(verification.checks[0].passed)

    def test_terminal_report_must_change_in_current_invocation(self):
        contract = SharedPolicy(Condition.R3).decide(checkpoint()).contract
        self.assertIsNotNone(contract)
        before = ArtifactRecord("r", "report/report.md", "terminal_report", "same", 300)
        result = InvocationResult("write", "done", [before], [before])
        self.assertEqual(verify_result(contract, result).verdict.value, "refuted")

    def test_r5_reroute_avoids_failed_capability_once(self):
        cp = checkpoint([Issue("I-1", "experiment", "still missing", attempts=2)])
        policy = SharedPolicy(Condition.R5)
        first = policy.decide(cp)
        result = InvocationResult(first.capability_id or "", "", [], [])
        self.assertIs(policy.evaluate(cp, first, result).action, Action.REROUTE)
        rerouted = policy.decide(cp)
        self.assertNotEqual(rerouted.capability_id, first.capability_id)
