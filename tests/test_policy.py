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

    def test_r2_routes_by_issue_without_contract(self):
        issue = Issue("I-1", "experiment", "missing baseline")
        decision = SharedPolicy(Condition.R2).decide(checkpoint([issue]))
        self.assertEqual(decision.capability_id, "run")
        self.assertIsNone(decision.contract)

    def test_r3_routes_by_issue_and_creates_contract(self):
        issue = Issue("I-1", "experiment", "missing baseline")
        decision = SharedPolicy(Condition.R3).decide(checkpoint([issue]))
        self.assertEqual(decision.capability_id, "run")
        self.assertIsNotNone(decision.contract)
        self.assertEqual(decision.contract.issue_ids, ("I-1",))

    def test_r1_uses_native_fixed_successor_with_communication(self):
        decision = SharedPolicy(Condition.R1).decide(checkpoint(), native_next="write")
        self.assertEqual(decision.capability_id, "write")
        self.assertIsNone(decision.contract)
        self.assertIn("SharedNet", decision.reason)

    def test_r1_timeout_marks_native_step_failed_without_stopping_episode(self):
        policy = SharedPolicy(Condition.R1)
        cp = checkpoint()
        decision = policy.decide(cp, native_next="run")
        result = InvocationResult(
            "run",
            "",
            (),
            (),
            Usage(),
            timed_out=True,
            metrics={"native_timeout_continuable": 1.0},
        )
        evaluated = policy.evaluate(cp, decision, result)
        self.assertIs(evaluated.action, Action.REVERIFY)
        self.assertIn("mark failed", evaluated.reason)

    def test_r2_continuable_timeout_marks_native_step_failed_without_stopping_episode(self):
        policy = SharedPolicy(Condition.R2)
        cp = checkpoint([Issue("I-1", "experiment", "missing baseline")])
        decision = policy.decide(cp)
        result = InvocationResult(
            "run",
            "",
            (),
            (),
            Usage(),
            timed_out=True,
            metrics={"native_timeout_continuable": 1.0},
        )
        evaluated = policy.evaluate(cp, decision, result)
        self.assertIs(evaluated.action, Action.REVERIFY)
        self.assertIn("next capability", evaluated.reason)

    def test_r2_timeout_without_native_continue_marker_still_stops(self):
        policy = SharedPolicy(Condition.R2)
        cp = checkpoint([Issue("I-1", "experiment", "missing baseline")])
        decision = policy.decide(cp)
        result = InvocationResult("run", "", (), (), Usage(), timed_out=True)
        self.assertIs(policy.evaluate(cp, decision, result).action, Action.STOP)

    def test_r1_timeout_without_native_continue_marker_still_stops(self):
        policy = SharedPolicy(Condition.R1)
        cp = checkpoint()
        decision = policy.decide(cp, native_next="run")
        result = InvocationResult("run", "", (), (), Usage(), timed_out=True)
        self.assertIs(policy.evaluate(cp, decision, result).action, Action.STOP)

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

        for condition in (Condition.R1, Condition.R2, Condition.R3):
            with self.subTest(condition=condition.value):
                decision = SharedPolicy(condition).decide(
                    cp,
                    native_next="researcher" if condition is Condition.R1 else None,
                )
                self.assertEqual(decision.capability_id, "researcher")

    def test_native_requirement_falls_back_to_tags_when_named_capability_is_unavailable(self):
        issue = Issue(
            "native:researcher",
            "native_requirement",
            "research capability is unavailable",
            required_tags=("planning",),
        )
        cp = Checkpoint("ep", 0, "answer", "researcher", [], [issue], budget(), cards())
        self.assertEqual(SharedPolicy(Condition.R2).decide(cp).capability_id, "plan")

    def test_ark_experimenter_contract_allows_generated_report_figures(self):
        config = Path(__file__).resolve().parents[1] / "configs" / "hosts" / "ark.json"
        capabilities = capability_cards(load_host_manifest(config))
        issue = Issue(
            "native:experimenter",
            "native_requirement",
            "experimenter work remains",
            required_tags=("experiment",),
        )
        cp = Checkpoint("ep", 1, "answer", "experimenter", [], [issue], budget(), capabilities)
        decision = SharedPolicy(Condition.R3).decide(cp)
        self.assertEqual(decision.capability_id, "experimenter")
        self.assertIsNotNone(decision.contract)

        result = InvocationResult(
            "experimenter",
            "completed experiment",
            [],
            [
                ArtifactRecord("result", "results/result.json", "result", "new-result", 20),
                ArtifactRecord("figure", "report/images/result.pdf", "figure", "new-figure", 20),
            ],
        )
        verification = verify_result(decision.contract, result)
        self.assertEqual(verification.verdict.value, "supported")

    def test_every_host_native_stage_keeps_its_named_capability_for_r1_through_r3(self):
        host_configs = Path(__file__).resolve().parents[1] / "configs" / "hosts"
        conditions = (Condition.R1, Condition.R2, Condition.R3)

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
                        decision = SharedPolicy(condition).decide(
                            cp,
                            native_next=card.capability_id if condition is Condition.R1 else None,
                        )
                        self.assertEqual(decision.capability_id, card.capability_id)

    def test_r3_timeout_verification_is_advisory(self):
        cp = checkpoint([Issue("I-1", "experiment", "missing baseline")])
        policy = SharedPolicy(Condition.R3)
        decision = policy.decide(cp)
        before = ArtifactRecord("r", "results", "result", "old", 20)
        after = ArtifactRecord("r", "results", "result", "new", 20)
        result = InvocationResult("run", "", [before], [after], timed_out=True)
        evaluated = policy.evaluate(cp, decision, result)
        self.assertIs(evaluated.action, Action.REVERIFY)
        self.assertEqual(evaluated.verification.verdict.value, "refuted")

    def test_mixed_review_issues_are_scoped_to_the_selected_role(self):
        capabilities = capability_cards(load_host_manifest(
            Path(__file__).resolve().parents[1] / "configs/hosts/ark.json"))
        issues = [Issue(f"review:{kind}", kind, f"fix {kind}")
                  for kind in ("figure", "methodology", "writing", "execution")]
        cp = Checkpoint("ep", 4, "finish research", "writer", [], issues, budget(), capabilities)
        policy = SharedPolicy(Condition.R3)
        decision = policy.decide(cp)
        self.assertEqual(decision.capability_id, "writer")
        self.assertEqual(decision.contract.issue_ids, ("review:writing",))
        self.assertIn("only these scoped issues", decision.contract.objective)
        self.assertIn("review:writing: fix writing", decision.contract.objective)
        self.assertNotIn("review:execution", decision.contract.objective)
        self.assertNotIn("fix execution", decision.contract.objective)
        issues[-1].attempts = 1
        decision = policy.decide(cp)
        self.assertEqual(decision.capability_id, "coder")
        self.assertEqual(set(decision.contract.issue_ids), {"review:figure", "review:execution"})

    def test_authority_violation_is_a_warning_when_required_evidence_exists(self):
        for condition in (Condition.R3,):
            with self.subTest(condition=condition):
                cp = checkpoint()
                policy = SharedPolicy(condition)
                decision = policy.decide(cp)
                result = InvocationResult("write", "done", [], [
                    ArtifactRecord("report", "report/report.md", "terminal_report", "new", 300),
                    ArtifactRecord("code", "code/main.py", "code", "new", 20),
                ])
                evaluated = policy.evaluate(cp, decision, result)
                self.assertIs(evaluated.action, Action.REVERIFY)
                self.assertEqual(evaluated.verification.verdict.value, "supported")
                authority = next(item for item in evaluated.verification.checks if item.name == "declared_writes_only")
                self.assertTrue(authority.passed)
                self.assertIn("warning:", authority.detail)

    def test_error_cannot_be_overridden_by_satisfied_artifact_evidence(self):
        cp = checkpoint()
        policy = SharedPolicy(Condition.R3)
        decision = policy.decide(cp)
        report = ArtifactRecord("r", "report/report.md", "terminal_report", "new", 300)
        result = InvocationResult("write", "apparently done", [], [report], error="provider failed", proposed_done=True)
        verification = verify_result(decision.contract, result)
        self.assertEqual(verification.verdict.value, "refuted")
        self.assertIn("invocation_error", [item.name for item in verification.checks])

    def test_provider_http_402_stops_instead_of_rerouting(self):
        policy = SharedPolicy(Condition.R3)
        cp = checkpoint([Issue("I-1", "experiment", "needs code")])
        decision = policy.decide(cp)
        result = InvocationResult(
            decision.capability_id or "", "", [], [],
            error="ProviderStreamError: Error code: 402 - request limit reached",
        )
        evaluated = policy.evaluate(cp, decision, result)
        self.assertIs(evaluated.action, Action.STOP)
        self.assertIn("402", evaluated.reason)

    def test_native_terminal_error_stops_without_retry_or_recovery(self):
        cp = checkpoint()
        for condition in (Condition.R3,):
            policy = SharedPolicy(condition)
            decision = policy.decide(cp)
            result = InvocationResult("write", "", [], [
                ArtifactRecord("report", "report", "terminal_report", "new", 300),
            ], error="native provider rejected the request", terminal_error=True)
            self.assertIs(policy.evaluate(cp, decision, result).action, Action.STOP)

    def test_verifier_warns_but_accepts_undeclared_write_with_required_evidence(self):
        contract = SharedPolicy(Condition.R3).decide(checkpoint()).contract
        self.assertIsNotNone(contract)
        report = ArtifactRecord("r", "report/report.md", "terminal_report", "new", 300)
        secret = ArtifactRecord("s", "data/input.csv", "state", "changed", 20)
        result = InvocationResult("write", "done", [], [report, secret])
        verification = verify_result(contract, result)
        self.assertEqual(verification.verdict.value, "supported")
        self.assertTrue(verification.checks[0].passed)
        self.assertIn("warning:", verification.checks[0].detail)

    def test_r3_missing_expected_change_is_advisory_and_selects_afresh(self):
        cp = checkpoint()
        policy = SharedPolicy(Condition.R3)
        decision = policy.decide(cp)
        result = InvocationResult(
            "write",
            "claimed completion",
            [],
            [ArtifactRecord("scratch", "scratch/notes.txt", "state", "new", 20)],
        )
        evaluated = policy.evaluate(cp, decision, result)
        self.assertIs(evaluated.action, Action.REVERIFY)
        self.assertIsNone(evaluated.capability_id)
        self.assertEqual(evaluated.verification.verdict.value, "refuted")

    def test_terminal_report_must_change_in_current_invocation(self):
        contract = SharedPolicy(Condition.R3).decide(checkpoint()).contract
        self.assertIsNotNone(contract)
        before = ArtifactRecord("r", "report/report.md", "terminal_report", "same", 300)
        result = InvocationResult("write", "done", [before], [before])
        self.assertEqual(verify_result(contract, result).verdict.value, "refuted")

    def test_r3_refuted_capability_returns_to_runtime_selection(self):
        cp = checkpoint([Issue("I-1", "experiment", "still missing", attempts=2)])
        policy = SharedPolicy(Condition.R3)
        first = policy.decide(cp)
        result = InvocationResult(first.capability_id or "", "", [], [])
        self.assertIs(policy.evaluate(cp, first, result).action, Action.REVERIFY)

    def test_r3_supported_proposed_completion_is_still_advisory(self):
        cp = checkpoint()
        policy = SharedPolicy(Condition.R3)
        decision = policy.decide(cp)
        report = ArtifactRecord("r", "report/report.md", "terminal_report", "new", 300)
        result = InvocationResult("write", "done", [], [report], proposed_done=True)
        evaluated = policy.evaluate(cp, decision, result)
        self.assertIs(evaluated.action, Action.REVERIFY)
        self.assertEqual(evaluated.verification.verdict.value, "supported")
