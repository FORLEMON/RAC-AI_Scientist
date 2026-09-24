from __future__ import annotations

from fnmatch import fnmatch

from .conditions import Condition
from .schemas import (
    Action,
    CapabilityCard,
    Checkpoint,
    CheckResult,
    CoordinationDecision,
    EvidenceRequirement,
    InvocationResult,
    Verdict,
    Verification,
    WorkContract,
)


ISSUE_TAGS: dict[str, tuple[str, ...]] = {
    "experiment": ("experiment", "analysis"),
    "baseline_fairness": ("experiment", "analysis"),
    "methodology": ("planning", "experiment"),
    "figure": ("figure", "code"),
    "execution": ("debug", "code"),
    "citation": ("literature", "writing"),
    "writing": ("writing",),
    "artifact": ("debug", "code", "writing"),
}

TERMINAL_TAGS = ("terminal_review", "finalize")


class SharedPolicy:
    """Host-neutral cumulative N0--R3 coordination policy."""

    def __init__(self, condition: Condition | str, *, review_score_threshold: float = 8.0, task_spec=None):
        self.condition = Condition.parse(condition)
        self.review_score_threshold = review_score_threshold
        self.task_spec = task_spec

    def decide(self, checkpoint: Checkpoint, *, native_next: str | None = None) -> CoordinationDecision:
        if checkpoint.terminal:
            return CoordinationDecision(Action.STOP, None, "host reports a terminal state")
        if checkpoint.remaining_budget.exhausted():
            return CoordinationDecision(Action.STOP, None, "lifecycle budget exhausted")

        if not self.condition.enables("runtime_routing"):
            if native_next is None:
                return CoordinationDecision(Action.STOP, None, "native workflow has no successor")
            card = self._card(checkpoint, native_next)
            reason = "native fixed successor"
            if self.condition.enables("runtime_communication"):
                reason += " with SharedNet communication"
            return CoordinationDecision(Action.CONTINUE, card.capability_id, reason)

        card = self._select(checkpoint)
        if card is None:
            return CoordinationDecision(Action.ESCALATE, None, "no admitted capability matches the current state")
        contract = self._contract(checkpoint, card) if self.condition.enables("work_contracts") else None
        return CoordinationDecision(Action.CONTINUE, card.capability_id, "shared runtime selection", contract)

    def evaluate(
        self,
        checkpoint: Checkpoint,
        decision: CoordinationDecision,
        result: InvocationResult,
    ) -> CoordinationDecision:
        if result.error and "Error code: 402" in result.error:
            return CoordinationDecision(Action.STOP, None, "model transport rejected further requests (HTTP 402)", decision.contract)
        if result.error and result.terminal_error:
            return CoordinationDecision(Action.STOP, None, result.error, decision.contract)
        if not self.condition.enables("verifier"):
            if (
                result.timed_out
                and self.condition in {Condition.R1, Condition.R2}
                and result.metrics.get("native_timeout_continuable") == 1.0
            ):
                return CoordinationDecision(
                    Action.REVERIFY,
                    None,
                    "native capability timed out; mark failed and continue with the next capability",
                )
            if result.error or result.timed_out:
                return CoordinationDecision(Action.STOP, None, result.error or "capability timed out")
            if result.proposed_done:
                return CoordinationDecision(Action.STOP, None, "native workflow reports completion")
            return CoordinationDecision(Action.REVERIFY, None, "take a fresh checkpoint; verification disabled")

        if decision.contract is None:
            raise ValueError("R3+ requires a work contract")
        verification = verify_result(decision.contract, result)
        # R3 verification is advisory.  Its evidence remains in the ledger and
        # SharedNet context, but it never stops, retries, reroutes, or rolls back
        # an invocation.  The next loop takes a fresh checkpoint and performs a
        # normal runtime selection.  Transport-fatal errors above remain fatal.
        reason = f"verification advisory ({verification.verdict.value}): {verification.reason}"
        return CoordinationDecision(Action.REVERIFY, None, reason, decision.contract, verification)

    @staticmethod
    def _card(checkpoint: Checkpoint, capability_id: str) -> CapabilityCard:
        for card in checkpoint.capabilities:
            if card.capability_id == capability_id and card.available:
                return card
        raise ValueError(f"native successor {capability_id!r} is not an available capability")

    def _select(self, checkpoint: Checkpoint) -> CapabilityCard | None:
        cards = [card for card in checkpoint.capabilities if card.available]
        if not cards:
            return None
        open_issues = [issue for issue in checkpoint.issues if not issue.resolved]
        desired: list[str] = []
        if open_issues:
            issue = max(open_issues, key=lambda item: (item.attempts, item.issue_id))
            if issue.kind == "native_requirement" and issue.issue_id.startswith("native:"):
                native_capability_id = issue.issue_id.removeprefix("native:")
                native_card = next(
                    (card for card in cards if card.capability_id == native_capability_id),
                    None,
                )
                if native_card is not None:
                    return native_card
            desired.extend(issue.required_tags or ISSUE_TAGS.get(issue.kind, (issue.kind,)))
        else:
            kind = "terminal_submission" if self.task_spec and self.task_spec.benchmark_id != "researchclawbench" else "terminal_report"
            has_report = any(item.kind == kind for item in checkpoint.artifacts)
            desired.append("terminal_review" if has_report else "finalize")

        scored: list[tuple[int, str, CapabilityCard]] = []
        for card in cards:
            score = sum(1 for tag in desired if tag in card.tags)
            scored.append((score, card.capability_id, card))
        best = max(scored, key=lambda item: (item[0], -len(item[2].tags), item[1]))
        if best[0] == 0:
            # A planner is the shared decomposition fallback, expressed as a tag.
            planners = [card for card in cards if "planning" in card.tags]
            return sorted(planners, key=lambda item: item.capability_id)[0] if planners else None
        return best[2]

    def _contract(self, checkpoint: Checkpoint, card: CapabilityCard) -> WorkContract:
        scoped_issues = tuple(
            item for item in checkpoint.issues
            if not item.resolved and (
                item.issue_id == f"native:{card.capability_id}"
                or (not item.issue_id.startswith("native:") and
                    any(tag in card.tags for tag in (item.required_tags or ISSUE_TAGS.get(item.kind, (item.kind,)))))
            )
        )
        issue_ids = tuple(item.issue_id for item in scoped_issues)
        objective = checkpoint.objective
        if issue_ids:
            objective += "; resolve only these scoped issues:\n" + "\n".join(
                f"- {item.issue_id}: {item.summary}" for item in scoped_issues
            )
        if self.task_spec and self.task_spec.benchmark_id != "researchclawbench" and "finalize" in card.tags:
            evidence = (EvidenceRequirement("benchmark_submission"), EvidenceRequirement("nonempty_output"))
        elif "finalize" in card.tags and "terminal_review" in card.tags:
            evidence = (
                EvidenceRequirement("terminal_report", minimum_bytes=200),
                EvidenceRequirement("nonempty_output"),
            )
        elif "finalize" in card.tags:
            evidence = (EvidenceRequirement("terminal_report", minimum_bytes=200),)
        elif "terminal_review" in card.tags:
            evidence = (
                EvidenceRequirement("artifact_changed", ("review",), minimum_bytes=20),
                EvidenceRequirement("nonempty_output"),
            )
        else:
            evidence = (
                EvidenceRequirement("artifact_changed", card.produces, minimum_bytes=1),
                EvidenceRequirement("nonempty_output"),
            )
        return WorkContract(
            contract_id=f"{checkpoint.episode_id}:hop-{checkpoint.hop}:{card.capability_id}",
            capability_id=card.capability_id,
            objective=objective,
            readable_artifacts=card.readable_artifacts,
            writable_artifacts=card.writable_artifacts,
            required_evidence=evidence,
            issue_ids=issue_ids,
        )


def changed_artifacts(result: InvocationResult) -> list[str]:
    before = {item.relative_path: item.sha256 for item in result.artifacts_before}
    return sorted(
        item.relative_path
        for item in result.artifacts_after
        if before.get(item.relative_path) != item.sha256
    )


def verify_result(contract: WorkContract, result: InvocationResult) -> Verification:
    changed = set(changed_artifacts(result))
    after = {item.relative_path: item for item in result.artifacts_after}
    checks: list[CheckResult] = []
    unauthorized = sorted(path for path in changed if not any(fnmatch(path, pattern) for pattern in contract.writable_artifacts))
    # Host manifests cannot enumerate every native scratch/cache/output path.
    # Keep out-of-scope writes visible in the ledger, but do not turn them into
    # false verification failures or roll back otherwise valid host work.
    authority_detail = ""
    if unauthorized:
        authority_detail = "warning: writes outside declared authority: " + ", ".join(unauthorized)
    checks.append(CheckResult("declared_writes_only", True, authority_detail))
    for requirement in contract.required_evidence:
        if requirement.kind == "nonempty_output":
            passed = bool(result.output.strip())
            checks.append(CheckResult("nonempty_output", passed))
        elif requirement.kind == "artifact_changed":
            candidates = [item for item in result.artifacts_after if not requirement.artifact_kinds or item.kind in requirement.artifact_kinds]
            passed = any(item.relative_path in changed and item.size_bytes >= requirement.minimum_bytes for item in candidates)
            checks.append(CheckResult("artifact_changed", passed, ", ".join(sorted(changed))))
        elif requirement.kind == "artifact_exists":
            candidates = [item for item in after.values() if not requirement.artifact_kinds or item.kind in requirement.artifact_kinds]
            passed = any(item.size_bytes >= requirement.minimum_bytes for item in candidates)
            checks.append(CheckResult("artifact_exists", passed))
        elif requirement.kind == "benchmark_submission":
            checks.append(CheckResult("benchmark_submission", result.metrics.get("submission_valid") == 1.0))
        elif requirement.kind == "terminal_report":
            candidates = [item for item in after.values() if item.kind == "terminal_report"]
            passed = any(item.relative_path in changed and item.size_bytes >= requirement.minimum_bytes for item in candidates)
            checks.append(CheckResult("terminal_report", passed))
        else:  # pragma: no cover - dataclass Literal protects typed callers
            checks.append(CheckResult(str(requirement.kind), False, "unknown evidence requirement"))

    if result.error:
        checks.append(CheckResult("invocation_error", False, result.error))
    if result.timed_out:
        checks.append(CheckResult("invocation_timeout", False, "capability timed out"))
    passed = checks and all(item.passed for item in checks)
    if passed:
        reason = "all contract evidence checks passed"
        if authority_detail:
            reason += "; " + authority_detail
        return Verification(Verdict.SUPPORTED, reason, tuple(checks))
    if result.error or result.timed_out or any(not item.passed for item in checks):
        return Verification(Verdict.REFUTED, "one or more contract evidence checks failed", tuple(checks))
    return Verification(Verdict.INCONCLUSIVE, "evidence was insufficient", tuple(checks))
