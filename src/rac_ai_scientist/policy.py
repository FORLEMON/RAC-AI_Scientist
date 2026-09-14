from __future__ import annotations

from collections import Counter
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
    """Host-neutral cumulative N0--R5 coordination policy."""

    def __init__(self, condition: Condition | str, *, review_score_threshold: float = 8.0):
        self.condition = Condition.parse(condition)
        self.review_score_threshold = review_score_threshold
        self._avoid_once: set[str] = set()

    def decide(self, checkpoint: Checkpoint, *, native_next: str | None = None) -> CoordinationDecision:
        if checkpoint.terminal:
            return CoordinationDecision(Action.STOP, None, "host reports a terminal state")
        if checkpoint.remaining_budget.exhausted():
            return CoordinationDecision(Action.STOP, None, "lifecycle budget exhausted")

        if self.condition is Condition.N0:
            if native_next is None:
                return CoordinationDecision(Action.STOP, None, "native workflow has no successor")
            card = self._card(checkpoint, native_next)
            return CoordinationDecision(Action.CONTINUE, card.capability_id, "native fixed successor")

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
        if not self.condition.enables("verifier"):
            if result.error or result.timed_out:
                return CoordinationDecision(Action.STOP, None, result.error or "capability timed out")
            if result.proposed_done:
                return CoordinationDecision(Action.STOP, None, "native workflow reports completion")
            return CoordinationDecision(Action.REVERIFY, None, "take a fresh checkpoint; verification disabled")

        if decision.contract is None:
            raise ValueError("R3+ requires a work contract")
        verification = verify_result(decision.contract, result)
        if verification.verdict is Verdict.SUPPORTED:
            card = self._card(checkpoint, decision.capability_id or "")
            if "terminal_review" in card.tags and result.metrics.get("review_score", float("-inf")) >= self.review_score_threshold:
                return CoordinationDecision(Action.STOP, None, "quality threshold and evidence contract are satisfied", decision.contract, verification)
            action = Action.STOP if result.proposed_done else Action.REVERIFY
            return CoordinationDecision(action, None, verification.reason, decision.contract, verification)

        changed = changed_artifacts(result)
        if self.condition.enables("recovery") and changed and (result.timed_out or not result.output.strip()):
            return CoordinationDecision(
                Action.RECOVER,
                result.proposed_next,
                "preserve verified partial artifacts after interrupted/empty return",
                decision.contract,
                verification,
            )

        if self.condition.enables("issue_aware_control"):
            attempts: Counter[str] = Counter()
            for item in checkpoint.issues:
                if not item.resolved:
                    attempts[item.kind] = max(attempts[item.kind], item.attempts)
            if any(value >= 2 for value in attempts.values()):
                if decision.capability_id:
                    self._avoid_once.add(decision.capability_id)
                return CoordinationDecision(Action.REROUTE, None, "repeated unresolved issue requires a different capability", decision.contract, verification)

        return CoordinationDecision(Action.RETRY, decision.capability_id, verification.reason, decision.contract, verification)

    @staticmethod
    def _card(checkpoint: Checkpoint, capability_id: str) -> CapabilityCard:
        for card in checkpoint.capabilities:
            if card.capability_id == capability_id and card.available:
                return card
        raise ValueError(f"native successor {capability_id!r} is not an available capability")

    def _select(self, checkpoint: Checkpoint) -> CapabilityCard | None:
        cards = [card for card in checkpoint.capabilities if card.available and card.capability_id not in self._avoid_once]
        self._avoid_once.clear()
        if not cards:
            return None
        open_issues = [issue for issue in checkpoint.issues if not issue.resolved]
        desired: list[str] = []
        if open_issues:
            issue = max(open_issues, key=lambda item: (item.attempts, item.issue_id))
            desired.extend(issue.required_tags or ISSUE_TAGS.get(issue.kind, (issue.kind,)))
        else:
            has_report = any(item.kind == "terminal_report" for item in checkpoint.artifacts)
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

    @staticmethod
    def _contract(checkpoint: Checkpoint, card: CapabilityCard) -> WorkContract:
        issue_ids = tuple(item.issue_id for item in checkpoint.issues if not item.resolved)
        objective = checkpoint.objective
        if issue_ids:
            objective += "; resolve issues " + ", ".join(issue_ids)
        if "finalize" in card.tags and "terminal_review" in card.tags:
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
    checks.append(CheckResult("declared_writes_only", not unauthorized, ", ".join(unauthorized)))
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
        elif requirement.kind == "terminal_report":
            candidates = [item for item in after.values() if item.kind == "terminal_report"]
            passed = any(item.relative_path in changed and item.size_bytes >= requirement.minimum_bytes for item in candidates)
            checks.append(CheckResult("terminal_report", passed))
        else:  # pragma: no cover - dataclass Literal protects typed callers
            checks.append(CheckResult(str(requirement.kind), False, "unknown evidence requirement"))

    if result.error and not changed:
        checks.append(CheckResult("invocation_error", False, result.error))
    passed = checks and all(item.passed for item in checks)
    if passed:
        return Verification(Verdict.SUPPORTED, "all contract evidence checks passed", tuple(checks))
    if result.error or result.timed_out or any(not item.passed for item in checks):
        return Verification(Verdict.REFUTED, "one or more contract evidence checks failed", tuple(checks))
    return Verification(Verdict.INCONCLUSIVE, "evidence was insufficient", tuple(checks))
