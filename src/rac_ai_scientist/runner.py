from __future__ import annotations

from dataclasses import dataclass

from .artifacts import WorkspaceTransaction
from .bridge import HostBridge, validate_bridge_checkpoint
from .conditions import Condition
from .ledger import JsonlLedger
from .policy import SharedPolicy
from .schemas import Action, CoordinationDecision, to_jsonable


@dataclass(frozen=True)
class EpisodeOutcome:
    status: str
    hops: int
    reason: str


class EpisodeRunner:
    """Condition-aware loop; all decisions are recorded before execution."""

    def __init__(self, bridge: HostBridge, condition: Condition | str, ledger: JsonlLedger, *, review_score_threshold: float = 8.0):
        self.bridge = bridge
        self.condition = Condition.parse(condition)
        self.policy = SharedPolicy(self.condition, review_score_threshold=review_score_threshold)
        self.ledger = ledger
        self._issue_sightings: dict[str, int] = {}

    def run(self, *, hard_hop_limit: int) -> EpisodeOutcome:
        if hard_hop_limit <= 0:
            raise ValueError("hard_hop_limit must be positive")
        pending: CoordinationDecision | None = None
        invocations = 0
        while invocations < hard_hop_limit:
            checkpoint = self.bridge.checkpoint()
            validate_bridge_checkpoint(checkpoint)
            for issue in checkpoint.issues:
                if issue.resolved:
                    continue
                key = f"{issue.kind}:{issue.issue_id}"
                self._issue_sightings[key] = self._issue_sightings.get(key, 0) + 1
                issue.attempts = max(issue.attempts, self._issue_sightings[key] - 1)
            self.ledger.append({"type": "checkpoint", "condition": self.condition.name, "payload": checkpoint})

            if checkpoint.terminal or checkpoint.remaining_budget.exhausted():
                decision = self.policy.decide(checkpoint)
            elif pending is None or pending.action in {Action.REROUTE, Action.REVERIFY}:
                native = self.bridge.native_next(checkpoint) if not self.condition.enables("runtime_routing") else None
                decision = self.policy.decide(checkpoint, native_next=native)
            elif pending.action is Action.RETRY:
                decision = pending
            else:
                decision = pending

            self.ledger.append({"type": "decision", "hop": invocations, "payload": decision})
            if decision.action is Action.STOP and not checkpoint.terminal and checkpoint.remaining_budget.exhausted():
                return EpisodeOutcome("budget_exhausted", invocations, decision.reason)
            if decision.action in {Action.STOP, Action.ABSTAIN, Action.ESCALATE}:
                return EpisodeOutcome(decision.action.value, invocations, decision.reason)
            if not decision.capability_id:
                return EpisodeOutcome("invalid", invocations, "decision did not name a capability")

            transaction = WorkspaceTransaction(self.bridge.transaction_workspace())
            try:
                result = self.bridge.invoke(decision.capability_id, decision.contract)
                self.ledger.append({"type": "invocation", "hop": invocations, "payload": result})
                pending = self.policy.evaluate(checkpoint, decision, result)
                self.ledger.append({"type": "evaluation", "hop": invocations, "payload": pending})
                invocations += 1

                if pending.action in {Action.RETRY, Action.REROUTE, Action.ESCALATE}:
                    transaction.rollback()
                    self.bridge.reject_invocation(result, pending)
                elif pending.action is Action.RECOVER:
                    writable = decision.contract.writable_artifacts if decision.contract is not None else ()
                    transaction.rollback(preserve_patterns=writable)
                    self.bridge.reject_invocation(result, pending)
                    pending = None
                elif result.error or result.timed_out:
                    transaction.rollback()
                    if result.timed_out and pending.action is Action.REVERIFY:
                        self.bridge.fail_invocation(result, pending)
                    else:
                        self.bridge.reject_invocation(result, pending)
                else:
                    self.bridge.accept_invocation(result, pending)
            except Exception:
                transaction.rollback()
                raise
            finally:
                transaction.close()

            if pending is not None and pending.action is Action.ESCALATE:
                return EpisodeOutcome("escalate", invocations, pending.reason)
            if pending is not None and pending.action is Action.STOP:
                if result.error and "Error code: 402" in result.error:
                    return EpisodeOutcome("budget_exhausted", invocations, pending.reason)
                if result.error or result.timed_out:
                    return EpisodeOutcome("timed_out" if result.timed_out else "failed", invocations, pending.reason)
                return EpisodeOutcome("completed", invocations, pending.reason)
        return EpisodeOutcome("budget_exhausted", invocations, "hard hop limit reached")
