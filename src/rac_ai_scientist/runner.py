from __future__ import annotations

from dataclasses import dataclass

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
        for hop in range(hard_hop_limit):
            checkpoint = self.bridge.checkpoint()
            validate_bridge_checkpoint(checkpoint)
            for issue in checkpoint.issues:
                if issue.resolved:
                    continue
                key = f"{issue.kind}:{issue.issue_id}"
                self._issue_sightings[key] = self._issue_sightings.get(key, 0) + 1
                issue.attempts = max(issue.attempts, self._issue_sightings[key] - 1)
            self.ledger.append({"type": "checkpoint", "condition": self.condition.name, "payload": checkpoint})

            if pending is None or pending.action in {Action.REROUTE, Action.REVERIFY}:
                native = self.bridge.native_next(checkpoint) if self.condition is Condition.N0 else None
                decision = self.policy.decide(checkpoint, native_next=native)
            elif pending.action is Action.RETRY:
                decision = pending
            elif pending.action is Action.RECOVER:
                pending = None
                continue
            else:
                decision = pending

            self.ledger.append({"type": "decision", "hop": hop, "payload": decision})
            if decision.action in {Action.STOP, Action.ABSTAIN, Action.ESCALATE}:
                return EpisodeOutcome(decision.action.value, hop, decision.reason)
            if not decision.capability_id:
                return EpisodeOutcome("invalid", hop, "decision did not name a capability")

            result = self.bridge.invoke(decision.capability_id, decision.contract)
            self.ledger.append({"type": "invocation", "hop": hop, "payload": result})
            pending = self.policy.evaluate(checkpoint, decision, result)
            self.ledger.append({"type": "evaluation", "hop": hop, "payload": pending})
            if pending.action is Action.STOP:
                return EpisodeOutcome("completed", hop + 1, pending.reason)
        return EpisodeOutcome("budget_exhausted", hard_hop_limit, "hard hop limit reached")
