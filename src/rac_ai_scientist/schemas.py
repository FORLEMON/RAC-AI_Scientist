from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class Verdict(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class Action(str, Enum):
    CONTINUE = "continue"
    RETRY = "retry"
    RECOVER = "recover"
    REROUTE = "reroute"
    REVERIFY = "reverify"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"
    STOP = "stop"


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    relative_path: str
    kind: str
    sha256: str
    size_bytes: int
    produced_by: str | None = None

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact paths must be workspace-relative and non-traversing")


@dataclass(frozen=True)
class EvidenceRequirement:
    kind: Literal[
        "artifact_changed",
        "artifact_exists",
        "nonempty_output",
        "terminal_report",
    ]
    artifact_kinds: tuple[str, ...] = ()
    minimum_bytes: int = 1


@dataclass(frozen=True)
class CapabilityCard:
    capability_id: str
    description: str
    tags: tuple[str, ...]
    readable_artifacts: tuple[str, ...]
    writable_artifacts: tuple[str, ...]
    produces: tuple[str, ...] = ()
    available: bool = True
    authority_scope: tuple[str, ...] = ()
    native_successors: tuple[str, ...] = ()


@dataclass
class Issue:
    issue_id: str
    kind: str
    summary: str
    affected_artifacts: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    acceptance_checks: tuple[str, ...] = ()
    required_tags: tuple[str, ...] = ()
    attempts: int = 0
    resolved: bool = False


@dataclass(frozen=True)
class Budget:
    provider_cost_usd: float
    input_tokens: int
    output_tokens: int
    agent_calls: int
    wall_seconds: float
    hops: int

    def exhausted(self) -> bool:
        return any(
            value <= 0
            for value in (
                self.provider_cost_usd,
                self.input_tokens,
                self.output_tokens,
                self.agent_calls,
                self.wall_seconds,
                self.hops,
            )
        )


@dataclass
class Checkpoint:
    episode_id: str
    hop: int
    objective: str
    native_stage: str
    artifacts: list[ArtifactRecord]
    issues: list[Issue]
    remaining_budget: Budget
    capabilities: list[CapabilityCard]
    history: list[dict[str, Any]] = field(default_factory=list)
    terminal: bool = False


@dataclass(frozen=True)
class WorkContract:
    contract_id: str
    capability_id: str
    objective: str
    readable_artifacts: tuple[str, ...]
    writable_artifacts: tuple[str, ...]
    required_evidence: tuple[EvidenceRequirement, ...]
    issue_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Usage:
    provider_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    agent_calls: int = 0
    wall_seconds: float = 0.0
    cost_source: str = "unavailable"
    token_source: str = "unavailable"


@dataclass
class NativeRunResult:
    """Outcome reported by a host after it runs its own native scheduler."""

    status: str
    reason: str
    native_iterations: int
    artifacts_before: list[ArtifactRecord]
    artifacts_after: list[ArtifactRecord]
    usage: Usage = field(default_factory=Usage)
    native_status: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class InvocationResult:
    capability_id: str
    output: str
    artifacts_before: list[ArtifactRecord]
    artifacts_after: list[ArtifactRecord]
    usage: Usage = field(default_factory=Usage)
    timed_out: bool = False
    error: str | None = None
    proposed_next: str | None = None
    proposed_done: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    terminal_error: bool = False


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class Verification:
    verdict: Verdict
    reason: str
    checks: tuple[CheckResult, ...]


@dataclass(frozen=True)
class CoordinationDecision:
    action: Action
    capability_id: str | None
    reason: str
    contract: WorkContract | None = None
    verification: Verification | None = None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
