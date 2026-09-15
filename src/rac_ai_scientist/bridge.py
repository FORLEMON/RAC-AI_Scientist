from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .schemas import Checkpoint, InvocationResult, NativeRunResult, WorkContract


class HostBridge(ABC):
    """Policy-free interface implemented by every AI-scientist host."""

    host_id: str

    @abstractmethod
    def initialize(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        """Initialize or resume host-native state in ``workspace``."""

    def initialize_native(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        """Initialize a host-owned N0 lifecycle without RAC phase scheduling.

        Hosts that need a distinct native setup may override this method. The
        default preserves compatibility for hosts whose normal initialization
        already prepares their complete native workflow.
        """
        self.initialize(episode_id=episode_id, workspace=workspace, objective=objective, seed=seed)

    def run_native(self) -> NativeRunResult:
        """Run the host's complete native scheduler for an N0 episode."""
        raise NotImplementedError(f"{type(self).__name__} does not expose a native N0 runner")

    @abstractmethod
    def checkpoint(self) -> Checkpoint:
        """Serialize current native state without choosing the next action."""

    @abstractmethod
    def native_next(self, checkpoint: Checkpoint) -> str | None:
        """Return the host's unmodified fixed-workflow successor for N0."""

    @abstractmethod
    def invoke(self, capability_id: str, contract: WorkContract | None) -> InvocationResult:
        """Invoke exactly one declared native capability."""


class BridgeContractError(RuntimeError):
    pass


def validate_bridge_checkpoint(checkpoint: Checkpoint) -> None:
    ids = [card.capability_id for card in checkpoint.capabilities]
    if len(ids) != len(set(ids)):
        raise BridgeContractError("capability IDs must be unique within a checkpoint")
    artifact_ids = [item.artifact_id for item in checkpoint.artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise BridgeContractError("artifact IDs must be unique within a checkpoint")
    if checkpoint.hop < 0:
        raise BridgeContractError("checkpoint hop cannot be negative")
