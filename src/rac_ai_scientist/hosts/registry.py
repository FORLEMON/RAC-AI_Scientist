from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any


HOST_SPECS = {
    "ark": ("ARK", "rac_ai_scientist.hosts.ark", "ArkBridge"),
    "agent_laboratory": ("AgentLaboratory-main", "rac_ai_scientist.hosts.agent_laboratory", "AgentLaboratoryBridge"),
    "evo_scientist": ("EvoScientist-main", "rac_ai_scientist.hosts.evo_scientist", "EvoScientistBridge"),
}

HOST_IDS = tuple(HOST_SPECS)
SHAREDNET_HOST_IDS = (
    "ark",
    "agent_laboratory",
    "evo_scientist",
)


def local_snapshot_name(host_id: str) -> str:
    return HOST_SPECS[host_id][0]


def make_bridge(host_id: str, upstream: Path, manifest: Path, budget: Any, model: str, api_key: str):
    _, module_name, class_name = HOST_SPECS[host_id]
    bridge_type = getattr(import_module(module_name), class_name)
    return bridge_type(upstream, manifest, budget, model, api_key)
