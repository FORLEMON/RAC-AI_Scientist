from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import CapabilityCard


FORBIDDEN_POLICY_KEYS = {
    "route",
    "routing",
    "retry",
    "acceptance",
    "threshold",
    "verifier",
    "recovery",
    "issue_map",
}


class ManifestError(ValueError):
    pass


def load_host_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManifestError("host manifest must be a JSON object")
    forbidden = sorted(FORBIDDEN_POLICY_KEYS.intersection(data))
    if forbidden:
        raise ManifestError("host manifest contains RAC-owned policy keys: " + ", ".join(forbidden))
    if not data.get("host_id") or not isinstance(data.get("capabilities"), list):
        raise ManifestError("host manifest requires host_id and capabilities")
    ids = [item.get("capability_id") for item in data["capabilities"]]
    if None in ids or len(ids) != len(set(ids)):
        raise ManifestError("host capability IDs must be present and unique")
    return data


def capability_cards(manifest: dict[str, Any]) -> list[CapabilityCard]:
    cards: list[CapabilityCard] = []
    for raw in manifest["capabilities"]:
        cards.append(
            CapabilityCard(
                capability_id=raw["capability_id"],
                description=raw.get("description", ""),
                tags=tuple(raw.get("tags", [])),
                readable_artifacts=tuple(raw.get("readable_artifacts", [])),
                writable_artifacts=tuple(raw.get("writable_artifacts", [])),
                produces=tuple(raw.get("produces", [])),
                available=bool(raw.get("available", True)),
                authority_scope=tuple(raw.get("authority_scope", [])),
                native_successors=tuple(raw.get("native_successors", [])),
            )
        )
    return cards
