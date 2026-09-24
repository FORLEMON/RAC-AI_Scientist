from __future__ import annotations

import hashlib
from pathlib import Path


IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def expected_tree_hash(spec: dict, actual: str, *, packaged: bool = False, snapshot: bool = False) -> str | None:
    """Choose an explicitly locked tree, never accept an unlisted runtime tree.

    A packaged host may include pinned submodules or historical server patches;
    a fresh checkout still has to match the canonical Git tree.
    """
    if packaged and (spec.get("runtime_tree_sha256") or spec.get("runtime_tree_sha256s")):
        allowed = [spec["runtime_tree_sha256"]] if spec.get("runtime_tree_sha256") else []
        allowed.extend(spec.get("runtime_tree_sha256s", []))
        return actual if actual in allowed else allowed[0]
    if snapshot and spec.get("snapshot_tree_sha256"):
        return spec["snapshot_tree_sha256"]
    return spec.get("tree_sha256")


def _is_generated_package_metadata(relative: Path) -> bool:
    return any(part.endswith((".egg-info", ".dist-info")) for part in relative.parts)


def tree_hash(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or IGNORED_PARTS.intersection(relative.parts)
            or _is_generated_package_metadata(relative)
            or path.suffix in IGNORED_SUFFIXES
        ):
            continue
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        encoded_path = relative.as_posix().encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(file_digest.digest())
        count += 1
        size += path.stat().st_size
    return digest.hexdigest(), count, size
