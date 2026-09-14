from __future__ import annotations

import hashlib
from pathlib import Path


IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def tree_hash(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if not path.is_file() or IGNORED_PARTS.intersection(relative.parts) or path.suffix in IGNORED_SUFFIXES:
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
