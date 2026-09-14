#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


def blob_oid(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode("ascii"), usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare an extracted source tree with a Git revision without checking out blobs")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("git_repository", type=Path)
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    raw = subprocess.run(
        ["git", "-C", str(args.git_repository), "ls-tree", "-rz", args.revision],
        check=True,
        capture_output=True,
    ).stdout
    entries: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        if kind == "blob" and mode != "120000":
            entries[encoded_path.decode("utf-8")] = oid
    actual_paths = {
        item.relative_to(args.snapshot).as_posix()
        for item in args.snapshot.rglob("*")
        if item.is_file() and ".git" not in item.relative_to(args.snapshot).parts
    }
    expected_paths = set(entries)
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    changed = sorted(path for path in expected_paths & actual_paths if blob_oid(args.snapshot / path) != entries[path])
    if missing or extra or changed:
        print(f"[MISMATCH] missing={len(missing)} extra={len(extra)} changed={len(changed)}")
        for label, paths in (("missing", missing), ("extra", extra), ("changed", changed)):
            for path in paths[:20]:
                print(f"  {label}: {path}")
        return 1
    print(f"[OK] snapshot matches {args.revision}: files={len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
