#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
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
    parser.add_argument(
        "--windows-zip-normalization",
        action="store_true",
        help="match archive names whose Windows extractor replaced illegal filename characters with underscores",
    )
    parser.add_argument(
        "--allow-windows-omissions",
        action="store_true",
        help="allow files with Windows-illegal names to be absent and read those blobs from Git when computing the canonical hash",
    )
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
    local_for_expected = {path: path for path in expected_paths}
    if args.windows_zip_normalization:
        def portable(path: str) -> str:
            return "/".join(re.sub(r'[:*?"<>|]', "_", part).rstrip(". ") for part in path.split("/"))
        local_for_expected = {path: portable(path) for path in expected_paths}
    mapped_actual = set(local_for_expected.values())
    missing = sorted(path for path, local in local_for_expected.items() if local not in actual_paths)
    extra = sorted(actual_paths - mapped_actual)
    changed = sorted(path for path, local in local_for_expected.items()
                     if local in actual_paths and blob_oid(args.snapshot / local) != entries[path])
    allowed_omissions = args.allow_windows_omissions and all(
        re.search(r'[:*?"<>|]', path) for path in missing
    )
    if (missing and not allowed_omissions) or extra or changed:
        print(f"[MISMATCH] missing={len(missing)} extra={len(extra)} changed={len(changed)}")
        for label, paths in (("missing", missing), ("extra", extra), ("changed", changed)):
            for path in paths[:20]:
                print(f"  {label}: {path}")
        return 1
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(entries):
        local = args.snapshot / local_for_expected[path]
        encoded = path.encode("utf-8")
        if local.is_file():
            content = local.read_bytes()
        else:
            safe_repo = str(args.git_repository.resolve()).replace("\\", "/")
            command = ["git", "-c", f"safe.directory={safe_repo}", "-C",
                       str(args.git_repository), "cat-file", "blob", entries[path]]
            content = subprocess.run(
                command,
                check=True,
                capture_output=True,
            ).stdout
        content_hash = hashlib.sha256(content).digest()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(content_hash)
        total_bytes += len(content)
    state = "OK-WITH-WINDOWS-OMISSIONS" if missing else "OK"
    print(f"[{state}] snapshot matches {args.revision}: files={len(entries)} omitted={len(missing)} bytes={total_bytes} tree_sha256={digest.hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
