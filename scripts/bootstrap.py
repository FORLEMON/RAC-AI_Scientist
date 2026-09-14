#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "upstream.lock.json"


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clone reproducibility-pinned upstream repositories")
    parser.add_argument("--allow-floating", action="store_true", help="allow unresolved revisions for development only")
    parser.add_argument("--only", action="append", default=[], help="bootstrap only the named upstream")
    args = parser.parse_args(argv)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    destination_root = ROOT / "upstreams"
    destination_root.mkdir(exist_ok=True)
    failed = False
    for name, spec in lock["upstreams"].items():
        if args.only and name not in args.only:
            continue
        revision = spec.get("revision")
        if not revision and not args.allow_floating:
            if spec.get("required_for_live", True):
                print(f"[BLOCKED] {name}: revision is unresolved", file=sys.stderr)
                failed = True
            else:
                print(f"[SKIP] {name}: optional source has no accessible frozen revision")
            continue
        destination = destination_root / name
        if not destination.exists():
            run(["git", "clone", "--filter=blob:none", spec["url"], str(destination)])
        if revision:
            run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision])
            run(["git", "-C", str(destination), "checkout", "--detach", revision])
        else:
            print(f"[FLOATING] {name}: using clone default branch")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
