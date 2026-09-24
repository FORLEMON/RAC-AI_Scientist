"""Fetch fixed benchmark revisions. Existing checkouts are verified, never reset."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("discoverybench", "corebench"), action="append")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from rac_ai_scientist.benchmarks.upstream import assert_revision
    lock = json.loads((root / "benchmark.lock.json").read_text(encoding="utf-8"))
    for name, spec in lock["benchmarks"].items():
        if args.only and name not in args.only:
            continue
        target = root / spec["directory"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", spec["url"], str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", spec["revision"]], check=True)
        assert_revision(target, name)
        print(f"[OK] {name} {spec['revision']}")


if __name__ == "__main__":
    main()
