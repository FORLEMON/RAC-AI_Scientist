#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rac_ai_scientist.provenance import tree_hash

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute a deterministic source-tree fingerprint")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    for raw in args.paths:
        path = Path(raw).resolve()
        digest, count, size = tree_hash(path)
        print(f"{digest}  files={count} bytes={size}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
