"""Load only the pinned upstream definitions needed by preparation/scoring.

No upstream command-line entrypoints, downloads or host imports are executed.
The function/class AST bodies are unchanged; only import-time dependencies are
provided explicitly. Hashes apply to LF-normalized UTF-8 source.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess

SOURCES = {
    "corebench": ("hal/benchmarks/corebench.py", "c93daf73f81d74c8984a74feee802b3e0fb61d09366fc57ba3b90a44e2056900"),
    "discoverybench": ("eval/new_eval.py", "83406ddbe6894429cc5012b17cb82a209d5397aa7863b0aba1bb51edf1dd6c65"),
}


def assert_revision(source: Path, benchmark: str) -> None:
    from .corebench import REVISION as CORE_REVISION
    from .discoverybench import REVISION as DISCOVERY_REVISION
    expected = {"corebench": CORE_REVISION, "discoverybench": DISCOVERY_REVISION}[benchmark]
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=20).stdout.strip()
    if revision != expected:
        raise ValueError(f"{benchmark} checkout must be pinned to {expected}")
    paths = ["hal/benchmarks/corebench.py"] if benchmark == "corebench" else ["discoverybench/real", "eval", "utils"]
    # Windows checkouts may inherit core.autocrlf from the user's Git config;
    # a Linux scorer container does not inherit that global config. Compare
    # normalized tracked contents consistently without modifying the checkout.
    changed = subprocess.run(["git", "-c", "core.autocrlf=true", "-C", str(source), "status", "--porcelain", "--", *paths],
                             check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    if changed:
        raise ValueError(f"{benchmark} inputs/scorer differ from pinned checkout")


def definitions(source: Path, benchmark: str, namespace: dict) -> dict:
    relative, expected = SOURCES[benchmark]
    path = source / relative
    text = path.read_text(encoding="utf-8")
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual != expected:
        raise ValueError(f"pinned {benchmark} source hash mismatch: {relative}")
    tree = ast.parse(text, filename=str(path))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    namespace = {"__name__": f"rac_upstream_{benchmark}", **namespace}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def core_harness(source: Path, *, scoring: bool = False):
    import json
    import logging
    import math
    import os
    from typing import Any, Dict, Optional

    class BaseBenchmark:
        def _normalize_agent_output(self, value):
            return value

    namespace = dict(json=json, os=os, math=math, logger=logging.getLogger(__name__),
                     Any=Any, Dict=Dict, Optional=Optional, BaseBenchmark=BaseBenchmark)
    if scoring:
        import numpy
        from scipy.stats import t
        namespace.update(np=numpy, t=t)
    module = definitions(source, "corebench", namespace)
    return object.__new__(module["CoreBenchHard"])
