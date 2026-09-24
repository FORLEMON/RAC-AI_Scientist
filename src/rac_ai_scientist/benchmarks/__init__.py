"""Benchmark inputs and submissions; private references belong only to scorers."""

from .base import TaskSpec, ScoreResult, load_prepared, copy_prepared
from .registry import BENCHMARK_IDS, get_adapter

__all__ = ["TaskSpec", "ScoreResult", "load_prepared", "copy_prepared", "BENCHMARK_IDS", "get_adapter"]
