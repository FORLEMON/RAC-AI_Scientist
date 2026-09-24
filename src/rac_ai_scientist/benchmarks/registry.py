from __future__ import annotations

BENCHMARK_IDS = ("researchclawbench", "discoverybench", "corebench")


def get_adapter(benchmark_id: str):
    from .researchclawbench import ResearchClawBenchAdapter
    from .discoverybench import DiscoveryBenchAdapter
    from .corebench import CoreBenchAdapter
    classes = {"researchclawbench": ResearchClawBenchAdapter, "discoverybench": DiscoveryBenchAdapter, "corebench": CoreBenchAdapter}
    try:
        return classes[benchmark_id]()
    except KeyError as exc:
        raise ValueError(f"unknown benchmark: {benchmark_id}") from exc
