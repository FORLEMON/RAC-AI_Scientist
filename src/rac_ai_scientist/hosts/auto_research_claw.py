from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..artifacts import snapshot_workspace
from ..bridge import HostBridge
from ..issues import extract_review_issues, parse_review_score
from ..manifest import capability_cards, load_host_manifest
from ..reproducibility import seed_runtime
from ..schemas import Budget, Checkpoint, InvocationResult, Issue, NativeRunResult, Usage, WorkContract


ORDER = ("scope", "literature", "synthesis", "design", "experiment", "analysis", "writing", "finalize")
STAGE_RANGES = {
    "scope": (1, 2), "literature": (3, 6), "synthesis": (7, 8),
    "design": (9, 11), "experiment": (12, 13), "analysis": (14, 15),
    "writing": (16, 19), "finalize": (20, 23),
}
REQUIRED_TAGS = {
    "scope": ("planning",), "literature": ("literature",),
    "synthesis": ("planning",), "design": ("experiment", "planning"),
    "experiment": ("experiment", "code"), "analysis": ("analysis",),
    "writing": ("writing", "terminal_review"), "finalize": ("finalize",),
}


class AutoResearchClawBridge(HostBridge):
    """Thin phase bridge over AutoResearchClaw's native 23-stage executor."""

    host_id = "auto_research_claw"

    def __init__(self, upstream: Path, manifest: Path, budget: Budget, model: str, api_key: str):
        self.upstream = upstream.resolve()
        self.manifest = load_host_manifest(manifest)
        self.cards = capability_cards(self.manifest)
        self.initial_budget = budget
        self.model, self.api_key = model, api_key
        self.workspace: Path | None = None
        self.run_dir: Path | None = None
        self.config: Any = None
        self.adapters: Any = None
        self.episode_id = self.objective = ""
        self.completed: set[str] = set()
        self.hop = 0
        self.started = 0.0
        self.terminal = False
        self.usage = Usage()
        self.open_issues: list[Issue] = []

    def initialize(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        if not (self.upstream / "researchclaw" / "pipeline" / "runner.py").is_file():
            raise FileNotFoundError(f"AutoResearchClaw checkout not found: {self.upstream}")
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.workspace / "auto_research_claw_native"
        self.run_dir.mkdir(exist_ok=True)
        self.episode_id, self.objective = episode_id, objective
        self.started = time.monotonic()
        seed_runtime(seed)
        os.environ["OPENAI_API_KEY"] = self.api_key
        if str(self.upstream) not in sys.path:
            sys.path.insert(0, str(self.upstream))
        from researchclaw.adapters import AdapterBundle
        from researchclaw.config import RCConfig

        base_url = os.environ.get("AGENT_API_BASE", "https://api.openai.com/v1")
        data = {
            "project": {"name": episode_id, "mode": "full-auto"},
            "research": {"topic": objective, "quality_threshold": 0},
            "runtime": {"timezone": "UTC", "max_parallel_tasks": 1, "retry_limit": 0},
            "notifications": {"channel": "none", "on_stage_start": False, "on_stage_fail": False, "on_gate_required": False},
            "knowledge_base": {"backend": "markdown", "root": str(self.run_dir / "kb")},
            "openclaw_bridge": {},
            "llm": {"provider": "openai", "base_url": base_url, "api_key_env": "OPENAI_API_KEY", "api_key": self.api_key,
                    "primary_model": self.model, "fallback_models": [], "timeout_sec": min(600, max(1, int(self.initial_budget.wall_seconds)))},
            "security": {"hitl_required_stages": [], "allow_publish_without_approval": True},
            "experiment": {"mode": "sandbox", "time_budget_sec": max(1, min(3600, int(self.initial_budget.wall_seconds))),
                           "max_iterations": max(1, min(10, self.initial_budget.hops)),
                           "sandbox": {"python_path": sys.executable},
                           "cli_agent": {"provider": "llm", "max_budget_usd": self.initial_budget.provider_cost_usd}},
            "memory": {"enabled": True, "store_dir": str(self.run_dir / "memory")},
            "skills": {"enabled": True},
            "web_search": {"enabled": True},
        }
        self.config = RCConfig.from_dict(data, project_root=self.workspace, check_paths=False)
        self.adapters = AdapterBundle.from_config(self.config)
        self._install_usage_adapter()

    def initialize_native(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        self.initialize(episode_id=episode_id, workspace=workspace, objective=objective, seed=seed)

    def run_native(self) -> NativeRunResult:
        """Run the native full-auto pipeline as one scheduler-owned operation."""
        self._require_initialized()
        assert self.workspace is not None and self.run_dir is not None
        from researchclaw.pipeline.runner import execute_pipeline

        before, usage_before = snapshot_workspace(self.workspace), self.usage
        started = time.monotonic()
        results = execute_pipeline(
            run_dir=self.run_dir,
            run_id=self.episode_id,
            config=self.config,
            adapters=self.adapters,
            auto_approve_gates=True,
        )
        self._normalize_products()
        report = self.workspace / "report" / "report.md"
        stages_done = sum(1 for result in results if getattr(result.status, "value", result.status) == "done")
        complete = bool(results) and stages_done == len(results) and report.is_file()
        self.terminal = True
        return NativeRunResult(
            status="completed" if complete else "stop",
            reason="AutoResearchClaw native full-auto pipeline completed" if complete else "AutoResearchClaw native pipeline stopped before completion",
            native_iterations=len(results),
            artifacts_before=before,
            artifacts_after=snapshot_workspace(self.workspace),
            usage=Usage(
                provider_cost_usd=self.usage.provider_cost_usd - usage_before.provider_cost_usd,
                input_tokens=self.usage.input_tokens - usage_before.input_tokens,
                output_tokens=self.usage.output_tokens - usage_before.output_tokens,
                agent_calls=self.usage.agent_calls - usage_before.agent_calls,
                wall_seconds=time.monotonic() - started,
                cost_source=self.usage.cost_source,
                token_source=self.usage.token_source,
            ),
            native_status="completed" if complete else "stopped",
            metrics={"stages_done": float(stages_done), "stages_returned": float(len(results))},
        )

    def checkpoint(self) -> Checkpoint:
        self._require_initialized()
        next_id = self._next_native()
        remaining = Budget(
            max(0.0, self.initial_budget.provider_cost_usd - self.usage.provider_cost_usd),
            max(0, self.initial_budget.input_tokens - self.usage.input_tokens),
            max(0, self.initial_budget.output_tokens - self.usage.output_tokens),
            max(0, self.initial_budget.agent_calls - self.usage.agent_calls),
            max(0.0, self.initial_budget.wall_seconds - (time.monotonic() - self.started)),
            max(0, self.initial_budget.hops - self.hop),
        )
        issues = [] if self.terminal else self.open_issues or [Issue(f"native:{next_id}", "native_requirement",
            f"AutoResearchClaw phase {next_id} remains", required_tags=REQUIRED_TAGS[next_id])]
        return Checkpoint(self.episode_id, self.hop, self.objective,
            "terminal" if self.terminal else next_id, snapshot_workspace(self.workspace),
            issues, remaining, self._available_cards(), terminal=self.terminal)

    def native_next(self, checkpoint: Checkpoint) -> str | None:
        return None if self.terminal else self._next_native()

    def invoke(self, capability_id: str, contract: WorkContract | None) -> InvocationResult:
        self._require_initialized()
        if capability_id not in {c.capability_id for c in self._available_cards() if c.available}:
            raise ValueError(f"AutoResearchClaw capability prerequisites are not satisfied: {capability_id}")
        before, usage_before = snapshot_workspace(self.workspace), self.usage
        started, output, error = time.monotonic(), "", None
        proposed_done = False
        metrics: dict[str, float] = {}
        try:
            from researchclaw.pipeline.executor import execute_stage
            from researchclaw.pipeline.stages import Stage, StageStatus
            first, last = STAGE_RANGES[capability_id]
            results = []
            for stage_number in range(first, last + 1):
                result = execute_stage(Stage(stage_number), run_dir=self.run_dir,
                    run_id=self.episode_id, config=self.config, adapters=self.adapters,
                    auto_approve_gates=True)
                results.append(result)
                if result.status != StageStatus.DONE:
                    break
            output = "\n".join(f"{r.stage.name}: {r.status.value}{': ' + r.error if r.error else ''}" for r in results)
            if results and all(r.status.value == "done" for r in results):
                self.completed.add(capability_id)
            self._normalize_products()
            if capability_id in {"writing", "finalize"}:
                review = self._review_text()
                self.open_issues = extract_review_issues(review)
                score = parse_review_score(review)
                if score is None:
                    score = self._quality_score()
                if score is not None:
                    metrics["review_score"] = score
                if capability_id == "finalize" and results and all(r.status.value == "done" for r in results):
                    proposed_done = True
                    self.terminal = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.hop += 1
        after = snapshot_workspace(self.workspace)
        return InvocationResult(capability_id, output, before, after, Usage(
            provider_cost_usd=self.usage.provider_cost_usd - usage_before.provider_cost_usd,
            input_tokens=self.usage.input_tokens - usage_before.input_tokens,
            output_tokens=self.usage.output_tokens - usage_before.output_tokens,
            agent_calls=self.usage.agent_calls - usage_before.agent_calls,
            wall_seconds=time.monotonic() - started, cost_source=self.usage.cost_source, token_source="provider_response"),
            error=error, proposed_done=proposed_done, metrics=metrics)

    def _install_usage_adapter(self) -> None:
        from researchclaw.llm.client import LLMClient
        bridge, original = self, LLMClient.chat
        if getattr(original, "_rac_wrapped", False):
            return
        def chat(client, messages, **kwargs):
            if bridge.usage.agent_calls >= bridge.initial_budget.agent_calls:
                raise RuntimeError("lifecycle agent-call budget exhausted")
            if time.monotonic() - bridge.started >= bridge.initial_budget.wall_seconds:
                raise TimeoutError("lifecycle wall-time budget exhausted")
            kwargs["model"] = bridge.model
            kwargs["max_tokens"] = min(int(kwargs.get("max_tokens") or bridge.initial_budget.output_tokens),
                                        max(1, bridge.initial_budget.output_tokens - bridge.usage.output_tokens))
            response = original(client, messages, **kwargs)
            raw = response.raw if isinstance(response.raw, dict) else {}
            cost = raw.get("response_cost", raw.get("provider_cost"))
            bridge.usage = replace(bridge.usage,
                provider_cost_usd=bridge.usage.provider_cost_usd + float(cost or 0),
                input_tokens=bridge.usage.input_tokens + int(response.prompt_tokens or 0),
                output_tokens=bridge.usage.output_tokens + int(response.completion_tokens or 0),
                agent_calls=bridge.usage.agent_calls + 1,
                cost_source="provider_response" if cost is not None else "unavailable",
                token_source="provider_response")
            return response
        chat._rac_wrapped = True
        LLMClient.chat = chat

    def _normalize_products(self) -> None:
        assert self.workspace is not None and self.run_dir is not None
        mappings = [
            (self.run_dir / "stage-10" / "experiment", self.workspace / "code" / "auto_research_claw"),
            (self.run_dir / "stage-14" / "analysis.md", self.workspace / "outputs" / "auto_research_claw" / "analysis.md"),
            (self.run_dir / "stage-18" / "reviews.md", self.workspace / "state" / "auto_research_claw" / "reviews.md"),
        ]
        for source, destination in mappings:
            if source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                for item in source.rglob("*"):
                    if item.is_file():
                        target = destination / item.relative_to(source)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(item, target)
            elif source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        for source in (self.run_dir / "stage-23" / "paper_final_verified.md", self.run_dir / "stage-22" / "paper_final.md",
                       self.run_dir / "stage-19" / "paper_revised.md", self.run_dir / "stage-17" / "paper_draft.md"):
            if source.is_file():
                report = self.workspace / "report" / "report.md"
                report.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, report)
                break

    def _review_text(self) -> str:
        assert self.run_dir is not None
        path = self.run_dir / "stage-18" / "reviews.md"
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""

    def _quality_score(self) -> float | None:
        assert self.run_dir is not None
        path = self.run_dir / "stage-20" / "quality_report.json"
        if not path.is_file(): return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            value = data.get("score", data.get("overall_score"))
            return float(value) if value is not None else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError): return None

    def _available_cards(self):
        idx = ORDER.index(self._next_native()) if not self.terminal else len(ORDER)
        return [replace(c, available=ORDER.index(c.capability_id) <= min(idx + 1, len(ORDER) - 1)) for c in self.cards]

    def _next_native(self) -> str:
        return next((item for item in ORDER if item not in self.completed), "finalize")

    def _require_initialized(self) -> None:
        if self.workspace is None or self.config is None or self.adapters is None or self.run_dir is None:
            raise RuntimeError("AutoResearchClawBridge.initialize must be called first")
