from __future__ import annotations

import os
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
from ..schemas import Budget, Checkpoint, InvocationResult, Issue, Usage, WorkContract
from .ark import render_contract_prompt


ORDER = ("planner", "research", "code", "debug", "data_analysis", "writing", "review")
REQUIRED_TAGS = {
    "planner": ("planning",),
    "research": ("literature",),
    "code": ("code", "experiment"),
    "debug": ("debug",),
    "data_analysis": ("analysis", "figure"),
    "writing": ("writing",),
    "review": ("terminal_review",),
}
SUBAGENTS = {
    "planner": "planner-agent",
    "research": "research-agent",
    "code": "code-agent",
    "debug": "debug-agent",
    "data_analysis": "data-analysis-agent",
    "writing": "writing-agent",
}


class EvoScientistBridge(HostBridge):
    """Programmatic bridge over EvoScientist's check-pointed Deep Agent."""

    host_id = "evo_scientist"

    def __init__(self, upstream: Path, manifest: Path, budget: Budget, model: str, api_key: str):
        self.upstream = upstream.resolve()
        self.manifest = load_host_manifest(manifest)
        self.cards = capability_cards(self.manifest)
        self.initial_budget = budget
        self.model = model
        self.api_key = api_key
        self.workspace: Path | None = None
        self.agent: Any = None
        self.episode_id = ""
        self.objective = ""
        self.completed: set[str] = set()
        self.hop = 0
        self.started = 0.0
        self.terminal = False
        self.usage = Usage()
        self._seen_messages: set[str] = set()
        self.open_issues: list[Issue] = []

    def initialize(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        if not (self.upstream / "EvoScientist" / "EvoScientist.py").is_file():
            raise FileNotFoundError(f"EvoScientist checkout not found: {self.upstream}")
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        seed_runtime(seed)
        self.episode_id, self.objective = episode_id, objective
        self.started = time.monotonic()
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["EVOSCIENTIST_DISABLE_UPDATE_CHECK"] = "1"
        if str(self.upstream) not in sys.path:
            sys.path.insert(0, str(self.upstream))
        from EvoScientist.config import EvoScientistConfig
        from EvoScientist.llm import get_chat_model
        from EvoScientist.EvoScientist import create_cli_agent

        base_url = os.environ.get("AGENT_API_BASE", "https://api.openai.com/v1")
        cfg = EvoScientistConfig(
            provider="custom-openai",
            model=self.model,
            custom_openai_api_key=self.api_key,
            custom_openai_base_url=base_url,
            default_mode="run",
            default_workdir=str(self.workspace),
            auto_approve=True,
            dangerous_mode=False,
            enable_async_subagents=False,
            enable_scheduler=False,
            memory_profile_enabled=False,
            memory_observations_enabled=False,
            memory_workers_enabled=False,
            memory_skill_synthesis_enabled=False,
            recursion_limit=max(20, min(5000, self.initial_budget.agent_calls * 20)),
        )
        os.environ["CUSTOM_OPENAI_API_KEY"] = self.api_key
        os.environ["CUSTOM_OPENAI_BASE_URL"] = base_url
        chat_model = get_chat_model(model=self.model, provider="custom-openai")
        self.agent = create_cli_agent(
            workspace_dir=str(self.workspace), config=cfg, chat_model=chat_model
        )
        native = self.workspace / "evo_scientist_native"
        native.mkdir(exist_ok=True)
        (native / "objective.md").write_text(objective, encoding="utf-8")

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
        issues = [] if self.terminal else self.open_issues or [Issue(
            f"native:{next_id}", "native_requirement",
            f"EvoScientist capability {next_id} remains", required_tags=REQUIRED_TAGS[next_id]
        )]
        return Checkpoint(self.episode_id, self.hop, self.objective,
                          "terminal" if self.terminal else next_id,
                          snapshot_workspace(self.workspace), issues, remaining,
                          self._available_cards(), terminal=self.terminal)

    def native_next(self, checkpoint: Checkpoint) -> str | None:
        return None if self.terminal else self._next_native()

    def invoke(self, capability_id: str, contract: WorkContract | None) -> InvocationResult:
        self._require_initialized()
        available = {c.capability_id for c in self._available_cards() if c.available}
        if capability_id not in available:
            raise ValueError(f"EvoScientist capability prerequisites are not satisfied: {capability_id}")
        before = snapshot_workspace(self.workspace)
        usage_before = self.usage
        started = time.monotonic()
        output, error, proposed_done = "", None, False
        metrics: dict[str, float] = {}
        try:
            prompt = self._prompt(capability_id, contract)
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": self.episode_id}},
            )
            messages = result.get("messages", []) if isinstance(result, dict) else []
            output = _message_text(messages[-1]) if messages else str(result)
            self._collect_usage(messages)
            self._persist(capability_id, output)
            self.completed.add(capability_id)
            if capability_id == "review":
                self.open_issues = extract_review_issues(output)
                score = parse_review_score(output)
                if score is not None:
                    metrics["review_score"] = score
                    proposed_done = score >= 8.0
                if proposed_done:
                    self.terminal = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.hop += 1
        after = snapshot_workspace(self.workspace)
        return InvocationResult(capability_id, output, before, after,
            Usage(
                provider_cost_usd=self.usage.provider_cost_usd - usage_before.provider_cost_usd,
                input_tokens=self.usage.input_tokens - usage_before.input_tokens,
                output_tokens=self.usage.output_tokens - usage_before.output_tokens,
                agent_calls=self.usage.agent_calls - usage_before.agent_calls,
                wall_seconds=time.monotonic() - started,
                cost_source="unavailable", token_source="provider_response",
            ), error=error, proposed_done=proposed_done, metrics=metrics)

    def _prompt(self, capability_id: str, contract: WorkContract | None) -> str:
        base = render_contract_prompt(self.objective, capability_id, contract)
        if capability_id == "review":
            role = "Independently review report/report.md and persisted code/results. Return a numeric Score: X/10 and actionable findings. Do not revise it."
        else:
            role = f"Use EvoScientist's native {SUBAGENTS[capability_id]} for this task."
        return base + "\n" + role + "\nNever access target_study. Work only in the current workspace."

    def _collect_usage(self, messages: list[Any]) -> None:
        inp = out = calls = 0
        for message in messages:
            key = str(getattr(message, "id", "") or id(message))
            if key in self._seen_messages:
                continue
            self._seen_messages.add(key)
            usage = getattr(message, "usage_metadata", None) or {}
            if usage:
                inp += int(usage.get("input_tokens", 0) or 0)
                out += int(usage.get("output_tokens", 0) or 0)
                calls += 1
        self.usage = replace(self.usage, input_tokens=self.usage.input_tokens + inp,
                             output_tokens=self.usage.output_tokens + out,
                             agent_calls=self.usage.agent_calls + calls)

    def _persist(self, capability_id: str, output: str) -> None:
        assert self.workspace is not None
        path = self.workspace / "state" / "evo_scientist" / f"{capability_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
        if capability_id == "writing":
            report = self.workspace / "report" / "report.md"
            report.parent.mkdir(exist_ok=True)
            if not report.exists():
                report.write_text(output, encoding="utf-8")

    def _available_cards(self):
        idx = ORDER.index(self._next_native()) if not self.terminal else len(ORDER)
        return [replace(c, available=ORDER.index(c.capability_id) <= min(idx + 1, len(ORDER) - 1)) for c in self.cards]

    def _next_native(self) -> str:
        return next((item for item in ORDER if item not in self.completed), "review")

    def _require_initialized(self) -> None:
        if self.workspace is None or self.agent is None:
            raise RuntimeError("EvoScientistBridge.initialize must be called first")


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return str(content)
