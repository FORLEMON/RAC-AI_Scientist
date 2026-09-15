from __future__ import annotations

import asyncio
import json
import os
import subprocess
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
from .ark import render_contract_prompt


ORDER = ("idea", "survey", "implementation_plan", "implementation", "experiment_analysis", "paper_writing", "review")
REQUIRED_TAGS = {
    "idea": ("planning", "literature"), "survey": ("literature",),
    "implementation_plan": ("planning", "experiment"),
    "implementation": ("code", "experiment"), "experiment_analysis": ("analysis",),
    "paper_writing": ("writing",), "review": ("terminal_review",),
}


class _LocalCommandEnv:
    """AI-Researcher's DockerEnv protocol implemented inside its host container."""
    def __init__(self, root: Path, timeout: int):
        self.root = root
        self.local_workplace = str(root)
        self.docker_workplace = str(root)
        self.workplace_name = root.name
        self.timeout = timeout

    def run_command(self, command: str, stream_callback=None):
        try:
            result = subprocess.run(command, shell=True, cwd=self.root, capture_output=True,
                                    text=True, timeout=self.timeout, encoding="utf-8", errors="replace")
            output = (result.stdout or "") + (result.stderr or "")
            if stream_callback and output:
                stream_callback(output)
            return {"status": result.returncode, "result": output}
        except subprocess.TimeoutExpired as exc:
            return {"status": 124, "result": f"command timed out: {exc}"}


class AIResearcherBridge(HostBridge):
    """Capability bridge over AI-Researcher's native MetaChain agents."""

    host_id = "ai_researcher"

    def __init__(self, upstream: Path, manifest: Path, budget: Budget, model: str, api_key: str):
        self.upstream = upstream.resolve()
        self.manifest = load_host_manifest(manifest)
        self.cards = capability_cards(self.manifest)
        self.initial_budget = budget
        self.model, self.api_key = model, api_key
        self.workspace: Path | None = None
        self.client: Any = None
        self.agents: dict[str, Any] = {}
        self.context: dict[str, Any] = {}
        self.episode_id = self.objective = ""
        self.completed: set[str] = set()
        self.hop = 0
        self.started = 0.0
        self.terminal = False
        self.usage = Usage()
        self.open_issues: list[Issue] = []

    def initialize(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        if not (self.upstream / "research_agent" / "inno" / "core.py").is_file():
            raise FileNotFoundError(f"AI-Researcher checkout not found: {self.upstream}")
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.episode_id, self.objective = episode_id, objective
        self.started = time.monotonic()
        seed_runtime(seed)
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["COMPLETION_MODEL"] = self.model
        os.environ["CHEEP_MODEL"] = self.model
        os.environ["API_BASE_URL"] = os.environ.get("AGENT_API_BASE", "https://api.openai.com/v1")
        for source in (self.upstream, self.upstream / "research_agent"):
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))
        from research_agent.inno.core import MetaChain
        from research_agent.inno.types import Agent
        from research_agent.inno.agents.inno_agent.idea_agent import get_idea_agent
        from research_agent.inno.agents.inno_agent.survey_agent import get_survey_agent
        from research_agent.inno.agents.inno_agent.plan_agent import get_coding_plan_agent
        from research_agent.inno.agents.inno_agent.ml_agent import get_ml_agent
        from research_agent.inno.agents.inno_agent.exp_analyser import get_exp_analyser_agent

        native = self.workspace / "ai_researcher_native"
        native.mkdir(exist_ok=True)
        code_env = _LocalCommandEnv(self.workspace, max(1, min(1800, int(self.initial_budget.wall_seconds))))
        from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser
        file_env = RequestsMarkdownBrowser(local_root=str(self.workspace.parent), workplace_name=self.workspace.name)
        file_env.local_workplace = str(self.workspace)
        file_env.docker_workplace = str(self.workspace)
        self.client = MetaChain(log_path=str(native / "metachain.log"))
        self.agents = {
            "idea": get_idea_agent(self.model, file_env=file_env),
            "survey": get_survey_agent(self.model, file_env=file_env, code_env=code_env),
            "implementation_plan": get_coding_plan_agent(self.model, code_env=code_env),
            "implementation": get_ml_agent(self.model, code_env=code_env),
            "experiment_analysis": get_exp_analyser_agent(self.model, file_env=file_env, code_env=code_env),
            "paper_writing": Agent(name="Paper Generation Agent", model=self.model,
                instructions="Write a complete evidence-grounded scientific report from the current workspace. Do not invent results or citations."),
            "review": Agent(name="Paper Review Agent", model=self.model,
                instructions="Review the supplied report against persisted evidence. Return Score: X/10 and actionable issues; do not revise the report."),
        }
        self.context = {"working_dir": self.workspace.as_posix().lstrip("/"), "notes": [],
                        "innovative_idea": objective, "objective": objective}
        self._install_usage_adapter()

    def initialize_native(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        self.initialize(episode_id=episode_id, workspace=workspace, objective=objective, seed=seed)

    def run_native(self) -> NativeRunResult:
        """Run the fixed Level-1 research sequence without RAC routing decisions.

        AI-Researcher's published Level-1 launcher is tied to its own ML benchmark
        instance schema and nested Docker environment.  The compatibility layer
        therefore retains its MetaChain agents and fixed survey -> plan -> build ->
        analyse -> write order, while mapping the supplied ResearchClawBench
        workspace into that sequence.
        """
        self._require_initialized()
        assert self.workspace is not None
        before, usage_before = snapshot_workspace(self.workspace), self.usage
        started = time.monotonic()
        idea_path = self.workspace / "state" / "ai_researcher" / "idea.md"
        idea_path.parent.mkdir(parents=True, exist_ok=True)
        idea_path.write_text(self.objective, encoding="utf-8")
        self.completed.add("idea")
        iterations = 0
        error: str | None = None
        for capability_id in ("survey", "implementation_plan", "implementation", "experiment_analysis", "paper_writing"):
            result = self.invoke(capability_id, None)
            iterations += 1
            if result.error:
                error = result.error
                break
        report = self.workspace / "report" / "report.md"
        complete = error is None and report.is_file() and bool(report.read_text(encoding="utf-8", errors="replace").strip())
        self.terminal = True
        return NativeRunResult(
            status="completed" if complete else "stop",
            reason="AI-Researcher native Level-1 sequence completed" if complete else (error or "AI-Researcher returned without a report"),
            native_iterations=iterations,
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
        )

    def checkpoint(self) -> Checkpoint:
        self._require_initialized()
        next_id = self._next_native()
        remaining = Budget(max(0.0, self.initial_budget.provider_cost_usd - self.usage.provider_cost_usd),
            max(0, self.initial_budget.input_tokens - self.usage.input_tokens),
            max(0, self.initial_budget.output_tokens - self.usage.output_tokens),
            max(0, self.initial_budget.agent_calls - self.usage.agent_calls),
            max(0.0, self.initial_budget.wall_seconds - (time.monotonic() - self.started)),
            max(0, self.initial_budget.hops - self.hop))
        issues = [] if self.terminal else self.open_issues or [Issue(f"native:{next_id}", "native_requirement",
            f"AI-Researcher capability {next_id} remains", required_tags=REQUIRED_TAGS[next_id])]
        return Checkpoint(self.episode_id, self.hop, self.objective,
            "terminal" if self.terminal else next_id, snapshot_workspace(self.workspace), issues,
            remaining, self._available_cards(), terminal=self.terminal)

    def native_next(self, checkpoint: Checkpoint) -> str | None:
        return None if self.terminal else self._next_native()

    def invoke(self, capability_id: str, contract: WorkContract | None) -> InvocationResult:
        self._require_initialized()
        if capability_id not in {c.capability_id for c in self._available_cards() if c.available}:
            raise ValueError(f"AI-Researcher capability prerequisites are not satisfied: {capability_id}")
        before, usage_before = snapshot_workspace(self.workspace), self.usage
        started, output, error, proposed_done = time.monotonic(), "", None, False
        metrics: dict[str, float] = {}
        try:
            prompt = render_contract_prompt(self.objective, capability_id, contract)
            if capability_id == "paper_writing":
                prompt += "\nRead persisted plans, code, and outputs, then return the complete Markdown report."
            elif capability_id == "review":
                report = self.workspace / "report" / "report.md"
                prompt += "\nReport to review:\n" + (report.read_text(encoding="utf-8", errors="replace") if report.is_file() else "[missing]")
            response = asyncio.run(self.client.run_async(self.agents[capability_id],
                [{"role": "user", "content": prompt}], context_variables=self.context,
                model_override=self.model, debug=False, max_turns=max(2, self.initial_budget.agent_calls - self.usage.agent_calls)))
            self.context.update(response.context_variables)
            output = "\n".join(str(item.get("content", "")) for item in response.messages if isinstance(item, dict))
            self._persist(capability_id, output)
            self.completed.add(capability_id)
            if capability_id == "review":
                self.open_issues = extract_review_issues(output)
                score = parse_review_score(output)
                if score is not None:
                    metrics["review_score"] = score
                    proposed_done = score >= 8.0
                if proposed_done: self.terminal = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.hop += 1
        after = snapshot_workspace(self.workspace)
        return InvocationResult(capability_id, output, before, after, Usage(
            provider_cost_usd=self.usage.provider_cost_usd - usage_before.provider_cost_usd,
            input_tokens=self.usage.input_tokens - usage_before.input_tokens,
            output_tokens=self.usage.output_tokens - usage_before.output_tokens,
            agent_calls=self.usage.agent_calls - usage_before.agent_calls,
            wall_seconds=time.monotonic() - started,
            cost_source="provider_response" if self.usage.cost_source == "provider_response" else "unavailable",
            token_source="provider_response"), error=error, proposed_done=proposed_done, metrics=metrics)

    def _install_usage_adapter(self) -> None:
        import research_agent.inno.core as core
        bridge, original = self, core.acompletion
        async def completion(**kwargs):
            if bridge.usage.agent_calls >= bridge.initial_budget.agent_calls:
                raise RuntimeError("lifecycle agent-call budget exhausted")
            kwargs["model"] = bridge.model
            kwargs["base_url"] = os.environ["API_BASE_URL"]
            kwargs["api_key"] = bridge.api_key
            kwargs["max_tokens"] = max(1, bridge.initial_budget.output_tokens - bridge.usage.output_tokens)
            result = await original(**kwargs)
            raw_usage = getattr(result, "usage", None)
            inp = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
            out = int(getattr(raw_usage, "completion_tokens", 0) or 0)
            hidden = getattr(result, "_hidden_params", {}) or {}
            cost = hidden.get("response_cost")
            bridge.usage = Usage(bridge.usage.provider_cost_usd + float(cost or 0),
                bridge.usage.input_tokens + inp, bridge.usage.output_tokens + out,
                bridge.usage.agent_calls + 1, bridge.usage.wall_seconds,
                "provider_response" if cost is not None else "unavailable", "provider_response")
            return result
        core.acompletion = completion

    def _persist(self, capability_id: str, output: str) -> None:
        assert self.workspace is not None
        path = self.workspace / "state" / "ai_researcher" / f"{capability_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
        if capability_id == "paper_writing":
            report = self.workspace / "report" / "report.md"
            report.parent.mkdir(exist_ok=True)
            report.write_text(output, encoding="utf-8")

    def _available_cards(self):
        idx = ORDER.index(self._next_native()) if not self.terminal else len(ORDER)
        return [replace(c, available=ORDER.index(c.capability_id) <= min(idx + 1, len(ORDER) - 1)) for c in self.cards]

    def _next_native(self) -> str:
        return next((item for item in ORDER if item not in self.completed), "review")

    def _require_initialized(self) -> None:
        if self.workspace is None or self.client is None:
            raise RuntimeError("AIResearcherBridge.initialize must be called first")
