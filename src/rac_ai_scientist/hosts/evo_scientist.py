from __future__ import annotations

import asyncio
import os
import re
import sys
import threading
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

NATIVE_DELIVERY_RUBRIC = """Acceptance criteria for this unattended research run:
- report/report.md exists, is nonempty, and is a self-contained final scientific report.
- At least one nonempty analysis artifact exists under code/ or outputs/, and the report cites or describes the persisted evidence that supports its conclusions.
- The report clearly distinguishes measured or derived results from plans and limitations; incomplete experiments are disclosed rather than presented as completed evidence.
If any criterion is not met, continue the same research task using the existing workspace, address the grader feedback, and leave the required artifacts on disk before finishing."""

_NATIVE_RUBRIC_BUILD_LOCK = threading.Lock()


def _is_content_filter_error(exc: Exception) -> bool:
    """Return whether a provider rejected generated text via content policy."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    body = getattr(exc, "body", None)
    text = f"{exc} {body}".lower()
    marker = any(
        item in text
        for item in ("content_filter", "content filter", "finish_reason': 'content_filter", "jailbreak")
    )
    return marker and status_code in {None, 400}


def _provider_error_usage(exc: Exception) -> tuple[int, int]:
    """Read usage from a provider exception body or its serialized response."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        usage = body.get("usage")
        if isinstance(usage, dict):
            return (
                int(usage.get("prompt_tokens", 0) or 0),
                int(usage.get("completion_tokens", 0) or 0),
            )
    text = f"{exc}"
    prompt = re.search(r"['\"]prompt_tokens['\"]\s*:\s*(\d+)", text)
    completion = re.search(r"['\"]completion_tokens['\"]\s*:\s*(\d+)", text)
    return (
        int(prompt.group(1)) if prompt else 0,
        int(completion.group(1)) if completion else 0,
    )


def _content_filter_retry_prompt(
    objective: str,
    capability_id: str,
    contract: WorkContract | None,
) -> str:
    """Build a compact retry that keeps the task/contract but drops Room history."""
    prompt = render_contract_prompt(objective, capability_id, contract)
    return (
        "Continue the current academic research workflow from the files already present "
        "in the workspace. Complete only the assigned scientific step below. Treat prior "
        "coordination messages as background and do not restate them. Persist concrete "
        "evidence before returning.\n\n"
        f"{prompt}\n"
        "Never access target_study. Work only in the current workspace."
    )


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
        # Azure lists these exact deployments as text-input models.
        # https://learn.microsoft.com/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure
        azure_text_only = self.model.rsplit("/", 1)[-1].lower() in {
            "deepseek-v4-pro", "deepseek-v4-flash-0731",
        }
        model_options = {"profile": {"image_inputs": False, "pdf_inputs": False}} if azure_text_only else {}
        chat_model = get_chat_model(model=self.model, provider="custom-openai", **model_options)
        self.agent = _create_cli_agent_with_native_rubric(
            workspace_dir=str(self.workspace), config=cfg, chat_model=chat_model
        )
        native = self.workspace / "evo_scientist_native"
        native.mkdir(exist_ok=True)
        (native / "objective.md").write_text(objective, encoding="utf-8")

    def initialize_native(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        self.initialize(episode_id=episode_id, workspace=workspace, objective=objective, seed=seed)

    def run_native(self) -> NativeRunResult:
        """Run one native EvoScientist task with its own rubric-driven revision loop."""
        self._require_initialized()
        assert self.workspace is not None
        before, usage_before = snapshot_workspace(self.workspace), self.usage
        started = time.monotonic()
        prompt = (
            f"Complete this research task end to end using your native scientific workflow and sub-agents:\n{self.objective}\n\n"
            "Use only files already supplied in data/ and related_work/. Never seek or infer a hidden target study. "
            "Run real analyses where feasible, preserve code in code/ and results in outputs/, and write the final "
            "self-contained Markdown paper to report/report.md. Continue until the report is evidence-grounded and complete."
        )
        transcript_dir = self.workspace / "state" / "evo_scientist"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        retry_prompt = (
            "Complete the supplied academic research task using only files already present "
            "in the workspace. Persist executable analysis under code/ or outputs/ and write "
            "the evidence-grounded final report to report/report.md. Never access or infer a "
            f"hidden target study.\n\nObjective:\n{self.objective}"
        )
        result = self._invoke_agent(
            prompt,
            rubric=NATIVE_DELIVERY_RUBRIC,
            retry_prompt=retry_prompt,
        )
        messages = result.get("messages", []) if isinstance(result, dict) else []
        self._collect_usage(messages)
        output = _message_text(messages[-1]) if messages else str(result)
        (transcript_dir / "native_run.md").write_text(output, encoding="utf-8")
        complete = self._native_report_complete()
        self.terminal = True
        if complete:
            reason = "EvoScientist native rubric-guided deep-agent run completed"
            native_status = "completed"
        else:
            reason = "EvoScientist native rubric loop returned without report/report.md"
            native_status = "missing_report"
        return NativeRunResult(
            status="completed" if complete else "stop",
            reason=reason,
            # One RAC/native invocation. EvoScientist's RubricMiddleware owns
            # its internal grade-and-revise iterations (max_iterations=2).
            native_iterations=1,
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
            native_status=native_status,
        )

    def _native_report_complete(self) -> bool:
        assert self.workspace is not None
        report = self.workspace / "report" / "report.md"
        return report.is_file() and bool(
            report.read_text(encoding="utf-8", errors="replace").strip()
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
            native_prompt = self._prompt(capability_id, contract)
            prompt = self.communication_prompt(capability_id, native_prompt, contract)
            retry_prompt = _content_filter_retry_prompt(
                self.objective, capability_id, contract
            )
            result = self._invoke_agent(prompt, retry_prompt=retry_prompt)
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

    def _invoke_agent(
        self,
        prompt: str,
        *,
        rubric: str | None = None,
        retry_prompt: str | None = None,
    ):
        from EvoScientist.middleware.code_interpreter import aclose_code_interpreters

        try:
            for attempt in range(2):
                active_prompt = prompt if attempt == 0 else retry_prompt
                if active_prompt is None:
                    break
                payload: dict[str, Any] = {
                    "messages": [{"role": "user", "content": active_prompt}],
                }
                if rubric:
                    payload["rubric"] = rubric
                thread_id = self.episode_id
                if attempt:
                    thread_id = f"{self.episode_id}-provider-retry-h{getattr(self, 'hop', 0)}"
                try:
                    return self.agent.invoke(
                        payload,
                        config={"configurable": {"thread_id": thread_id}},
                    )
                except Exception as exc:
                    if attempt != 0 or retry_prompt is None or not _is_content_filter_error(exc):
                        raise
                    prompt_tokens, completion_tokens = _provider_error_usage(exc)
                    usage = getattr(self, "usage", Usage())
                    self.usage = replace(
                        usage,
                        input_tokens=usage.input_tokens + prompt_tokens,
                        output_tokens=usage.output_tokens + completion_tokens,
                        agent_calls=usage.agent_calls + 1,
                    )
                    print(
                        "[RAC] provider-filter retry with compact academic task context",
                        file=sys.stderr,
                    )
            raise RuntimeError("provider-filter retry was not configured")
        finally:
            # Provider errors skip native after-agent hooks. Close QuickJS workers
            # while Python can still service their event loops, not during GC.
            asyncio.run(aclose_code_interpreters())

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


def _create_cli_agent_with_native_rubric(*, workspace_dir: str, config: Any, chat_model: Any):
    """Build the normal CLI agent with EvoScientist's native rubric middleware last.

    The pinned upstream factory does not expose an ``extra_middleware`` hook. Its
    middleware builder is therefore wrapped only while the graph is constructed.
    The grader gets an equivalent read-only view of the same workspace, and the
    original builder is restored before this function returns.
    """
    import EvoScientist.EvoScientist as evo_api
    from EvoScientist.backends import CustomSandboxBackend
    from EvoScientist.subagents._factory import _scheduler_rubric_middleware

    grader_backend = CustomSandboxBackend(
        root_dir=workspace_dir,
        virtual_mode=True,
        timeout=config.sandbox_execute_timeout,
        dangerous=config.dangerous_mode,
        guard_dangerous=config.auto_approve,
    )
    original_builder = evo_api._get_default_middleware

    def build_with_rubric(*args, **kwargs):
        middleware = list(original_builder(*args, **kwargs))
        middleware.append(
            _scheduler_rubric_middleware(model=chat_model, backend=grader_backend)
        )
        return middleware

    # Graph construction is synchronous. Serialize the temporary module-level
    # override so concurrent bridge initialization cannot observe another build.
    with _NATIVE_RUBRIC_BUILD_LOCK:
        evo_api._get_default_middleware = build_with_rubric
        try:
            return evo_api.create_cli_agent(
                workspace_dir=workspace_dir,
                config=config,
                chat_model=chat_model,
            )
        finally:
            evo_api._get_default_middleware = original_builder
