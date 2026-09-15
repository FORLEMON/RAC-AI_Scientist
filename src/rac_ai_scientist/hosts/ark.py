from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from ..artifacts import snapshot_workspace
from ..bridge import HostBridge
from ..issues import extract_review_issues, parse_review_score
from ..manifest import capability_cards, load_host_manifest
from ..reproducibility import seed_runtime
from ..schemas import Budget, Checkpoint, InvocationResult, Issue, NativeRunResult, Usage, WorkContract


ORDER = ("researcher", "experimenter", "writer", "reviewer", "planner")
NATIVE_DEV_ITERATIONS = 3
NATIVE_REVIEW_ITERATIONS = 3
REQUIRED_TAGS = {
    "researcher": ("planning",),
    "experimenter": ("experiment",),
    "writer": ("writing",),
    "reviewer": ("terminal_review",),
    "planner": ("planning",),
}


class ArkBridge(HostBridge):
    """ARK adapter with a host-owned N0 path and capability-level RAC path."""

    host_id = "ark"

    def __init__(self, upstream: Path, manifest: Path, budget: Budget, model: str, api_key: str):
        self.upstream = upstream.resolve()
        self.manifest = load_host_manifest(manifest)
        self.cards = capability_cards(self.manifest)
        self.initial_budget = budget
        self.model = model
        self.api_key = api_key
        self.workspace: Path | None = None
        self.orchestrator: Any = None
        self.episode_id = ""
        self.objective = ""
        self.hop = 0
        self.native_capability = ORDER[0]
        self.started = 0.0
        self.open_issues: list[Issue] = []
        self.native_mode = False

    def initialize(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        self._initialize(episode_id=episode_id, workspace=workspace, objective=objective, seed=seed, native=False)

    def initialize_native(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        self._initialize(episode_id=episode_id, workspace=workspace, objective=objective, seed=seed, native=True)

    def _initialize(self, *, episode_id: str, workspace: Path, objective: str, seed: int, native: bool) -> None:
        if not (self.upstream / "ark" / "orchestrator").is_dir():
            raise FileNotFoundError(f"ARK checkout not found: {self.upstream}")
        self.workspace = workspace.resolve()
        self.episode_id = episode_id
        self.objective = objective
        self.native_mode = native
        seed_runtime(seed)
        self.workspace.mkdir(parents=True, exist_ok=True)
        project = self.workspace / (Path(".ark") / "native_project" if native else Path(".rac") / "ark_project")
        agents = project / "agents"
        agents.mkdir(parents=True, exist_ok=True)
        templates = self.upstream / "ark" / "templates" / "agents"
        for source in templates.glob("*.prompt"):
            content = source.read_text(encoding="utf-8")
            replacements = {
                "{PROJECT_NAME}": episode_id,
                "{PAPER_TITLE}": episode_id,
                "{VENUE_NAME}": "ResearchClawBench",
                "{VENUE_FORMAT}": "benchmark report",
                "{VENUE_PAGES}": "unlimited",
                "{LATEX_DIR}": "report",
                "{FIGURES_DIR}": "report/images",
            }
            for old, new in replacements.items():
                content = content.replace(old, new)
            (agents / source.name).write_text(content, encoding="utf-8")
        hook_comment = (
            "# ResearchClawBench native ARK run: no project-specific hooks.\n"
            if native
            else "# RAC benchmark bridge: no project-specific hooks.\n"
        )
        (project / "hooks.py").write_text(hook_comment, encoding="utf-8")
        (project / "config.yaml").write_text(self._config_text(native=native), encoding="utf-8")
        report = self.workspace / "report"
        report.mkdir(exist_ok=True)
        (report / "images").mkdir(exist_ok=True)
        if not native:
            main = report / "main.tex"
            if not main.exists():
                main.write_text(
                    "\\documentclass{article}\n\\usepackage{graphicx}\n\\begin{document}\n"
                    "\\section*{Research report}\nWork in progress.\n\\end{document}\n",
                    encoding="utf-8",
                )
            (report / "references.bib").touch(exist_ok=True)
            state = self.workspace / "auto_research" / "state"
            state.mkdir(parents=True, exist_ok=True)
            (state / "idea.md").write_text(objective, encoding="utf-8")
            (state / "project_context.md").write_text(self._context_text(), encoding="utf-8")
        os.environ.setdefault("OPENAI_API_KEY", self.api_key)
        if os.environ.get("AGENT_API_BASE"):
            os.environ.setdefault("OPENAI_API_BASE", os.environ["AGENT_API_BASE"])
        if str(self.upstream) not in sys.path:
            sys.path.insert(0, str(self.upstream))
        from ark.orchestrator import Orchestrator

        self.orchestrator = Orchestrator(
            project=episode_id,
            max_days=max(self.initial_budget.wall_seconds / 86400.0, 0.001),
            max_iterations=NATIVE_REVIEW_ITERATIONS if native else max(1, self.initial_budget.hops),
            model=self.model,
            code_dir=str(self.workspace),
            project_dir=str(project),
            mode="paper",
        )
        self.started = time.monotonic()
        self.open_issues = [] if native else [Issue("native:researcher", "native_requirement", "initial research framing is incomplete", required_tags=("planning",))]

    def run_native(self) -> NativeRunResult:
        """Run ARK's complete scheduler once; RAC makes no phase decisions."""
        self._require_initialized()
        if not self.native_mode:
            raise RuntimeError("initialize_native must be used before run_native")
        assert self.workspace is not None
        before = snapshot_workspace(self.workspace)
        usage_before = self._usage_totals()
        started = time.monotonic()

        self.orchestrator.run()
        self._normalize_report()

        after = snapshot_workspace(self.workspace)
        usage_after = self._usage_totals()
        paper_state = self.orchestrator.load_paper_state()
        native_status = str(paper_state.get("status", "unknown"))
        iterations = int(getattr(self.orchestrator, "iteration", 0) or 0)
        review_score = float(paper_state.get("current_score", 0) or 0)
        stopped = bool(getattr(self.orchestrator, "_stop_requested", False))
        if native_status == "accepted":
            status = "completed"
            reason = "ARK native acceptance threshold reached"
        elif stopped:
            status = "stop"
            reason = "ARK native stop was requested"
        elif iterations >= NATIVE_REVIEW_ITERATIONS:
            status = "stop"
            reason = "ARK native review-iteration limit reached"
        else:
            status = "stop"
            reason = f"ARK native workflow returned with paper status {native_status}"
        return NativeRunResult(
            status=status,
            reason=reason,
            native_iterations=iterations,
            artifacts_before=before,
            artifacts_after=after,
            usage=Usage(
                provider_cost_usd=usage_after.provider_cost_usd - usage_before.provider_cost_usd,
                input_tokens=usage_after.input_tokens - usage_before.input_tokens,
                output_tokens=usage_after.output_tokens - usage_before.output_tokens,
                agent_calls=usage_after.agent_calls - usage_before.agent_calls,
                wall_seconds=time.monotonic() - started,
                cost_source=usage_after.cost_source,
                token_source=usage_after.token_source,
            ),
            native_status=native_status,
            metrics={"review_score": review_score},
        )

    def checkpoint(self) -> Checkpoint:
        self._require_initialized()
        elapsed = time.monotonic() - self.started
        stats = self._usage_totals()
        remaining = Budget(
            max(0.0, self.initial_budget.provider_cost_usd - stats.provider_cost_usd),
            max(0, self.initial_budget.input_tokens - stats.input_tokens),
            max(0, self.initial_budget.output_tokens - stats.output_tokens),
            max(0, self.initial_budget.agent_calls - stats.agent_calls),
            max(0.0, self.initial_budget.wall_seconds - elapsed),
            max(0, self.initial_budget.hops - self.hop),
        )
        return Checkpoint(
            self.episode_id,
            self.hop,
            self.objective,
            self.native_capability,
            snapshot_workspace(self.workspace),
            list(self.open_issues),
            remaining,
            self.cards,
            terminal=False,
        )

    def native_next(self, checkpoint: Checkpoint) -> str | None:
        return self.native_capability

    def invoke(self, capability_id: str, contract: WorkContract | None) -> InvocationResult:
        self._require_initialized()
        if capability_id not in {card.capability_id for card in self.cards}:
            raise ValueError(f"unknown ARK capability: {capability_id}")
        before = snapshot_workspace(self.workspace)
        usage_before = self._usage_totals()
        started = time.monotonic()
        error = None
        output = ""
        metrics: dict[str, float] = {}
        proposed_done = False
        try:
            prompt = render_contract_prompt(self.objective, capability_id, contract)
            if capability_id == "reviewer":
                try:
                    self.orchestrator.compile_latex()
                except Exception:
                    pass
            output = self.orchestrator.run_agent(capability_id, prompt, timeout=self._timeout())
            self._persist_output(capability_id, output)
            if capability_id == "reviewer":
                score = parse_review_score(output)
                if score is not None:
                    metrics["review_score"] = score
                    proposed_done = score >= 8.0
                self.open_issues = extract_review_issues(output)
                if score is None:
                    self.open_issues.append(Issue("review:score_missing", "artifact", "review score is missing", required_tags=("terminal_review",)))
            else:
                self.open_issues = [
                    Issue(
                        f"native:{self._successor(capability_id)}",
                        "native_requirement",
                        f"{self._successor(capability_id)} work remains",
                        required_tags=REQUIRED_TAGS[self._successor(capability_id)],
                    )
                ]
            if capability_id in {"writer", "reviewer"}:
                self._normalize_report()
            self.native_capability = self._successor(capability_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.hop += 1
        after = snapshot_workspace(self.workspace)
        usage_after = self._usage_totals()
        return InvocationResult(
            capability_id,
            output,
            before,
            after,
            Usage(
                provider_cost_usd=usage_after.provider_cost_usd - usage_before.provider_cost_usd,
                input_tokens=usage_after.input_tokens - usage_before.input_tokens,
                output_tokens=usage_after.output_tokens - usage_before.output_tokens,
                agent_calls=usage_after.agent_calls - usage_before.agent_calls,
                wall_seconds=time.monotonic() - started,
                cost_source="host_estimate",
                token_source="provider_response",
            ),
            error=error,
            proposed_done=proposed_done,
            metrics=metrics,
        )

    def _successor(self, capability_id: str) -> str:
        if capability_id == "reviewer":
            return "planner"
        if capability_id == "planner":
            return "experimenter"
        index = ORDER.index(capability_id)
        return ORDER[min(index + 1, len(ORDER) - 1)]

    def _persist_output(self, capability_id: str, output: str) -> None:
        assert self.workspace is not None
        relative = {
            "researcher": Path("state/ark/research_plan.md"),
            "experimenter": Path("outputs/ark/experimenter_result.md"),
            "coder": Path("code/ark/coder_result.md"),
            "writer": Path("state/ark/writer_result.md"),
            "reviewer": Path("state/ark/reviewer_review.md"),
            "planner": Path("state/ark/action_plan.md"),
        }[capability_id]
        destination = self.workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output or "", encoding="utf-8")

    def _normalize_report(self) -> None:
        assert self.workspace is not None
        source = self.workspace / "report" / "main.tex"
        if source.is_file() and source.stat().st_size > 0:
            shutil.copyfile(source, self.workspace / "report" / "report.md")

    def _usage_totals(self) -> Usage:
        stats = getattr(self.orchestrator, "_agent_stats", []) if self.orchestrator else []
        return Usage(
            provider_cost_usd=sum(float(item.get("cost_usd", 0) or 0) for item in stats),
            input_tokens=sum(int(item.get("input_tokens", 0) or 0) for item in stats),
            output_tokens=sum(int(item.get("output_tokens", 0) or 0) for item in stats),
            agent_calls=len(stats),
            cost_source="host_estimate",
            token_source="provider_response",
        )

    def _timeout(self) -> int:
        elapsed = time.monotonic() - self.started
        return max(1, int(min(1800, self.initial_budget.wall_seconds - elapsed)))

    def _config_text(self, *, native: bool = False) -> str:
        provider = self.model.split("/", 1)[0].upper() if "/" in self.model else "OPENAI"
        key_name = provider.lower() + "_api_key"
        config = (
            f"model: {self.model}\n"
            f"bot_model: {self.model}\n"
            f"{key_name}: \"\"\n"
            "latex_dir: report\n"
            "figures_dir: report/images\n"
            "paper_accept_threshold: 8\n"
            "log_verbosity: normal\n"
            "intervention:\n  enabled: false\n"
            "artifact_store:\n  type: local\n"
        )
        if native:
            config += (
                f"research_idea: {json.dumps(self.objective, ensure_ascii=False)}\n"
                f"max_dev_iterations: {NATIVE_DEV_ITERATIONS}\n"
                f"max_iterations: {NATIVE_REVIEW_ITERATIONS}\n"
            )
        return config

    def _context_text(self) -> str:
        assert self.workspace is not None
        return (
            f"# ResearchClawBench episode\n\n{self.objective}\n\n"
            "Use only task.json, data/, and related_work/. The target study and scoring checklist are hidden. "
            "Persist executable work in code/, measurements in outputs/, and the final report in report/report.md.\n"
        )

    def _require_initialized(self) -> None:
        if self.workspace is None or self.orchestrator is None:
            raise RuntimeError("ArkBridge.initialize must be called first")


def render_contract_prompt(objective: str, capability_id: str, contract: WorkContract | None) -> str:
    if contract is None:
        return f"Objective: {objective}\nPerform the {capability_id} step using only the current workspace. Persist all work before returning."
    readable = "\n".join(f"- {item}" for item in contract.readable_artifacts) or "- none"
    writable = "\n".join(f"- {item}" for item in contract.writable_artifacts) or "- none"
    evidence = "\n".join(f"- {item.kind}: {', '.join(item.artifact_kinds) or 'general'}" for item in contract.required_evidence)
    return (
        f"Objective: {contract.objective}\nCapability: {capability_id}\n"
        f"Readable artifacts:\n{readable}\nWritable artifacts:\n{writable}\n"
        f"Required persisted evidence:\n{evidence}\n"
        "Do not claim completion unless the required evidence exists on disk."
    )
