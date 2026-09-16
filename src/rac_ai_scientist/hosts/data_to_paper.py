from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any

from ..artifacts import snapshot_workspace
from ..bridge import HostBridge
from ..manifest import capability_cards, load_host_manifest
from ..reproducibility import seed_runtime
from ..schemas import Budget, Checkpoint, InvocationResult, Issue, NativeRunResult, Usage, WorkContract


ORDER = (
    "data_exploration",
    "research_goal",
    "literature_review_goal",
    "hypothesis_plan",
    "data_analysis",
    "tables_figures",
    "paper_writing",
    "compile",
)
REQUIRED_TAGS = {
    "data_exploration": ("analysis",),
    "research_goal": ("planning",),
    "literature_review_goal": ("literature",),
    "hypothesis_plan": ("planning", "experiment"),
    "data_analysis": ("experiment", "analysis"),
    "tables_figures": ("figure",),
    "paper_writing": ("writing",),
    "compile": ("finalize",),
}


def _write_data_descriptions(workspace: Path, objective: str, data_filenames: list[str]) -> None:
    info = json.loads((workspace / "task.json").read_text(encoding="utf-8"))
    declared = {
        Path(item["path"]): str(item["description"])
        for item in info.get("data", [])
    }
    basenames = [Path(relative).name for relative in data_filenames]
    if len(basenames) != len(set(basenames)):
        raise ValueError("data-to-paper requires unique data basenames for description files")
    (workspace / "general_description.txt").write_text(objective, encoding="utf-8")
    for relative in data_filenames:
        path = Path(relative)
        matches = [source for source in declared if source == path or source in path.parents]
        if not matches:
            raise ValueError(f"missing benchmark description for data file: {relative}")
        source = max(matches, key=lambda item: len(item.parts))
        (workspace / (path.name + ".description.txt")).write_text(
            declared[source], encoding="utf-8"
        )


class DataToPaperBridge(HostBridge):
    """Thin bridge over data-to-paper's existing step runner and product store."""

    host_id = "data_to_paper"

    def __init__(self, upstream: Path, manifest: Path, budget: Budget, model: str, api_key: str):
        self.upstream = upstream.resolve()
        self.manifest = load_host_manifest(manifest)
        self.cards = capability_cards(self.manifest)
        self.initial_budget = budget
        self.model = model
        self.api_key = api_key
        self.workspace: Path | None = None
        self.runner: Any = None
        self.stage: Any = None
        self.episode_id = ""
        self.objective = ""
        self.completed: set[str] = set()
        self.hop = 0
        self.started = 0.0
        self.terminal = False
        self.provider_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.provider_cost_usd = 0.0
        self.cost_is_provider_reported = True

    def initialize(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        source = self.upstream / "src"
        if not (source / "data_to_paper").is_dir():
            raise FileNotFoundError(f"data-to-paper checkout not found: {self.upstream}")
        self.workspace = workspace.resolve()
        seed_runtime(seed)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        os.environ["OPENAI_API_KEY"] = self.api_key
        if os.environ.get("AGENT_API_BASE"):
            os.environ["OPENAI_API_BASE"] = os.environ["AGENT_API_BASE"]
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))

        from data_to_paper.research_types.hypothesis_testing.scientific_stage import ScientificStage
        from data_to_paper.research_types.hypothesis_testing.steps_runner import HypothesisTestingStepsRunner

        self._install_model_adapter()
        self.stage = ScientificStage
        config = {
            "research_goal": objective,
            "data_filenames": [
                path.relative_to(self.workspace).as_posix()
                for path in sorted((self.workspace / "data").rglob("*"))
                if path.is_file()
            ],
            "data_files_is_binary": [
                _is_binary(path)
                for path in sorted((self.workspace / "data").rglob("*"))
                if path.is_file()
            ],
            "should_do_data_exploration": True,
            "should_do_data_preprocessing": False,
            "should_prepare_hypothesis_testing_plan": True,
            "should_do_literature_search": True,
            "project_specific_goal_guidelines": "Use only supplied benchmark inputs; do not seek the hidden target study.",
            "excluded_citation_titles": [],
            "max_goal_refinement_iterations": 3,
        }
        _write_data_descriptions(
            self.workspace,
            objective,
            config["data_filenames"],
        )
        (self.workspace / HypothesisTestingStepsRunner.PROJECT_PARAMETERS_FILENAME).write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        output = self.workspace / "d2p_native"
        self.runner = HypothesisTestingStepsRunner(
            project_directory=self.workspace,
            output_directory=output,
            should_mock=False,
            should_remove_temp_folder=False,
            temp_folder_to_run_in=self.workspace / ".rac" / "d2p_temp",
        )
        self.runner.server_caller = self._server_caller
        self.runner.server_caller.set_current_stage_callback(self.runner._get_current_stage)
        self.runner.server_caller.set_api_cost_callback(self.runner._add_cost_to_stage)
        self.runner._create_or_clean_output_folder()
        self.runner._create_temp_folder_to_run_in()
        self.runner._pre_run_preparations()
        self.episode_id = episode_id
        self.objective = objective

    def initialize_native(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        self.initialize(episode_id=episode_id, workspace=workspace, objective=objective, seed=seed)

    def run_native(self) -> NativeRunResult:
        """Run data-to-paper's own all-steps loop, including native resets."""
        self._require_initialized()
        assert self.workspace is not None
        before = snapshot_workspace(self.workspace)
        calls_before, input_before = self.provider_calls, self.input_tokens
        output_before, cost_before = self.output_tokens, self.provider_cost_usd
        started = time.monotonic()
        self.runner.run_all_steps()
        self._normalize_products()
        report = self.workspace / "report" / "report.md"
        complete = report.is_file() and bool(report.read_text(encoding="utf-8", errors="replace").strip())
        self.terminal = True
        return NativeRunResult(
            status="completed" if complete else "stop",
            reason="data-to-paper native step runner completed" if complete else "data-to-paper returned without a report",
            native_iterations=1,
            artifacts_before=before,
            artifacts_after=snapshot_workspace(self.workspace),
            usage=Usage(
                provider_cost_usd=self.provider_cost_usd - cost_before,
                input_tokens=self.input_tokens - input_before,
                output_tokens=self.output_tokens - output_before,
                agent_calls=self.provider_calls - calls_before,
                wall_seconds=time.monotonic() - started,
                cost_source="provider_response" if self.cost_is_provider_reported else "unavailable",
                token_source="provider_response",
            ),
            native_status="completed" if complete else "missing_report",
        )

    def checkpoint(self) -> Checkpoint:
        self._require_initialized()
        next_id = self._next_native()
        issues = [] if self.terminal else [
            Issue(
                issue_id=f"native:{next_id}",
                kind="native_requirement",
                summary=f"data-to-paper capability {next_id} is incomplete",
                required_tags=REQUIRED_TAGS[next_id],
            )
        ]
        elapsed = time.monotonic() - self.started
        remaining = Budget(
            max(0.0, self.initial_budget.provider_cost_usd - self.provider_cost_usd),
            max(0, self.initial_budget.input_tokens - self.input_tokens),
            max(0, self.initial_budget.output_tokens - self.output_tokens),
            max(0, self.initial_budget.agent_calls - self.provider_calls),
            max(0.0, self.initial_budget.wall_seconds - elapsed),
            max(0, self.initial_budget.hops - self.hop),
        )
        return Checkpoint(
            self.episode_id,
            self.hop,
            self.objective,
            next_id if not self.terminal else "terminal",
            snapshot_workspace(self.workspace),
            issues,
            remaining,
            self._available_cards(),
            terminal=self.terminal,
        )

    def native_next(self, checkpoint: Checkpoint) -> str | None:
        return None if self.terminal else self._next_native()

    def invoke(self, capability_id: str, contract: WorkContract | None) -> InvocationResult:
        self._require_initialized()
        available = {card.capability_id for card in self._available_cards() if card.available}
        if capability_id not in available:
            raise ValueError(f"data-to-paper capability prerequisites are not satisfied: {capability_id}")
        before = snapshot_workspace(self.workspace)
        calls_before, input_before = self.provider_calls, self.input_tokens
        output_before, cost_before = self.output_tokens, self.provider_cost_usd
        output = io.StringIO()
        error = None
        started = time.monotonic()
        proposed_done = False
        interrupted = False
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                for stage in self._stages_for(capability_id):
                    self.runner.advance_stage(stage)
                    returned = self.runner._run_stage(stage)
                    if isinstance(stage, Enum) and isinstance(returned, type(stage)):
                        stages = list(type(stage))
                        if stages.index(returned) > stages.index(stage):
                            # Native stages return forward jumps when supplied
                            # products make intervening work unnecessary.
                            for candidate in ORDER:
                                native_stages = self._stages_for(candidate)
                                if native_stages and all(stages.index(s) < stages.index(returned) for s in native_stages):
                                    self.completed.add(candidate)
                            output.write(f"Native stage {stage.name} completed; next stage {returned.name}.\n")
                            break
                    if returned not in (None, True):
                        # Record a native reset request as an unresolved state;
                        # the common policy decides the next invocation.
                        interrupted = True
                        if returned is False:
                            error = "data-to-paper native stage terminated"
                        break
                    output.write(f"Native stage {getattr(stage, 'name', stage)} completed.\n")
            self._persist_output(capability_id, output.getvalue())
            self._normalize_products()
            if not interrupted:
                if capability_id == "compile":
                    report = self.workspace / "report" / "report.md"
                    if report.is_file() and report.read_text(encoding="utf-8", errors="replace").strip():
                        self.completed.add(capability_id)
                        self.terminal = True
                        proposed_done = True
                    else:
                        error = "data-to-paper compile produced no report"
                else:
                    self.completed.add(capability_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.hop += 1
        after = snapshot_workspace(self.workspace)
        return InvocationResult(
            capability_id,
            output.getvalue(),
            before,
            after,
            Usage(
                provider_cost_usd=self.provider_cost_usd - cost_before,
                input_tokens=self.input_tokens - input_before,
                output_tokens=self.output_tokens - output_before,
                agent_calls=self.provider_calls - calls_before,
                wall_seconds=time.monotonic() - started,
                cost_source="provider_response" if self.cost_is_provider_reported else "unavailable",
                token_source="provider_response",
            ),
            error=error,
            proposed_done=proposed_done,
        )

    def _stages_for(self, capability_id: str):
        s = self.stage
        return {
            "data_exploration": (s.DATA, s.EXPLORATION),
            "research_goal": (s.GOAL,),
            "literature_review_goal": (s.LITERATURE_REVIEW_GOAL, s.ASSESS_NOVELTY),
            "hypothesis_plan": (s.PLAN,),
            "data_analysis": (s.CODE,),
            "tables_figures": (s.DISPLAYITEMS,),
            "paper_writing": (
                s.INTERPRETATION,
                s.LITERATURE_REVIEW_WRITING,
                s.WRITING_RESULTS,
                s.WRITING_TITLE_AND_ABSTRACT,
                s.WRITING_METHODS,
                s.WRITING_INTRODUCTION,
                s.WRITING_DISCUSSION,
            ),
            "compile": (s.COMPILE,),
        }[capability_id]

    def _available_cards(self):
        next_index = ORDER.index(self._next_native()) if not self.terminal else len(ORDER)
        cards = []
        for card in self.cards:
            index = ORDER.index(card.capability_id)
            available = index <= min(next_index + 1, len(ORDER) - 1)
            cards.append(
                type(card)(
                    capability_id=card.capability_id,
                    description=card.description,
                    tags=card.tags,
                    readable_artifacts=card.readable_artifacts,
                    writable_artifacts=card.writable_artifacts,
                    produces=card.produces,
                    available=available,
                    authority_scope=card.authority_scope,
                    native_successors=card.native_successors,
                )
            )
        return cards

    def _next_native(self) -> str:
        for capability_id in ORDER:
            if capability_id not in self.completed:
                return capability_id
        return "compile"

    def _normalize_products(self) -> None:
        assert self.workspace is not None
        native = self.workspace / "d2p_native"
        report_dir = self.workspace / "report"
        report_dir.mkdir(exist_ok=True)
        candidates = [native / "paper.tex", native / "paper.pdf"]
        for source in candidates:
            if source.is_file():
                shutil.copyfile(source, report_dir / source.name)
        tex = report_dir / "paper.tex"
        if tex.is_file():
            shutil.copyfile(tex, report_dir / "report.md")
        for source in native.rglob("*") if native.is_dir() else ():
            if not source.is_file() or source.name in {"paper.tex", "paper.pdf"}:
                continue
            relative = source.relative_to(native)
            suffix = source.suffix.lower()
            if suffix == ".py":
                destination = self.workspace / "code" / "data_to_paper" / relative
            elif suffix in {".png", ".jpg", ".jpeg", ".svg"}:
                destination = self.workspace / "report" / "images" / relative
            elif suffix in {".csv", ".tsv", ".json", ".txt", ".pkl"}:
                destination = self.workspace / "outputs" / "data_to_paper" / relative
            else:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    def _persist_output(self, capability_id: str, output: str) -> None:
        assert self.workspace is not None
        destination = self.workspace / "state" / "data_to_paper" / f"{capability_id}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output, encoding="utf-8")

    def _install_model_adapter(self) -> None:
        import openai
        from data_to_paper.servers.llm_call import LLMResponse, LLMServerCaller, OPENAI_SERVER_CALLER

        bridge = self

        def call(_cls, messages, model_engine, **kwargs):
            if bridge.provider_calls >= bridge.initial_budget.agent_calls:
                raise RuntimeError("lifecycle agent-call budget exhausted")
            if time.monotonic() - bridge.started >= bridge.initial_budget.wall_seconds:
                raise TimeoutError("lifecycle wall-time budget exhausted")
            openai.api_key = bridge.api_key
            openai.api_base = os.environ.get("AGENT_API_BASE", "https://api.openai.com/v1")
            kwargs["max_tokens"] = min(
                int(kwargs.get("max_tokens", bridge.initial_budget.output_tokens)),
                max(1, bridge.initial_budget.output_tokens - bridge.output_tokens),
            )
            response = openai.ChatCompletion.create(
                model=bridge.model,
                messages=[message.to_llm_dict() for message in messages],
                **kwargs,
            )
            bridge.provider_calls += 1
            usage = response.get("usage", {})
            bridge.input_tokens += int(usage.get("prompt_tokens", 0) or 0)
            bridge.output_tokens += int(usage.get("completion_tokens", 0) or 0)
            cost = response.get("response_cost") or response.get("provider_cost")
            if cost is None:
                bridge.cost_is_provider_reported = False
            else:
                bridge.provider_cost_usd += float(cost)
            return LLMResponse(response["choices"][0]["message"]["content"])

        LLMServerCaller._get_server_response = classmethod(call)
        self._server_caller = OPENAI_SERVER_CALLER

    def _require_initialized(self) -> None:
        if self.workspace is None or self.runner is None:
            raise RuntimeError("DataToPaperBridge.initialize must be called first")


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            sample = handle.read(4096)
    except OSError:
        return True
    return b"\x00" in sample
