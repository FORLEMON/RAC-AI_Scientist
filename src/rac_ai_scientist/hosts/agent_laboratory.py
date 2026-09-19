from __future__ import annotations

import contextlib
import io
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from ..artifacts import snapshot_workspace
from ..bridge import HostBridge
from ..manifest import capability_cards, load_host_manifest
from ..reproducibility import seed_runtime
from ..schemas import Budget, Checkpoint, InvocationResult, Issue, NativeRunResult, Usage, WorkContract
from .ark import render_contract_prompt


PHASES = (
    "literature_review",
    "plan_formulation",
    "data_preparation",
    "running_experiments",
    "results_interpretation",
    "report_writing",
    "report_refinement",
)
METHODS = {name: name for name in PHASES}
NATIVE_NAMES = {name: name.replace("_", " ") for name in PHASES}
REQUIRED_TAGS = {
    "literature_review": ("literature",),
    "plan_formulation": ("planning",),
    "data_preparation": ("code", "analysis"),
    "running_experiments": ("experiment",),
    "results_interpretation": ("analysis",),
    "report_writing": ("writing",),
    "report_refinement": ("terminal_review",),
}


class _ArxivTimedSession:
    def __init__(self, inner):
        self.inner = inner

    def get(self, url, **kwargs):
        return self.inner.get(url, timeout=(5, 30), **kwargs)


def _install_arxiv_transport() -> None:
    """Bound Agent Laboratory's pinned arxiv.py transport and surface search failures."""
    import arxiv
    from tools import ArxivSearch

    original_init = arxiv.Client.__init__
    if not getattr(original_init, "_rac_deadline", False):
        def init(client, *args, **kwargs):
            original_init(client, *args, **kwargs)
            client.num_retries = 0  # The native search tool already retries three times.
            client._session = _ArxivTimedSession(client._session)

        init._rac_deadline = True
        arxiv.Client.__init__ = init

    original_search = ArxivSearch.find_papers_by_str
    if not getattr(original_search, "_rac_error", False):
        def find_papers_by_str(search, query, N=20):
            papers = original_search(search, query, N)
            if papers is None:
                raise RuntimeError("arXiv API search failed after native retries")
            return papers

        find_papers_by_str._rac_error = True
        ArxivSearch.find_papers_by_str = find_papers_by_str


def _install_hf_data_search(workspace: Path) -> None:
    """Return the supplied benchmark inputs instead of searching external datasets."""
    import ai_lab_repo

    data = workspace / "data"
    paths = sorted(path.relative_to(workspace).as_posix() for path in data.iterdir()) if data.is_dir() else []
    notice = (
        "External Hugging Face dataset search is disabled for this benchmark. "
        "Read task.json and use only the supplied local data paths: "
        + ", ".join(repr(path) for path in paths)
    )

    class SuppliedDataSearch:
        def retrieve_ds(self, query):
            return []

        def results_str(self, datasets):
            return [notice]

    ai_lab_repo.HFDataSearch = SuppliedDataSearch


def _install_report_writing_scope() -> None:
    """Supply the globals incorrectly referenced by the pinned report writer."""
    import ai_lab_repo

    original = ai_lab_repo.LaboratoryWorkflow.report_writing
    if getattr(original, "_rac_report_scope", False):
        return

    def report_writing(workflow):
        # ponytail: pinned upstream uses module globals; remove when its self.* fix is locked.
        original.__globals__["research_topic"] = workflow.research_topic
        original.__globals__["compile_pdf"] = workflow.compile_pdf
        return original(workflow)

    report_writing._rac_report_scope = True
    ai_lab_repo.LaboratoryWorkflow.report_writing = report_writing


def _install_edit_range_validation() -> None:
    """Validate the native EDIT interval before it mutates the current code list."""
    import mlesolver

    original = mlesolver.Edit.execute_command
    if getattr(original, "_rac_edit_range", False):
        return

    def execute_command(editor, *args):
        start, end, lines, _, _ = args[0]
        if not 0 <= start <= end < len(lines):
            return False, None, f"EDIT range must satisfy 0 <= N <= M < {len(lines)}; got {start}:{end}."
        return original(editor, *args)

    execute_command._rac_edit_range = True
    mlesolver.Edit.execute_command = execute_command


class AgentLaboratoryBridge(HostBridge):
    """Thin state/invocation bridge over Agent Laboratory's existing phase methods."""

    host_id = "agent_laboratory"

    def __init__(self, upstream: Path, manifest: Path, budget: Budget, model: str, api_key: str):
        self.upstream = upstream.resolve()
        self.manifest = load_host_manifest(manifest)
        self.cards = capability_cards(self.manifest)
        self.initial_budget = budget
        self.model = model
        self.api_key = api_key
        self.workspace: Path | None = None
        self.episode_id = ""
        self.objective = ""
        self.workflow: Any = None
        self.hop = 0
        self.started = 0.0
        self.terminal = False
        self.provider_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.provider_cost_usd = 0.0
        self.cost_is_provider_reported = True
        self._contract_note: dict[str, Any] | None = None

    def initialize(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        if not (self.upstream / "ai_lab_repo.py").is_file():
            raise FileNotFoundError(f"Agent Laboratory checkout not found: {self.upstream}")
        self.workspace = workspace.resolve()
        seed_runtime(seed)
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / ".rac").mkdir(exist_ok=True)
        lab_dir = self.workspace / "agent_laboratory"
        (lab_dir / "src").mkdir(parents=True, exist_ok=True)
        (lab_dir / "tex").mkdir(exist_ok=True)
        (self.workspace / "state_saves").mkdir(exist_ok=True)
        self.episode_id = episode_id
        self.objective = objective
        self.started = time.monotonic()
        os.environ.setdefault("OPENAI_API_KEY", self.api_key)
        if str(self.upstream) not in sys.path:
            sys.path.insert(0, str(self.upstream))
        previous = Path.cwd()
        try:
            os.chdir(self.workspace)
            from ai_lab_repo import LaboratoryWorkflow

            _install_arxiv_transport()
            _install_hf_data_search(self.workspace)
            _install_report_writing_scope()
            _install_edit_range_validation()
            self._install_model_adapter()

            models = {native: self.model for native in NATIVE_NAMES.values()}
            models["paper refinement"] = self.model
            notes = [{"phases": list(NATIVE_NAMES.values()), "note": self._benchmark_note()}]
            human = {native: False for native in NATIVE_NAMES.values()}
            self.workflow = LaboratoryWorkflow(
                research_topic=objective,
                openai_api_key=self.api_key,
                max_steps=max(1, min(self.initial_budget.agent_calls, 100)),
                num_papers_lit_review=5,
                agent_model_backbone=models,
                notes=notes,
                human_in_loop_flag=human,
                compile_pdf=False,
                mlesolver_max_steps=3,
                papersolver_max_steps=3,
                paper_index=0,
                except_if_fail=True,
                parallelized=False,
                lab_dir=lab_dir.name,
                lab_index=0,
                agentRxiv=False,
            )
            self._save()
        finally:
            os.chdir(previous)

    def initialize_native(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        """Build the normal LaboratoryWorkflow used by Agent Laboratory itself."""
        self.initialize(episode_id=episode_id, workspace=workspace, objective=objective, seed=seed)

    def run_native(self) -> NativeRunResult:
        """Let LaboratoryWorkflow own all phase, retry, and refinement decisions."""
        self._require_initialized()
        assert self.workspace is not None
        before = snapshot_workspace(self.workspace)
        calls_before, input_before = self.provider_calls, self.input_tokens
        output_before, cost_before = self.output_tokens, self.provider_cost_usd
        started = time.monotonic()
        previous = Path.cwd()
        try:
            os.chdir(self.workspace)
            self.workflow.perform_research()
            self._persist_products()
            self._normalize_report()
            self.terminal = True
            self._save()
        finally:
            os.chdir(previous)
        report = self.workspace / "report" / "report.md"
        complete = report.is_file() and bool(report.read_text(encoding="utf-8", errors="replace").strip())
        return NativeRunResult(
            status="completed" if complete else "stop",
            reason="Agent Laboratory native workflow completed" if complete else "Agent Laboratory returned without a report",
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

    def _benchmark_note(self) -> str:
        assert self.workspace is not None
        return (
            f"Work only inside {self.workspace}. Use the supplied data/ and related_work/. "
            "The hidden target study is unavailable and must not be sought. Persist code under code/, "
            "results under outputs/, and the final ResearchClawBench report at report/report.md. "
            "Saved code must use workspace-relative paths; never embed the episode path."
        )

    def checkpoint(self) -> Checkpoint:
        self._require_initialized()
        next_id = self._next_native()
        issues = [] if self.terminal else [
            Issue(
                issue_id=f"native:{next_id}",
                kind="native_requirement",
                summary=f"Agent Laboratory phase {next_id} is incomplete",
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
        if capability_id not in METHODS:
            raise ValueError(f"unknown Agent Laboratory capability: {capability_id}")
        available = {card.capability_id for card in self._available_cards() if card.available}
        if capability_id not in available:
            raise ValueError(f"Agent Laboratory capability prerequisites are not satisfied: {capability_id}")
        before = snapshot_workspace(self.workspace)
        calls_before = self.provider_calls
        input_before = self.input_tokens
        output_before = self.output_tokens
        cost_before = self.provider_cost_usd
        output = io.StringIO()
        error = None
        started = time.monotonic()
        previous = Path.cwd()
        proposed_done = False
        try:
            os.chdir(self.workspace)
            phase_prompt = render_contract_prompt(self.objective, capability_id, contract)
            phase_prompt = self.communication_prompt(capability_id, phase_prompt, contract)
            if contract is not None or getattr(self, "sharednet", None) is not None:
                if self._contract_note is None:
                    self._contract_note = {"phases": [], "note": ""}
                    self.workflow.notes.append(self._contract_note)
                self._contract_note["phases"] = [NATIVE_NAMES[capability_id]]
                self._contract_note["note"] = phase_prompt
            method = getattr(self.workflow, METHODS[capability_id])
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                returned = method()
            native_name = NATIVE_NAMES[capability_id]
            if capability_id == "report_refinement":
                if returned:
                    for phase in NATIVE_NAMES.values():
                        if phase not in {"literature review"}:
                            self.workflow.phase_status[phase] = False
                else:
                    self.workflow.phase_status[native_name] = True
                    self.terminal = True
                    proposed_done = True
            elif not returned:
                self.workflow.phase_status[native_name] = True
            self._persist_products()
            if capability_id == "report_refinement":
                self._persist_review(output.getvalue())
            self._normalize_report()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            os.chdir(previous)
            self.hop += 1
            self._save()
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

    def _available_cards(self):
        next_index = PHASES.index(self._next_native()) if not self.terminal else len(PHASES)
        cards = []
        for card in self.cards:
            index = PHASES.index(card.capability_id)
            # Completed phases and the next two phases remain callable. This is
            # prerequisite admission, not a routing preference.
            available = index <= min(next_index + 1, len(PHASES) - 1)
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
        for capability_id in PHASES:
            if not self.workflow.phase_status.get(NATIVE_NAMES[capability_id], False):
                return capability_id
        return "report_refinement"

    def _normalize_report(self) -> None:
        assert self.workspace is not None
        source = self.workspace / "agent_laboratory" / "report.txt"
        if source.is_file():
            destination = self.workspace / "report" / "report.md"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    def _persist_products(self) -> None:
        """Serialize native in-memory products; this does not judge their quality."""
        assert self.workspace is not None
        state_dir = self.workspace / "state" / "agent_laboratory"
        output_dir = self.workspace / "outputs" / "agent_laboratory"
        code_dir = self.workspace / "code" / "agent_laboratory"
        state_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        code_dir.mkdir(parents=True, exist_ok=True)
        mappings = (
            (self.workflow.phd, "lit_review", state_dir / "literature_review.txt"),
            (self.workflow.phd, "plan", state_dir / "plan.txt"),
            (self.workflow.phd, "dataset_code", code_dir / "load_data.py"),
            (self.workflow.phd, "results_code", code_dir / "run_experiments.py"),
            (self.workflow.phd, "exp_results", output_dir / "experiment_results.txt"),
            (self.workflow.phd, "interpretation", state_dir / "interpretation.txt"),
            (self.workflow.phd, "report", self.workspace / "report" / "report.md"),
        )
        for owner, attribute, destination in mappings:
            value = getattr(owner, attribute, None)
            if value is None or value == "":
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(str(value), encoding="utf-8")

    def _persist_review(self, output: str) -> None:
        assert self.workspace is not None
        destination = self.workspace / "state" / "agent_laboratory" / "report_refinement_review.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output, encoding="utf-8")

    def _install_model_adapter(self) -> None:
        """Force every Agent Laboratory role through the experiment's exact model endpoint."""
        import ai_lab_repo
        import agents
        import inference
        import mlesolver
        import papersolver
        from openai import OpenAI

        base_url = os.environ.get("AGENT_API_BASE") or None
        client = OpenAI(api_key=self.api_key, base_url=base_url)

        def query_model(*, model_str, prompt, system_prompt, temp=None, **_kwargs):
            if self.provider_calls >= self.initial_budget.agent_calls:
                raise RuntimeError("lifecycle agent-call budget exhausted")
            if time.monotonic() - self.started >= self.initial_budget.wall_seconds:
                raise TimeoutError("lifecycle wall-time budget exhausted")
            request: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            if temp is not None:
                request["temperature"] = temp
            request["max_tokens"] = max(1, self.initial_budget.output_tokens - self.output_tokens)
            response = client.chat.completions.create(**request)
            self.provider_calls += 1
            usage = getattr(response, "usage", None)
            self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            cost = _response_cost(response)
            if cost is None:
                self.cost_is_provider_reported = False
            else:
                self.provider_cost_usd += cost
            return response.choices[0].message.content or ""

        for module in (inference, agents, mlesolver, papersolver, ai_lab_repo):
            if hasattr(module, "query_model"):
                setattr(module, "query_model", query_model)

    def _save(self) -> None:
        assert self.workspace is not None
        with (self.workspace / ".rac" / "agent_laboratory.pkl").open("wb") as handle:
            pickle.dump(self.workflow, handle)

    def _require_initialized(self) -> None:
        if self.workspace is None or self.workflow is None:
            raise RuntimeError("AgentLaboratoryBridge.initialize must be called first")


def _response_cost(response: Any) -> float | None:
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict) and hidden.get("response_cost") is not None:
        return float(hidden["response_cost"])
    extra = getattr(response, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("response_cost", "cost", "provider_cost"):
            if extra.get(key) is not None:
                return float(extra[key])
    return None
