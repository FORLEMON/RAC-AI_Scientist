from __future__ import annotations

import contextlib
import io
import json
import os
import pickle
import re
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

# The lifecycle output budget is cumulative across the entire episode.  It must
# not be sent verbatim as one request's ``max_tokens``: a generous lifecycle
# budget can be larger than the provider's context window and get rejected
# before inference starts.  Keep a conservative per-request ceiling while still
# allowing many calls to consume the full lifecycle allowance over time.
MAX_COMPLETION_TOKENS_PER_REQUEST = 65_536


def _request_completion_limit(remaining_output_tokens: int) -> int:
    """Return a context-safe per-request limit from the cumulative remainder."""
    if remaining_output_tokens <= 0:
        raise RuntimeError("lifecycle output-token budget exhausted")
    return min(remaining_output_tokens, MAX_COMPLETION_TOKENS_PER_REQUEST)


def _is_terminal_provider_error(exc: Exception) -> bool:
    """Identify deterministic request/configuration failures that cannot retry."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    return type(exc).__name__ == "BadRequestError" or status_code in {400, 401, 403, 404, 422}


def _is_content_filter_error(exc: Exception) -> bool:
    """Return whether Azure rejected an otherwise valid call via content policy."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    body = getattr(exc, "body", None)
    text = f"{exc} {body}".lower()
    return status_code == 400 and any(
        marker in text for marker in ("content_filter", "content filter", "jailbreak")
    )


def _workflow_labels(system_prompt: str) -> list[str]:
    """Return the small set of workflow labels understood by Agent Laboratory."""
    known_labels = {
        "ADD_PAPER", "DIALOGUE", "EDIT", "EXPIRATION", "FULL_TEXT",
        "INTERPRETATION", "LATEX", "PLAN", "REPLACE", "SCORE",
        "SEARCH_HF", "SUBMIT_CODE", "SUMMARY", "python",
    }
    labels: list[str] = []
    for label in re.findall(r"```\s*([A-Za-z][A-Za-z0-9_-]*)", system_prompt):
        if label in known_labels and label not in labels:
            labels.append(label)
    return labels


def _retry_phase(system_prompt: str, prompt: str) -> str:
    combined = f"{system_prompt}\n{prompt}".lower().replace("_", " ")
    for phase in NATIVE_NAMES.values():
        if phase in combined:
            return phase
    return "scientific analysis"


def _scientific_retry_context(prompt: str, *, compact: bool) -> dict[str, str]:
    """Extract evidence-bearing prose without replaying instruction scaffolding."""
    cleaned = re.sub(r"```\s*[A-Za-z][A-Za-z0-9_-]*", "\n", prompt)
    cleaned = cleaned.replace("```", "\n")
    chunks = re.split(r"\n{2,}|(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    control = re.compile(
        r"\b(?:must|exactly|only|command|system prompt|developer message|"
        r"ignore|disregard|override|jailbreak|role[- ]?play)\b",
        re.IGNORECASE,
    )
    role_prefix = re.compile(
        r"^(?:professor|postdoc|phd student|ml engineer|software engineer|assistant|user)\s*:\s*",
        re.IGNORECASE,
    )
    categories = {
        "objective": ("objective", "research topic", "scientific goal", "task", "problem", "develop"),
        "literature": ("literature", "paper", "study", "finding", "abstract", "related work", "prior work"),
        "data": ("data", "dataset", "input", "file", "sample", "measurement", "benchmark"),
    }
    selected: dict[str, list[str]] = {name: [] for name in categories}
    fallback: list[str] = []
    seen: set[str] = set()
    for raw in chunks:
        text = role_prefix.sub("", " ".join(raw.split())).strip(" -:\t")
        if len(text) < 12 or control.search(text):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        fallback.append(text)
        lowered = text.lower()
        for name, keywords in categories.items():
            if any(keyword in lowered for keyword in keywords):
                selected[name].append(text)

    limits = (
        {"objective": 1200, "literature": 0, "data": 700}
        if compact
        else {"objective": 2200, "literature": 2200, "data": 1600}
    )
    result: dict[str, str] = {}
    for name, limit in limits.items():
        if not limit:
            continue
        items = selected[name]
        if not items and name == "objective":
            items = fallback[:2]
        value = "\n".join(items)
        if value:
            result[name] = value[:limit]
    return result


def _content_filter_retry_messages(
    system_prompt: str,
    prompt: str,
    *,
    level: int = 1,
) -> list[dict[str, str]]:
    """Build a fresh, bounded retry thread without replaying the raw prompt."""
    if level not in {1, 2}:
        raise ValueError(f"unsupported content-filter retry level: {level}")
    labels = _workflow_labels(system_prompt)
    label = labels[0] if labels else "SUMMARY"
    phase = _retry_phase(system_prompt, prompt)
    sections = _scientific_retry_context(prompt, compact=level == 2)
    rendered = []
    for name in ("objective", "literature", "data"):
        value = sections.get(name)
        if value:
            rendered.append(f"{name.title()}:\n{value}")
    context = "\n\n".join(rendered) or "Objective:\nContinue the current scientific task."
    if level == 1:
        system = f"Scientific research assistant working on {phase}. Use neutral academic prose."
        request = (
            f"{context}\n\nPrepare the next concise {phase} result.\n\n"
            f"Format:\n```{label}\n<scientific content>\n```"
        )
    else:
        system = "Scientific research assistant."
        request = (
            f"{context}\n\nDraft the next short scientific result.\n\n"
            f"Format:\n```{label}\n<scientific content>\n```"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": request},
    ]


class _FilteredCompletionError(RuntimeError):
    status_code = 400

    def __init__(self, response: Any):
        super().__init__("content_filter: provider returned an empty filtered completion")
        self.usage = getattr(response, "usage", None)


class _EmptyCompletionError(RuntimeError):
    def __init__(self, response: Any):
        super().__init__("provider returned an empty completion")
        self.usage = getattr(response, "usage", None)


def _completion_text(response: Any) -> str:
    choice = response.choices[0]
    content = choice.message.content or ""
    finish_reason = str(getattr(choice, "finish_reason", "") or "").lower()
    if finish_reason == "content_filter":
        raise _FilteredCompletionError(response)
    if not content.strip():
        raise _EmptyCompletionError(response)
    return content


def _usage_counts(source: Any) -> tuple[int, int]:
    """Read token usage from a normal response or a provider exception body."""
    usage = getattr(source, "usage", None)
    if usage is None:
        body = getattr(source, "body", None)
        if isinstance(body, dict):
            usage = body.get("usage")
    if isinstance(usage, dict):
        return int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


class _ArxivTimedSession:
    def __init__(self, inner):
        self.inner = inner

    def get(self, url, **kwargs):
        return self.inner.get(url, timeout=(5, 30), **kwargs)


ARXIV_ID_PATTERN = re.compile(
    r"(?:(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/|arxiv\s*:\s*)?"
    r"(?<!\d)(\d{4}\.\d{4,5})(v\d+)?(?!\d)",
    re.IGNORECASE,
)

RCB_BENCHMARK_NAME = "researchclawbench"


def _is_researchclawbench_workspace(workspace: Path | None) -> bool:
    """Require an explicit RAC-created marker before applying RCB-only adapters."""
    if workspace is None:
        return False
    marker = workspace / ".rac" / "benchmark.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("benchmark") == RCB_BENCHMARK_NAME


def _arxiv_aliases(value: str) -> list[str]:
    """Return stable arXiv identifiers found in metadata, text, IDs, or URLs."""
    aliases: list[str] = []
    for match in ARXIV_ID_PATTERN.finditer(value):
        base = match.group(1)
        version = match.group(2)
        for alias in (base, f"{base}{version}" if version else None):
            if alias and alias not in aliases:
                aliases.append(alias)
    return aliases


def _lookup_local_paper(
    papers: dict[str, dict[str, Any]], query: str
) -> dict[str, Any] | None:
    """Resolve synthetic IDs and common arXiv ID spellings to a supplied PDF."""
    candidate = query.strip()
    if candidate in papers:
        return papers[candidate]
    query_aliases = _arxiv_aliases(candidate)
    for paper in papers.values():
        aliases = paper.get("aliases", ())
        if candidate in aliases or any(alias in aliases for alias in query_aliases):
            return paper
    return None


def _local_literature(workspace: Path) -> dict[str, dict[str, Any]]:
    """Load benchmark-supplied PDFs for Agent Laboratory's native literature tools."""
    papers: dict[str, dict[str, Any]] = {}
    related_work = workspace / "related_work"
    if not related_work.is_dir():
        return papers
    from pypdf import PdfReader

    for index, path in enumerate(sorted(related_work.glob("*.pdf"))):
        try:
            reader = PdfReader(str(path))
            pages = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append(f"--- Page {page_number} ---\n{text}")
        except Exception:
            continue
        full_text = "\n".join(pages).strip()
        if not full_text:
            continue
        metadata = getattr(reader, "metadata", None)
        title = str(getattr(metadata, "title", "") or path.stem).strip()
        local_id = f"local-paper-{index:03d}"
        metadata_text = " ".join(
            str(value or "")
            for value in (
                title,
                getattr(metadata, "subject", ""),
                getattr(metadata, "keywords", ""),
            )
        )
        arxiv_aliases = _arxiv_aliases(f"{metadata_text}\n{full_text}")
        paper_id = arxiv_aliases[0] if arxiv_aliases else local_id
        papers[paper_id] = {
            "title": title,
            "summary": " ".join(full_text.split())[:3000],
            "full_text": full_text[:50000],
            "path": path.relative_to(workspace).as_posix(),
            "aliases": tuple(dict.fromkeys((local_id, paper_id, *arxiv_aliases))),
        }
    return papers


def _install_arxiv_transport(workspace: Path | None = None) -> int:
    """Prefer supplied PDFs; otherwise bound arxiv.py and surface search failures."""
    import arxiv
    from tools import ArxivSearch

    local_papers = (
        _local_literature(workspace)
        if _is_researchclawbench_workspace(workspace)
        else {}
    )
    ArxivSearch._rac_local_papers = local_papers

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
            supplied = getattr(type(search), "_rac_local_papers", {})
            if supplied:
                summaries = []
                for paper_id, paper in list(supplied.items())[:N]:
                    summaries.append(
                        f"Title: {paper['title']}\n"
                        f"Summary: {paper['summary']}\n"
                        f"Publication Date: supplied benchmark input\n"
                        f"arXiv paper ID: {paper_id}"
                    )
                return "\n\n".join(summaries)
            papers = original_search(search, query, N)
            if papers is None:
                raise RuntimeError("arXiv API search failed after native retries")
            return papers

        find_papers_by_str._rac_error = True
        ArxivSearch.find_papers_by_str = find_papers_by_str

    original_full_text = ArxivSearch.retrieve_full_paper_text
    if not getattr(original_full_text, "_rac_local", False):
        def retrieve_full_paper_text(search, query, MAX_LEN=50000):
            supplied = getattr(type(search), "_rac_local_papers", {})
            paper = _lookup_local_paper(supplied, query)
            if paper is not None:
                return paper["full_text"][:MAX_LEN]
            # The requested paper is genuinely absent from the benchmark
            # bundle. Preserve Agent Laboratory's native arXiv fallback.
            return original_full_text(search, query, MAX_LEN=MAX_LEN)

        retrieve_full_paper_text._rac_local = True
        ArxivSearch.retrieve_full_paper_text = retrieve_full_paper_text
    return len(local_papers)


def _command_payload(response: str, command: str) -> str | None:
    match = re.search(rf"```{re.escape(command)}\s*\n(.*?)```", response, re.DOTALL)
    return match.group(1).strip() if match else None


class _ResearchClawBenchLiteratureGuard:
    """Pickle-safe bounded wrapper around the native PhD inference method."""

    def __init__(self, phd: Any, papers: dict[str, dict[str, Any]]):
        self.phd = phd
        self.papers = papers
        self.original_inference = phd.inference
        self.pending_id: str | None = None

    def _remaining_papers(self) -> list[tuple[str, dict[str, Any]]]:
        added = {
            str(item.get("arxiv_id") or "").strip()
            for item in getattr(self.phd, "lit_review", [])
            if isinstance(item, dict)
        }
        return [
            (paper_id, paper)
            for paper_id, paper in self.papers.items()
            if not any(_lookup_local_paper({paper_id: paper}, item) for item in added)
        ]

    def __call__(self, *args, **kwargs):
        response = self.original_inference(*args, **kwargs)
        phase = args[1] if len(args) > 1 else kwargs.get("phase")
        if phase != "literature review":
            return response

        remaining = self._remaining_papers()
        if not remaining:
            return response
        remaining_by_id = dict(remaining)
        if self.pending_id not in remaining_by_id:
            self.pending_id = None

        if self.pending_id is not None:
            payload = _command_payload(str(response), "ADD_PAPER")
            candidate = payload.splitlines()[0].strip() if payload else ""
            if candidate and _lookup_local_paper(
                {self.pending_id: remaining_by_id[self.pending_id]},
                candidate,
            ):
                self.pending_id = None
                return response
            paper = remaining_by_id[self.pending_id]
            summary = " ".join(str(paper["summary"]).split())[:1200]
            paper_id = self.pending_id
            self.pending_id = None
            return f"```ADD_PAPER\n{paper_id}\n{summary}\n```"

        paper_id, _paper = remaining[0]
        self.pending_id = paper_id
        return f"```FULL_TEXT\n{paper_id}\n```"


def _install_researchclawbench_literature_guard(workflow: Any, workspace: Path) -> int:
    """Bound the native paper protocol for RCB-supplied related work.

    DeepSeek can repeatedly issue SUMMARY for the same fixed local corpus until
    Agent Laboratory exhausts its phase retry limit.  RCB already declares the
    PDFs in ``related_work`` as the complete host-visible literature set, so
    guide the native protocol through each supplied paper exactly once.  A
    model-produced ADD_PAPER summary is retained when it names the pending
    paper; otherwise a short extractive fallback prevents another search loop.

    The explicit benchmark marker is mandatory so a future PaperBench or other
    benchmark workspace containing PDFs cannot activate this behavior.
    """
    if not _is_researchclawbench_workspace(workspace):
        return 0
    papers = _local_literature(workspace)
    if not papers:
        return 0

    workflow.phd.inference = _ResearchClawBenchLiteratureGuard(workflow.phd, papers)
    return len(papers)


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

            local_paper_count = _install_arxiv_transport(self.workspace)
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
            guarded_paper_count = _install_researchclawbench_literature_guard(
                self.workflow,
                self.workspace,
            )
            if (
                _is_researchclawbench_workspace(self.workspace)
                and guarded_paper_count != local_paper_count
            ):
                raise RuntimeError(
                    "ResearchClawBench local-literature adapters disagreed about the supplied PDF count"
                )
            if isinstance(local_paper_count, int) and local_paper_count > 0:
                # The benchmark bundle is the complete allowed literature set.
                # Do not keep iterating toward the upstream default of five when
                # fewer valid PDFs were supplied.
                self.workflow.num_papers_lit_review = min(
                    self.workflow.num_papers_lit_review,
                    local_paper_count,
                )
                self.workflow.num_ref_papers = min(
                    self.workflow.num_ref_papers,
                    local_paper_count,
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
        failure: Exception | None = None
        try:
            os.chdir(self.workspace)
            self.workflow.perform_research()
            self._persist_products()
            self._normalize_report()
            self.terminal = True
            self._save()
        except Exception as exc:
            # Native N0 used to let this escape, causing the episode writer to
            # lose the usage already recorded by failed provider calls.
            failure = exc
        finally:
            os.chdir(previous)
        report = self.workspace / "report" / "report.md"
        complete = report.is_file() and bool(report.read_text(encoding="utf-8", errors="replace").strip())
        if failure is not None:
            status = "failed"
            reason = f"{type(failure).__name__}: {failure}"
            native_status = "failed"
        else:
            status = "completed" if complete else "stop"
            reason = (
                "Agent Laboratory native workflow completed"
                if complete
                else "Agent Laboratory returned without a report"
            )
            native_status = "completed" if complete else "missing_report"
        return NativeRunResult(
            status=status,
            reason=reason,
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
            native_status=native_status,
        )

    def _benchmark_note(self) -> str:
        assert self.workspace is not None
        return (
            f"Work only inside {self.workspace}. Use the supplied data/ and related_work/. "
            "When related_work contains PDFs, use those supplied papers before any external literature search; "
            "external arXiv research is only a fallback when no local PDF is supplied. "
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
        terminal_error = False
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
            terminal_error = _is_terminal_provider_error(exc)
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
            terminal_error=terminal_error,
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
            request: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            if temp is not None:
                request["temperature"] = temp
            response = None
            content = None
            for attempt in range(3):
                if self.provider_calls >= self.initial_budget.agent_calls:
                    raise RuntimeError("lifecycle agent-call budget exhausted")
                if time.monotonic() - self.started >= self.initial_budget.wall_seconds:
                    raise TimeoutError("lifecycle wall-time budget exhausted")
                request["max_tokens"] = _request_completion_limit(
                    self.initial_budget.output_tokens - self.output_tokens
                )
                self.provider_calls += 1
                try:
                    response = client.chat.completions.create(**request)
                    content = _completion_text(response)
                except Exception as exc:
                    prompt_tokens, completion_tokens = _usage_counts(exc)
                    self.input_tokens += prompt_tokens
                    self.output_tokens += completion_tokens
                    if attempt < 2 and _is_content_filter_error(exc):
                        retry_level = attempt + 1
                        print(
                            f"[RAC] provider-filter retry level {retry_level} "
                            "with reduced scientific context",
                            file=sys.stderr,
                        )
                        request["messages"] = _content_filter_retry_messages(
                            system_prompt,
                            prompt,
                            level=retry_level,
                        )
                        continue
                    raise
                break
            assert response is not None
            assert content is not None
            prompt_tokens, completion_tokens = _usage_counts(response)
            self.input_tokens += prompt_tokens
            self.output_tokens += completion_tokens
            cost = _response_cost(response)
            if cost is None:
                self.cost_is_provider_reported = False
            else:
                self.provider_cost_usd += cost
            return content

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
