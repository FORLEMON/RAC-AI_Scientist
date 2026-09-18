from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..artifacts import snapshot_workspace
from ..bridge import HostBridge
from ..conditions import Condition
from ..issues import extract_review_issues, parse_review_score
from ..manifest import capability_cards, load_host_manifest
from ..reproducibility import seed_runtime
from ..schemas import Budget, Checkpoint, CoordinationDecision, InvocationResult, Issue, NativeRunResult, Usage, WorkContract
from ..sharednet import SharedNetInvite, SharedNetSession


ORDER = ("researcher", "experimenter", "writer", "reviewer", "planner")
SUCCESSORS = {
    "researcher": "experimenter",
    "experimenter": "writer",
    "coder": "writer",
    "writer": "reviewer",
    "reviewer": "planner",
    "planner": "experimenter",
}
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
        self._pending_transition: tuple[str, list[Issue]] | None = None
        self.condition = Condition.N0
        self.sharednet: SharedNetSession | None = None

    def configure_condition(self, condition: Condition | str) -> None:
        self.condition = Condition.parse(condition)

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
        os.environ.setdefault("OPENAI_API_KEY", self.api_key)
        if os.environ.get("AGENT_API_BASE"):
            os.environ.setdefault("OPENAI_API_BASE", os.environ["AGENT_API_BASE"])
        if str(self.upstream) not in sys.path:
            sys.path.insert(0, str(self.upstream))
        from ark.orchestrator import Orchestrator
        from ark.engines.cli import OpenHandsCLI

        self._install_openhands_usage(OpenHandsCLI)

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
        self._pending_transition = None
        self.open_issues = [] if native else [Issue("native:researcher", "native_requirement", "initial research framing is incomplete", required_tags=("planning",))]
        if not native and self.condition.enables("runtime_communication"):
            room_id = os.environ.get("SHAREDNET_ROOM_ID", "").strip()
            invite_text = os.environ.get("SHAREDNET_INVITE", "").strip()
            if not room_id:
                raise ValueError("ARK R1-R5 requires --sharednet-room-id or SHAREDNET_ROOM_ID")
            if not invite_text:
                raise ValueError("ARK R1-R5 requires SHAREDNET_INVITE with the selected Room's invite token")
            invite = SharedNetInvite.parse(
                invite_text,
                room_id=room_id,
                default_base=os.environ.get("SHAREDNET_BASE_URL", "https://www.sharednet.ai"),
            )
            self.sharednet = SharedNetSession(
                invite,
                episode_id,
                tuple(card.capability_id for card in self.cards),
            )
            self.sharednet.join()

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
        native_error = getattr(self.orchestrator, "_run_fatal", None) or getattr(self.orchestrator, "_terminal_error", None)
        if not native_error:
            self._normalize_report()

        after = snapshot_workspace(self.workspace)
        usage_after = self._usage_totals()
        paper_state = self.orchestrator.load_paper_state()
        native_status = str(paper_state.get("status", "unknown"))
        iterations = int(getattr(self.orchestrator, "iteration", 0) or 0)
        review_score = float(paper_state.get("current_score", 0) or 0)
        stopped = bool(getattr(self.orchestrator, "_stop_requested", False))
        if native_error:
            status = "failed"
            reason = native_error
        elif native_status == "accepted":
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

    def transaction_workspace(self) -> Path | None:
        return self.workspace

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
        timed_out = False
        next_issues: list[Issue] = []
        next_capability = self._successor(capability_id)
        prior_review = self._read_review() if capability_id == "reviewer" else ""
        try:
            prompt = render_contract_prompt(self.objective, capability_id, contract)
            sharednet = getattr(self, "sharednet", None)
            if sharednet is not None:
                prompt = sharednet.request(
                    capability_id,
                    self.hop,
                    prompt,
                    contract.contract_id if contract is not None else None,
                )
                if contract is not None:
                    prompt += "\nPrior Room messages are background, not additional assignments. Address only the issues in the current work contract and stay within its writable paths."
            if capability_id == "reviewer":
                self._clear_rendered_pages()
                try:
                    self.orchestrator.compile_latex()
                except Exception:
                    pass
            timeout = self._timeout()
            if capability_id == "researcher" and not self._research_prompts_specialized():
                output = self._run_native_research_specialization()
            else:
                output = self.orchestrator.run_agent(capability_id, prompt, timeout=timeout)
            if getattr(self.orchestrator, "_terminal_error", None):
                raise RuntimeError(self.orchestrator._terminal_error)
            timed_out = not output.strip() and time.monotonic() - started >= max(1, timeout - 1)
            if capability_id == "reviewer":
                review_text = self._review_text(output, previous=prior_review)
                self._persist_output(capability_id, review_text)
                score = parse_review_score(review_text)
                if score is not None:
                    metrics["review_score"] = score
                    proposed_done = score >= 8.0
                next_issues = extract_review_issues(review_text)
                if score is None:
                    next_issues.append(
                        Issue(
                            "review:score_missing",
                            "artifact",
                            "review score is missing; planner must turn the unstructured review into actionable work",
                            required_tags=("planning",),
                        )
                    )
                elif score < 8.0 and not next_issues:
                    next_issues.append(
                        Issue(
                            "review:below_threshold",
                            "methodology",
                            f"review score {score:.1f}/10 is below the acceptance threshold",
                            required_tags=("planning",),
                        )
                    )
            else:
                self._persist_output(capability_id, output)
                unassigned = [issue for issue in self.open_issues
                              if not issue.resolved and issue.issue_id not in contract.issue_ids] if contract else []
                next_issues = unassigned or [
                    Issue(
                        f"native:{next_capability}",
                        "native_requirement",
                        f"{next_capability} work remains",
                        required_tags=REQUIRED_TAGS[next_capability],
                    )
                ]
            if capability_id in {"writer", "reviewer"}:
                self._normalize_report()
            self._pending_transition = (next_capability, next_issues)
            if sharednet is not None:
                sharednet.result(capability_id, self.hop, output, next_capability)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._pending_transition = None
            sharednet = getattr(self, "sharednet", None)
            if sharednet is not None:
                try:
                    sharednet.result(capability_id, self.hop, output, capability_id, error=error)
                except Exception:
                    pass
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
            timed_out=timed_out,
            error=error,
            proposed_next=capability_id if timed_out or not output.strip() else next_capability,
            proposed_done=proposed_done,
            metrics=metrics,
            terminal_error=bool(getattr(self.orchestrator, "_terminal_error", None)),
        )

    def accept_invocation(self, result: InvocationResult, evaluation: CoordinationDecision) -> None:
        if self._pending_transition is None:
            return
        self.native_capability, self.open_issues = self._pending_transition
        self._pending_transition = None
        sharednet = getattr(self, "sharednet", None)
        if sharednet is not None:
            sharednet.disposition(
                self.hop - 1,
                accepted=True,
                reason=evaluation.reason,
                next_role=self.native_capability,
            )

    def reject_invocation(self, result: InvocationResult, evaluation: CoordinationDecision) -> None:
        self._pending_transition = None
        sharednet = getattr(self, "sharednet", None)
        if sharednet is not None:
            sharednet.disposition(
                self.hop - 1,
                accepted=False,
                reason=evaluation.reason,
                next_role=result.proposed_next,
            )

    def _successor(self, capability_id: str) -> str:
        try:
            return SUCCESSORS[capability_id]
        except KeyError as exc:
            raise ValueError(f"ARK capability has no declared successor: {capability_id}") from exc

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

    def _research_prompts_specialized(self) -> bool:
        """Return whether ARK's native researcher initialized downstream prompts."""
        return not self._missing_research_specializations()

    @staticmethod
    def _research_specialization_section(text: str) -> str:
        for heading in re.finditer(r"(?m)^## Project-Specific Knowledge[ \t]*\r?$", text):
            body = re.split(r"(?m)^#{1,2}(?:[ \t]+|$)", text[heading.end():], maxsplit=1)[0]
            if body.strip():
                return (text[heading.start():heading.end()] + body).strip()
        return ""

    def _missing_research_specializations(self) -> list[Path]:
        assert self.workspace is not None
        context = self.workspace / "auto_research" / "state" / "project_context.md"
        agents = self.workspace / ".rac" / "ark_project" / "agents"
        missing = []
        if not context.is_file() or not context.read_text(encoding="utf-8").strip():
            missing.append(context)
        for name in ("experimenter", "planner", "reviewer", "writer", "coder"):
            prompt = agents / f"{name}.prompt"
            if not prompt.is_file() or not self._research_specialization_section(prompt.read_text(encoding="utf-8")):
                missing.append(prompt)
        return missing

    def _restore_native_research_specializations(self) -> None:
        """Hand native on-disk sections to existing prompts when the return was a receipt."""
        assert self.workspace is not None
        state = self.workspace / "auto_research" / "state"
        for prompt in self._missing_research_specializations():
            source = state / f"{prompt.stem}_specialization.md"
            if prompt.suffix != ".prompt" or not prompt.is_file() or not source.is_file():
                continue
            section = self._research_specialization_section(source.read_text(encoding="utf-8"))
            if section:
                with prompt.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n\n{section}\n")

    def _run_native_research_specialization(self) -> str:
        """Run ARK's native research compiler before RAC schedules later roles.

        ARK's researcher owns proposal interpretation, project_context creation,
        skill/citation bootstrap, and project-specific specialization of every
        downstream role prompt.  Calling ``run_agent('researcher', ...)`` alone
        bypasses those host semantics, so the RAC bridge exposes the complete
        idempotent research phase as the first researcher capability invocation.
        """
        assert self.workspace is not None
        research_phase = getattr(self.orchestrator, "_run_research_phase", None)
        if not callable(research_phase):
            raise RuntimeError("ARK orchestrator does not expose its native research phase")
        research_phase()
        if terminal_error := getattr(self.orchestrator, "_terminal_error", None):
            raise RuntimeError(str(terminal_error))
        self._restore_native_research_specializations()

        # A resumed/partially initialized project may already have context while
        # one or more prompt append operations were interrupted.  ARK's research
        # phase skips specialization when project_context.md exists, so repair
        # only the missing prompt specializations through its native idempotent
        # helper before admitting the capability result.
        if not self._research_prompts_specialized():
            specialize = getattr(self.orchestrator, "_specialize_agent_prompts", None)
            if callable(specialize):
                specialize()
                if terminal_error := getattr(self.orchestrator, "_terminal_error", None):
                    raise RuntimeError(str(terminal_error))
                self._restore_native_research_specializations()
        missing = self._missing_research_specializations()
        if missing:
            raise RuntimeError("ARK researcher specialization is incomplete: " + ", ".join(path.name for path in missing))

        context = self.workspace / "auto_research" / "state" / "project_context.md"
        return (
            "ARK native research phase and downstream prompt specialization completed.\n\n"
            + context.read_text(encoding="utf-8").strip()
        )

    def _read_review(self) -> str:
        assert self.workspace is not None
        review = self.workspace / "auto_research" / "state" / "latest_review.md"
        try:
            return review.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return ""

    def _review_text(self, fallback: str, *, previous: str | None = None) -> str:
        text = self._read_review()
        if text and (previous is None or text != previous):
            return text
        return fallback

    def _clear_rendered_pages(self) -> None:
        assert self.workspace is not None
        for path in (self.workspace / "report").glob("page_*.png"):
            path.unlink(missing_ok=True)

    def _normalize_report(self) -> None:
        assert self.workspace is not None
        source = self.workspace / "report" / "main.tex"
        if not source.is_file() or source.stat().st_size == 0:
            return
        report = source.parent / "report.md"
        if report.is_file():
            existing = report.read_text(encoding="utf-8")
            if "TO BE WRITTEN" in existing or "Work in progress." in existing:
                drafts = self.workspace / "state" / "ark"
                drafts.mkdir(parents=True, exist_ok=True)
                shutil.move(report, drafts / f"report_draft_{self.hop}.md")
        if "Work in progress." in source.read_text(encoding="utf-8"):
            return
        temporary = source.parent / ".report.md.tmp"
        markdown_source = source.parent / ".report.source.tex"
        try:
            markdown_source.write_text(self._markdown_latex_source(source), encoding="utf-8")
            command = ["pandoc", "--from=latex", "--to=gfm", "--wrap=none", f"--output={temporary.name}", markdown_source.name]
            bibliography_names = [name.strip() for group in re.findall(r"\\bibliography\{([^}]+)\}", source.read_text(encoding="utf-8"))
                                  for name in group.split(",")]
            if bibliography_names:
                command += ["--citeproc", "--metadata=reference-section-title:References"]
                command += [f"--bibliography={name if name.endswith('.bib') else name + '.bib'}" for name in bibliography_names]
            subprocess.run(
                command,
                cwd=source.parent,
                check=True,
            )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("pandoc did not produce a non-empty Markdown report")
            temporary.replace(report)
        finally:
            temporary.unlink(missing_ok=True)
            markdown_source.unlink(missing_ok=True)

    @staticmethod
    def _markdown_latex_source(source: Path) -> str:
        text = source.read_text(encoding="utf-8")
        # ARK's PDF page-count probe is layout instrumentation, not report content.
        body_end_marker = (
            r"\makeatletter\pdfsavepos"
            r"\write\@auxout{\string\gdef\string\arkBodyEndY{\the\pdflastypos}"
            r"\string\gdef\string\arkPageH{\number\pdfpageheight}"
            r"\string\gdef\string\arkBodyEndPage{\arabic{page}}}"
            r"\makeatother"
        )
        text = text.replace(body_end_marker, "")
        bibliography = re.search(
            r"\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}",
            text,
            re.S,
        )
        citation_numbers: dict[str, int] = {}
        if bibliography:
            for index, key in enumerate(re.findall(r"\\bibitem\{([^}]+)\}", bibliography.group(1)), start=1):
                citation_numbers[key] = index

            converted = re.sub(r"\\bibitem\{[^}]+\}", "\\\\item ", bibliography.group(1))
            replacement = "\\section*{References}\n\\begin{enumerate}\n" + converted + "\n\\end{enumerate}"
            text = text[: bibliography.start()] + replacement + text[bibliography.end() :]

        def replace_citation(match: re.Match[str]) -> str:
            keys = [item.strip() for item in match.group(1).split(",")]
            labels = [str(citation_numbers.get(key, key)) for key in keys]
            return "[" + ", ".join(labels) + "]"

        if bibliography:
            text = re.sub(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", replace_citation, text)
        aux = source.with_suffix(".aux")
        if aux.is_file():
            try:
                labels = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}", aux.read_text(encoding="utf-8")))
            except (OSError, UnicodeError):
                labels = {}
            text = re.sub(r"\\ref\{([^}]+)\}", lambda match: labels.get(match.group(1), match.group(1)), text)
        return text

    def _usage_totals(self) -> Usage:
        stats = getattr(self.orchestrator, "_agent_stats", []) if self.orchestrator else []
        return Usage(
            provider_cost_usd=sum(float(item.get("cost_usd", 0) or 0) for item in stats),
            input_tokens=sum(int(item.get("input_tokens", 0) or 0) for item in stats),
            output_tokens=sum(int(item.get("output_tokens", 0) or 0) for item in stats),
            agent_calls=sum(int(item.get("model_requests", 1)) for item in stats),
            cost_source="host_estimate",
            token_source="provider_response",
        )

    @staticmethod
    def _install_openhands_usage(cli_type: type) -> None:
        original = cli_type.parse_output
        if getattr(original, "_rac_counted", False):
            return

        def parse_output(cli, stdout):
            parsed = original(cli, stdout)
            if not isinstance(parsed, dict):
                return parsed
            conversation_id = parsed.get("conversation_id")
            usage = parsed.get("usage")
            if not conversation_id or not isinstance(usage, dict):
                return parsed
            root = Path(os.environ.get("ARK_OPENHANDS_CONV_DIR") or Path.home() / ".openhands" / "conversations")
            state = root / str(conversation_id) / "base_state.json"
            try:
                metrics = json.loads(state.read_text(encoding="utf-8"))["stats"]["usage_to_metrics"]
                if not isinstance(metrics, dict):
                    return parsed
                count = sum(
                    len(token_usages)
                    for item in metrics.values()
                    if isinstance(item, dict)
                    and isinstance((token_usages := item.get("token_usages")), (list, tuple))
                )
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                # Metrics are optional instrumentation. Missing or partially written
                # state must retain the conservative one-request-per-phase fallback.
                return parsed
            usage["model_requests"] = count
            return parsed

        parse_output._rac_counted = True
        cli_type.parse_output = parse_output

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
            f"research_idea: {json.dumps(self.objective, ensure_ascii=False)}\n"
        )
        if native:
            config += (
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
        "Create, modify, or delete files only under the declared writable paths. "
        "Any validation outside those paths must be strictly read-only.\n"
        "Do not claim completion unless the required evidence exists on disk."
    )
