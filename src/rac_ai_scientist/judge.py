from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Judge did not return a JSON object: {text[:300]!r}")

    result = json.loads(text[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("Judge response is not a JSON object")
    return result


def _image_part(path: str) -> dict[str, Any]:
    image_path = Path(path)
    mime_type = mimetypes.guess_type(image_path.name)[0]

    supported = {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
    }
    if mime_type not in supported:
        raise ValueError(
            f"Unsupported judge image format: {image_path} ({mime_type})"
        )

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded}",
            "detail": "auto",
        },
    }


class AzureChatCompletionsJudge:
    """LLMAgent-compatible Azure Chat Completions judge."""

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        model_version: str,
        system_prompt: str = "",
        time_limit: int = 120,
        **_: Any,
    ) -> None:
        api_version = os.environ.get(
            "JUDGE_API_VERSION",
            "2025-04-01-preview",
        )

        if not api_key:
            raise ValueError("JUDGE_API_KEY is empty")
        if not api_base:
            raise ValueError("JUDGE_API_BASE is empty")
        if not model_version:
            raise ValueError("JUDGE_MODEL_NAME is empty")

        self.model = model_version
        self.system_prompt = system_prompt
        self.max_completion_tokens = int(
            os.environ.get("JUDGE_MAX_COMPLETION_TOKENS", "1000")
        )

        # Keep this import lazy: non-judge CLI commands and unit tests must not
        # require the Azure SDK merely because this integration module exists.
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            azure_endpoint=api_base.rstrip("/"),
            api_key=api_key,
            api_version=api_version,
            timeout=float(time_limit),
            max_retries=0,
        )

    def __call__(
        self,
        prompt: str,
        *,
        image_paths: list[str] | None = None,
        return_example: Any = None,
        max_try: int = 2,
        **_: Any,
    ) -> dict[str, Any]:
        if image_paths:
            user_content: Any = [{"type": "text", "text": prompt}]
            user_content.extend(_image_part(path) for path in image_paths)
        else:
            user_content = prompt

        messages = []
        if self.system_prompt:
            messages.append(
                {"role": "system", "content": self.system_prompt}
            )
        messages.append({"role": "user", "content": user_content})

        last_error: Exception | None = None

        for attempt in range(max(1, max_try)):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=self.max_completion_tokens,
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Judge returned empty content")

                result = _parse_json_object(content)

                if "score" not in result:
                    raise ValueError(
                        f"Judge JSON is missing score: {result!r}"
                    )

                return result

            except Exception as exc:
                last_error = exc
                if attempt + 1 < max(1, max_try):
                    time.sleep(10 * (attempt + 1))

        raise RuntimeError(
            f"Azure judge failed after {max_try} attempts"
        ) from last_error


def configure_researchclawbench_scorer(score_module: Any) -> None:
    """Install the Azure adapter without modifying benchmark source."""

    provider = os.environ.get("JUDGE_PROVIDER", "").strip().lower()
    if provider != "azure":
        return

    workers = int(os.environ.get("JUDGE_MAX_WORKERS", "1"))
    if workers != 1:
        raise ValueError(
            "Azure judge currently requires JUDGE_MAX_WORKERS=1"
        )

    # evaluation.score imported LLMAgent directly, so replace its module symbol.
    score_module.LLMAgent = AzureChatCompletionsJudge

    # Score serially to stay within the Azure TPM quota and let failures
    # propagate instead of silently converting a failed request to zero.
    def serial_map(inputs, function, **_):
        return [function(**item) for item in inputs]

    score_module.multi_thread = serial_map


def assert_complete_score(result: dict[str, Any]) -> None:
    """Reject benchmark results that contain an infrastructure failure.

    ResearchClawBench's default scorer catches provider exceptions and turns
    them into zero-score items.  A zero produced by a failed request is not a
    benchmark measurement, so make that distinction explicit at the RAC
    boundary.
    """

    failure_markers = (
        "scoring failed",
        "failed to parse scoring response",
        "resource not found",
    )
    failures: list[str] = []

    if result.get("error"):
        failures.append(f"top-level error: {result['error']}")
    items = result.get("items")
    if not isinstance(items, list) or not items:
        failures.append("score result has no checklist items")
        items = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            failures.append(f"item {index} is not an object")
            continue
        reasoning = str(item.get("reasoning", "")).strip()
        if not item.get("reasoning"):
            failures.append(f"item {index} has empty reasoning")
        elif any(marker in reasoning.lower() for marker in failure_markers):
            failures.append(f"item {index}: {reasoning[:160]}")
        if "score" not in item:
            failures.append(f"item {index} is missing score")

    if failures:
        raise RuntimeError("incomplete judge result: " + "; ".join(failures))
