from __future__ import annotations

import hashlib
import re

from .policy import ISSUE_TAGS
from .schemas import Issue


CATEGORY_TERMS = {
    "citation": ("citation", "bibliograph", "reference"),
    "figure": ("figure", "caption", "diagram", "plot"),
    "experiment": ("experiment", "seed", "confidence interval", "ablation", "result"),
    "baseline_fairness": ("baseline", "oracle", "unfair", "information advantage"),
    "methodology": ("method", "definition", "objective", "protocol"),
    "writing": ("clarity", "writing", "typo", "presentation"),
    "execution": ("error", "failed", "crash", "compile"),
}
ISSUE_LINE = re.compile(r"\b(major|minor|issue|missing|lacks?|incorrect|inconsistent|unresolved|weak|failure|blocker|unfair)\b", re.I)


def extract_review_issues(text: str) -> list[Issue]:
    lines = [line.strip() for line in text.splitlines() if ISSUE_LINE.search(line)]
    joined = "\n".join(lines).lower()
    issues: list[Issue] = []
    for category, terms in CATEGORY_TERMS.items():
        evidence = tuple(line for line in lines if any(term in line.lower() for term in terms))
        if not evidence:
            continue
        normalized = re.sub(r"\s+", " ", evidence[0].lower())[:300]
        issue_id = f"review:{category}:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:10]}"
        issues.append(
            Issue(
                issue_id=issue_id,
                kind=category,
                summary=evidence[0][:500],
                evidence=evidence[:5],
                required_tags=ISSUE_TAGS[category],
            )
        )
    return issues


def parse_review_score(text: str) -> float | None:
    patterns = (
        r"Overall\s+Score\s*[：:]\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"score\s*[：:]\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"Total(?:\s+Score)?\s*[：:=]\s*\**(\d+(?:\.\d+)?)\s*/\s*10",
        r"\|\s*\**Total\**\s*\|\s*\**(\d+(?:\.\d+)?)\s*/\s*10",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return float(match.group(1))
    return None
