#!/usr/bin/env python3
"""Fail when built documentation exposes unprocessed math or Markdown tables."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from html import unescape
from pathlib import Path


IGNORED_START = re.compile(
    r"<(?P<tag>pre|code|script|style|textarea)\b[^>]*>|"
    r"<(?P<math_tag>span|div)\b[^>]*class=[\"'][^\"']*\barithmatex\b[^\"']*[\"'][^>]*>",
    re.IGNORECASE,
)
HTML_TAG = re.compile(r"<[^>]+>")
RAW_MATH = re.compile(
    r"\$\$|(?<!\\)\$(?!\{)|\\(?:\[|\]|\(|\))|"
    r"\\(?:begin|end|frac|dfrac|tfrac|mathrm|mathbf|mathbb|mathsf|mathcal|"
    r"operatorname|text|left|right|llbracket|rrbracket|Psi|Gamma|Delta|Sigma|"
    r"theta|tau|alpha|beta|varepsilon|epsilon|widetilde|det|log|exp|sum|prod|"
    r"int|mapsto|longrightarrow|rightarrow|to|circ|mid|equiv|sim|cdot|dots|"
    r"quad|bigl|bigr|lvert|rvert)\b"
)
RAW_TABLE_RULE = re.compile(r"(?m)^\s*\|?\s*:?-{3,}:?\s*\|")


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    kind: str
    excerpt: str


IgnoreState = tuple[str, bool]


def visible_segments(
    line: str, ignored: IgnoreState | None
) -> tuple[list[str], IgnoreState | None, bool]:
    segments: list[str] = []
    nested_display_delimiter = False
    cursor = 0
    while cursor < len(line):
        if ignored is not None:
            ignored_tag, is_math = ignored
            closing_token = f"</{ignored_tag.lower()}>"
            closing_start = line.lower().find(closing_token, cursor)
            if closing_start < 0:
                if is_math and "$$" in line[cursor:]:
                    nested_display_delimiter = True
                return segments, ignored, nested_display_delimiter
            if is_math and "$$" in line[cursor:closing_start]:
                nested_display_delimiter = True
            cursor = closing_start + len(closing_token)
            ignored = None
            continue

        opening = IGNORED_START.search(line, cursor)
        if opening is None:
            segments.append(line[cursor:])
            break

        segments.append(line[cursor : opening.start()])
        ignored = (
            opening.group("tag") or opening.group("math_tag"),
            opening.group("math_tag") is not None,
        )
        cursor = opening.end()

    return segments, ignored, nested_display_delimiter


def audit_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    ignored: IgnoreState | None = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            segments, ignored, nested_display_delimiter = visible_segments(line, ignored)
            if nested_display_delimiter:
                findings.append(
                    Finding(path, line_number, "nested math delimiter", "$$ inside rendered math")
                )
            visible_html = " ".join(segments)
            if not any(marker in visible_html for marker in ("$", "\\", "---")):
                continue
            text = unescape(HTML_TAG.sub("", visible_html))
            for kind, pattern in (("raw math", RAW_MATH), ("raw table rule", RAW_TABLE_RULE)):
                match = pattern.search(text)
                if match is not None:
                    excerpt = " ".join(text[match.start() : match.start() + 100].split())
                    findings.append(Finding(path, line_number, kind, excerpt))
    return findings


def audit(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(root.rglob("*.html")):
        findings.extend(audit_file(path))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("site_dir", nargs="?", type=Path, default=Path("site"))
    args = parser.parse_args()

    findings = audit(args.site_dir)
    if not findings:
        print(f"Rendered math audit passed for {args.site_dir}")
        return 0

    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.kind}: {finding.excerpt}")
    print(f"Found {len(findings)} possible rendering error(s).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
