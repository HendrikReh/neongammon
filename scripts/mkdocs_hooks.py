"""MkDocs build hook for the GitHub Pages documentation site.

Pages under docs/ link to repository files outside docs/ (for example
``../cpp/src/multipy.cpp`` or ``../CLAUDE.md#core-domain-invariants``). Those
relative links work on GitHub but not on the generated site, so this hook
rewrites them to absolute GitHub URLs on the ``main`` branch. Links inside
docs/ and fenced code blocks are left unchanged.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

GITHUB_BASE = "https://github.com/HendrikReh/neongammon"
REF = "main"

_FENCE = re.compile(r"^(```|~~~)")
# Inline Markdown link or image target: ](target) or ](target "title").
_LINK = re.compile(r"\]\((?P<target>[^)\s]+)(?P<title>\s+\"[^\"]*\")?\)")


def _rewrite_target(target: str, page_dir: str, repo_root: Path) -> str:
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE) or target.startswith(("#", "/")):
        return target
    path, sep, fragment = target.partition("#")
    resolved = posixpath.normpath(posixpath.join("docs", page_dir, path))
    if resolved == "docs" or resolved.startswith("docs/") or resolved.startswith(".."):
        return target
    kind = "tree" if (repo_root / resolved).is_dir() else "blob"
    return f"{GITHUB_BASE}/{kind}/{REF}/{resolved}{sep}{fragment}"


def on_page_markdown(markdown: str, page, config, files) -> str:
    repo_root = Path(config["docs_dir"]).resolve().parent
    page_dir = posixpath.dirname(page.file.src_uri)
    out: list[str] = []
    in_fence = False
    for line in markdown.splitlines(keepends=True):
        if _FENCE.match(line.lstrip()):
            in_fence = not in_fence
        if not in_fence:
            line = _LINK.sub(
                lambda m: "]("
                + _rewrite_target(m.group("target"), page_dir, repo_root)
                + (m.group("title") or "")
                + ")",
                line,
            )
        out.append(line)
    return "".join(out)
