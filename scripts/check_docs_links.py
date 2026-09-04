"""Check local link targets in tracked root Markdown files and docs/.

Run from a Git checkout with EvalView's dependencies installed (Rich depends on
markdown-it-py). This uses a Markdown parser so code examples are not mistaken
for links. External URLs and heading anchors are deliberately not validated.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt


@dataclass(frozen=True)
class BrokenLink:
    source: Path
    line: int
    target: str

    def __str__(self) -> str:
        return f"{self.source}:{self.line}: missing local target: {self.target}"


class _HTMLLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attribute = {"a": "href", "img": "src"}.get(tag)
        if attribute:
            for name, value in attrs:
                if name == attribute and value:
                    self.links.append((self.getpos()[0], value))


def markdown_links(text: str) -> Iterator[tuple[int, str]]:
    """Yield rendered Markdown/HTML links and images with source line numbers."""
    for token in MarkdownIt().parse(text):
        line = token.map[0] + 1 if token.map else 1
        for child in [token, *(token.children or [])]:
            attribute = {"link_open": "href", "image": "src"}.get(child.type)
            if attribute:
                target = child.attrGet(attribute)
                if target:
                    yield line, target
            elif child.type in {"html_inline", "html_block"}:
                parser = _HTMLLinks()
                parser.feed(child.content)
                for offset, target in parser.links:
                    yield line + offset - 1, target


def tracked_docs(root: Path) -> list[Path]:
    """Exclude untracked drafts, generated reports, and non-documentation trees."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = [Path(name) for name in result.stdout.split("\0") if name]
    return [path for path in paths if len(path.parts) == 1 or path.parts[0] == "docs"]


def check_links(root: Path, sources: Iterable[Path]) -> list[BrokenLink]:
    root = root.resolve()
    broken = []
    for source in sources:
        for line, destination in markdown_links((root / source).read_text(encoding="utf-8")):
            url = urlsplit(destination)
            if url.scheme or url.netloc or not url.path:
                continue
            path = unquote(url.path)
            target = (
                root / path.lstrip("/") if path.startswith("/") else root / source.parent / path
            )
            if not target.resolve().is_relative_to(root) or not target.exists():
                broken.append(BrokenLink(source, line, destination))
    return broken


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sources = tracked_docs(root)
    broken = check_links(root, sources)
    for failure in broken:
        print(failure)
    if broken:
        print(f"Found {len(broken)} broken local links in {len(sources)} Markdown files.")
        return 1
    print(f"Local documentation links OK ({len(sources)} tracked Markdown files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
