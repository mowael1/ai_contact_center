"""HTML-first section extraction.

Strategy, in priority order:

1. ``html_structure``  - walk the cleaned main-content subtree in DOM order,
   opening a new section at every ``h1``-``h4`` and attaching the content that
   follows it. Hierarchy is tracked with a heading stack so every section
   carries a full ``section_path``.
2. ``html_component``  - accordion / tab / FAQ panels whose label is a genuine
   content heading (see :mod:`rag.htmlx.components`).
3. ``text_fallback``   - deterministic heading heuristics over the selected
   textual representation, used only when the HTML yields nothing usable.
4. ``no_section``      - the whole document as one untitled section. A null
   title is always preferred over an invented one.
"""

from __future__ import annotations

from typing import Iterable, Optional

from bs4 import BeautifulSoup, NavigableString, Tag

from rag.config import settings
from rag.htmlx.cleaner import in_boilerplate, parse_html, select_main_content, strip_noise
from rag.htmlx.components import Component, find_components
from rag.logging_utils import get_logger
from rag.models import Section
from rag.text_utils import normalize_text

logger = get_logger(__name__)

HEADING_TAGS = ("h1", "h2", "h3", "h4")
# h5/h6 are used in this corpus mostly for styling, so they are content by
# default and only promoted when a document has no real headings at all.
EXTENDED_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")

CONTENT_TAGS = ("p", "li", "td", "th", "dd", "dt", "figcaption", "blockquote", "pre")
BLOCK_CONTAINERS = ("table", "ul", "ol", "dl")


def _heading_level(tag: Tag) -> int:
    return int(tag.name[1])


def _clean(text: str) -> str:
    return normalize_text(text)


def _text_of_block(tag: Tag) -> str:
    """Render a content block to plain text, keeping tables readable."""
    if tag.name in ("table",):
        rows = []
        for tr in tag.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)
    if tag.name in ("ul", "ol"):
        items = [li.get_text(" ", strip=True) for li in tag.find_all("li", recursive=False)]
        items = [i for i in items if i]
        return "\n".join(f"- {i}" for i in items)
    return tag.get_text(" ", strip=True)


def _iter_content_blocks(root: Tag) -> Iterable[Tag]:
    """Yield headings and content blocks in DOM order, without double-counting.

    A ``<li>`` inside a ``<ul>`` is emitted as part of the list, not separately.
    """
    for tag in root.find_all(EXTENDED_HEADING_TAGS + CONTENT_TAGS + BLOCK_CONTAINERS):
        if in_boilerplate(tag):
            continue
        if tag.name in ("li", "td", "th", "dd", "dt"):
            if tag.find_parent(BLOCK_CONTAINERS):
                continue
        if tag.name in CONTENT_TAGS and tag.find_parent(BLOCK_CONTAINERS):
            continue
        yield tag


class SectionExtractor:
    """Turns one document's HTML (or text) into ordered :class:`Section` objects."""

    def __init__(self, min_section_chars: int | None = None) -> None:
        self.min_section_chars = (
            min_section_chars
            if min_section_chars is not None
            else settings.MIN_SECTION_CHARS
        )

    # -- public API ---------------------------------------------------------
    def extract(self, html: Optional[str], fallback_text: str = "") -> list[Section]:
        sections: list[Section] = []
        if html and html.strip():
            try:
                sections = self._extract_from_html(html)
            except Exception as exc:  # malformed HTML must never kill ingestion
                logger.warning("HTML section extraction failed, falling back: %s", exc)
                sections = []
        if not sections and fallback_text.strip():
            sections = self._extract_from_text(fallback_text)
        if not sections and fallback_text.strip():
            sections = [
                Section(
                    section_index=0,
                    section_title=None,
                    section_path=None,
                    heading_level=None,
                    text=_clean(fallback_text),
                    extraction_method="no_section",
                )
            ]
        for i, section in enumerate(sections):
            section.section_index = i
        return sections

    # -- HTML ---------------------------------------------------------------
    def _extract_from_html(self, html: str) -> list[Section]:
        soup = strip_noise(parse_html(html))
        components = find_components(soup)
        component_panels = {id(c.panel): c for c in components}

        root = select_main_content(soup)
        sections = self._walk_headings(root, component_panels)

        # Components whose panel sits outside the chosen content root (common
        # for tab panels rendered in a sibling container) are added explicitly.
        covered = {id(c.panel) for c in components if self._is_inside(c.panel, root)}
        for comp in components:
            if id(comp.panel) in covered:
                continue
            text = _clean(comp.panel.get_text("\n", strip=True))
            if len(text) < self.min_section_chars:
                continue
            sections.append(
                Section(
                    section_index=0,
                    section_title=_clean(comp.label),
                    section_path=[_clean(comp.label)],
                    heading_level=None,
                    text=text,
                    extraction_method="html_component",
                    component_type=comp.component_type,
                    anchor_id=comp.anchor_id,
                )
            )
        return [s for s in sections if len(s.text) >= self.min_section_chars]

    @staticmethod
    def _is_inside(tag: Tag, root: Tag) -> bool:
        return tag is root or root in tag.parents

    def _walk_headings(
        self, root: Tag, component_panels: dict[int, Component]
    ) -> list[Section]:
        """Single DOM-order pass maintaining a heading stack for hierarchy."""
        blocks = list(_iter_content_blocks(root))
        heading_tags = set(HEADING_TAGS)
        if not any(b.name in heading_tags for b in blocks):
            # Document styles its headings as h5/h6 only - promote them.
            heading_tags = set(EXTENDED_HEADING_TAGS)

        sections: list[Section] = []
        stack: list[tuple[int, str]] = []  # (level, title)
        current: Optional[Section] = None
        buffer: list[str] = []
        # Content preceding the first heading belongs to no section.
        preamble: list[str] = []

        def flush() -> None:
            nonlocal current, buffer
            if current is not None:
                current.text = _clean("\n".join(buffer))
                if len(current.text) >= self.min_section_chars:
                    sections.append(current)
            current, buffer = None, []

        for block in blocks:
            if block.name in heading_tags:
                flush()
                title = _clean(block.get_text(" ", strip=True))
                if not title or len(title) > 250:
                    continue
                level = _heading_level(block)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                comp = self._component_for(block, component_panels)
                current = Section(
                    section_index=0,
                    section_title=title,
                    section_path=[t for _, t in stack],
                    heading_level=level,
                    text="",
                    extraction_method=(
                        "html_component" if comp is not None else "html_structure"
                    ),
                    component_type=comp.component_type if comp else None,
                    anchor_id=comp.anchor_id if comp else None,
                )
                buffer = []
                continue

            text = _text_of_block(block)
            if not text:
                continue
            if current is None:
                preamble.append(text)
            else:
                buffer.append(text)
        flush()

        if not sections and preamble:
            body = _clean("\n".join(preamble))
            if len(body) >= self.min_section_chars:
                sections.append(
                    Section(
                        section_index=0,
                        section_title=None,
                        section_path=None,
                        heading_level=None,
                        text=body,
                        extraction_method="no_section",
                    )
                )
        return sections

    @staticmethod
    def _component_for(
        heading: Tag, component_panels: dict[int, Component]
    ) -> Optional[Component]:
        for comp in component_panels.values():
            if comp.control is heading or heading in comp.control.descendants:
                return comp
        return None

    # -- text fallback ------------------------------------------------------
    def _extract_from_text(self, text: str) -> list[Section]:
        """Heading heuristics over plain text. Never invents a title.

        A line is treated as a heading only when it is short, has no terminal
        punctuation, is followed by real body text, and is not a bullet.
        """
        lines = [l.strip() for l in _clean(text).split("\n")]
        sections: list[Section] = []
        current_title: Optional[str] = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer, current_title
            body = _clean("\n".join(buffer))
            if len(body) >= self.min_section_chars:
                sections.append(
                    Section(
                        section_index=0,
                        section_title=current_title,
                        section_path=[current_title] if current_title else None,
                        heading_level=1 if current_title else None,
                        text=body,
                        extraction_method="text_fallback" if current_title else "no_section",
                    )
                )
            buffer = []

        for i, line in enumerate(lines):
            if self._is_text_heading(line, lines[i + 1 :]):
                flush()
                current_title = line.rstrip(":：").strip()
                continue
            if line:
                buffer.append(line)
        flush()
        return [s for s in sections if s.section_title]

    @staticmethod
    def _is_text_heading(line: str, following: list[str]) -> bool:
        if not (3 <= len(line) <= 80):
            return False
        if line[0] in "-•*0123456789":
            return False
        if line.rstrip().endswith((".", "!", "؟", "?", "،", ",", ";")):
            return False
        if len(line.split()) > 10:
            return False
        nxt = next((l for l in following if l), "")
        return len(nxt) >= 40
