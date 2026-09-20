"""Deterministic boilerplate removal for the Vodafone Egypt scrape.

The selectors below were derived by surveying the real ``raw_html`` in
``rag/data/*.parquet`` (Liferay portal + a bespoke component library), not
guessed. They are kept as module-level tuples so they are easy to tune as the
dataset evolves.
"""

from __future__ import annotations

import re
from typing import Iterable

from bs4 import BeautifulSoup, Tag

# Tags that never carry readable page content.
DROP_TAGS = (
    "script", "style", "noscript", "template", "svg", "iframe", "object",
    "embed", "canvas", "link", "meta", "form", "input", "select", "textarea",
)

# Whole site-wide regions.
DROP_TAG_NAMES = ("nav", "footer", "header", "aside")

# Class/id substrings observed as site chrome in this dataset.
BOILERPLATE_CLASS_PATTERNS = (
    "navigation", "nav-item", "nav-chevron", "navbar", "megamenu", "mega-menu",
    "footer", "cookie", "consent", "breadcrumb", "skip-link", "search-overlay",
    "language-switcher", "switcher-link", "social-links", "back-to-top",
    "portlet-topper", "control-menu", "banner-ad", "advertisement",
    "js-navigation", "site-header", "site-footer", "modal-backdrop",
    # Owl Carousel duplicates every slide into ".cloned" nodes, which would
    # otherwise multiply the same copy 3-5x inside one document.
    "cloned",
    # Cross-site personalisation widgets repeated on dozens of pages.
    "save-articles", "user-preferences", "newsletter-signup",
)

BOILERPLATE_ROLES = ("navigation", "banner", "contentinfo", "search", "menubar", "menu")

# Containers that, in this corpus, reliably wrap the editorial body.
MAIN_CONTENT_SELECTORS = (
    "main",
    "[role=main]",
    "#main-content",
    "#content",
    ".journal-content-article",
    ".portlet-content-container",
    "article",
)

_CLASS_RE = re.compile("|".join(re.escape(p) for p in BOILERPLATE_CLASS_PATTERNS), re.I)


def _attr_blob(tag: Tag) -> str:
    classes = " ".join(tag.get("class") or [])
    return f"{classes} {tag.get('id') or ''}"


def is_boilerplate(tag: Tag) -> bool:
    """True when a tag looks like site-wide chrome rather than page content."""
    if tag.name in DROP_TAG_NAMES:
        return True
    if (tag.get("role") or "").lower() in BOILERPLATE_ROLES:
        return True
    if tag.get("aria-hidden") == "true":
        return True
    return bool(_CLASS_RE.search(_attr_blob(tag)))


def in_boilerplate(tag: Tag) -> bool:
    """True when the tag itself or any ancestor is site chrome."""
    node: Tag | None = tag
    while node is not None and getattr(node, "name", None):
        if is_boilerplate(node):
            return True
        node = node.parent  # type: ignore[assignment]
    return False


def parse_html(html: str) -> BeautifulSoup:
    """Parse with lxml, falling back to the stdlib parser on malformed input."""
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:  # pragma: no cover - lxml is very tolerant
        return BeautifulSoup(html, "html.parser")


def strip_noise(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove non-content tags and site-wide chrome, in place."""
    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()
    for tag in soup.find_all(DROP_TAG_NAMES):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        if not tag.parent:
            continue
        if is_boilerplate(tag):
            tag.decompose()
    return soup


def select_main_content(soup: BeautifulSoup) -> Tag:
    """Pick the densest plausible content root, else fall back to <body>.

    "Densest" is measured in text length so that an empty ``<main>`` shell does
    not win over the Liferay article that actually holds the copy.
    """
    body = soup.body or soup
    best: Tag = body
    best_len = len(body.get_text(" ", strip=True))

    candidates: list[Tag] = []
    for selector in MAIN_CONTENT_SELECTORS:
        candidates.extend(soup.select(selector))

    # Multiple journal-content-article blocks are common: keep them all by
    # preferring their lowest common ancestor rather than only the first.
    if len(candidates) > 1:
        merged = _lowest_common_ancestor(candidates)
        if merged is not None:
            candidates.append(merged)

    for candidate in candidates:
        length = len(candidate.get_text(" ", strip=True))
        # Require a real share of the body text before overriding it.
        if length > best_len * 0.35 and length > 200:
            if length > len(best.get_text(" ", strip=True)) or best is body:
                best = candidate
                best_len = length
    return best


def _lowest_common_ancestor(tags: Iterable[Tag]) -> Tag | None:
    tags = list(tags)
    if not tags:
        return None
    chains = []
    for tag in tags:
        chain = []
        node = tag
        while node is not None and getattr(node, "name", None):
            chain.append(node)
            node = node.parent
        chains.append(list(reversed(chain)))
    common: Tag | None = None
    for nodes in zip(*chains):
        first = nodes[0]
        if all(n is first for n in nodes):
            common = first
        else:
            break
    return common
