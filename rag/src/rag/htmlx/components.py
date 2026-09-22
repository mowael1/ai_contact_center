"""Deterministic detection of accordion / tab / FAQ components.

Survey of the real corpus (see README "Dataset findings") showed two very
different populations of "expandable" widgets:

1. **Navigation menus** - ``<a class="js-accordion-heading js-navigation-link"
   role="tab" aria-expanded="false">Shop</a>``. These are site chrome and must
   never become sections.
2. **Editorial FAQ accordions** - ``<h3 class="accordion-title js-collapser"
   aria-controls="collapse-0" data-toggle="collapse">`` paired with
   ``<div id="collapse-0" aria-labelledby="heading-0">``.

The discriminator is therefore *not* "is it a button" but a scored combination
of: a resolvable controlled region, that region holding real prose, and the
control not living inside navigation chrome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup, Tag

from rag.htmlx.cleaner import in_boilerplate

# Classes seen on genuine content accordions in this corpus.
CONTENT_ACCORDION_HINTS = (
    "accordion-title", "js-collapser", "card-collapsing", "accordion-item",
    "js-accordion-item", "faq", "collapsible",
)
NAV_HINTS = ("navigation", "nav-item", "menu", "switcher", "footer")

CONTROL_ATTRS = ("aria-controls", "data-target", "data-bs-target", "href")

MIN_PANEL_CHARS = 60


@dataclass(slots=True)
class Component:
    """A heading control paired with the content region it expands."""

    label: str
    control: Tag
    panel: Tag
    component_type: str  # accordion | tab
    anchor_id: Optional[str]
    confidence: float


def _resolve_panel(soup: BeautifulSoup, control: Tag) -> tuple[Optional[Tag], Optional[str]]:
    """Follow aria-controls / data-target / href="#id" to the controlled region."""
    for attr in CONTROL_ATTRS:
        raw = control.get(attr)
        if not raw:
            continue
        if isinstance(raw, list):
            raw = raw[0]
        target_id = str(raw).strip()
        if attr == "href":
            if not target_id.startswith("#") or len(target_id) < 2:
                continue
        target_id = target_id.lstrip("#")
        if not target_id:
            continue
        panel = soup.find(id=target_id)
        if isinstance(panel, Tag) and panel is not control and control not in panel.parents:
            return panel, target_id
    return None, None


def _label_of(control: Tag) -> str:
    """Prefer the dedicated label span over the whole control (drops chevrons)."""
    for cls in ("accordion-title-heading", "card-title", "tab-label"):
        span = control.find(class_=cls)
        if isinstance(span, Tag):
            text = span.get_text(" ", strip=True)
            if text:
                return text
    return control.get_text(" ", strip=True)


def _looks_navigational(tag: Tag) -> bool:
    blob = " ".join(tag.get("class") or []) + " " + (tag.get("id") or "")
    return any(h in blob.lower() for h in NAV_HINTS)


def find_components(soup: BeautifulSoup) -> list[Component]:
    """Return content-bearing accordion/tab components in DOM order.

    Each candidate must clear every hard gate before it is scored, so an
    ordinary button or in-page anchor link cannot become a section.
    """
    components: list[Component] = []
    seen_panels: set[int] = set()

    candidates = soup.find_all(
        lambda t: isinstance(t, Tag)
        and (
            t.has_attr("aria-controls")
            or t.has_attr("data-target")
            or t.has_attr("data-bs-target")
            or (t.get("role") in ("tab", "button") and t.has_attr("href"))
        )
    )

    for control in candidates:
        # --- hard gates -------------------------------------------------
        if in_boilerplate(control) or _looks_navigational(control):
            continue
        label = _label_of(control)
        if not label or len(label) < 3 or len(label) > 200:
            continue
        panel, anchor = _resolve_panel(soup, control)
        if panel is None or id(panel) in seen_panels:
            continue
        panel_text = panel.get_text(" ", strip=True)
        if len(panel_text) < MIN_PANEL_CHARS:
            continue
        if label in panel_text[: len(label) + 5] and len(panel_text) < len(label) * 2:
            continue  # panel just echoes its own label

        # --- scoring ----------------------------------------------------
        score = 0.4
        blob = " ".join(control.get("class") or []).lower()
        if any(h in blob for h in CONTENT_ACCORDION_HINTS):
            score += 0.3
        if control.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            score += 0.2
        if control.has_attr("aria-controls"):
            score += 0.1
        if control.has_attr("aria-expanded"):
            score += 0.05
        if panel.get("aria-labelledby") and control.get("id") == panel.get("aria-labelledby"):
            score += 0.15
        if panel.get("role") == "tabpanel" or control.get("role") == "tab":
            score += 0.05
        if len(panel_text) > 200:
            score += 0.05

        if score < 0.6:
            continue

        component_type = (
            "tab"
            if (panel.get("role") == "tabpanel" or control.get("role") == "tab")
            else "accordion"
        )
        seen_panels.add(id(panel))
        components.append(
            Component(
                label=label,
                control=control,
                panel=panel,
                component_type=component_type,
                anchor_id=anchor,
                confidence=round(min(score, 1.0), 3),
            )
        )
    return components
