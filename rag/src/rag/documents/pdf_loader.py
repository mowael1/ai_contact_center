"""PDF text and structure extraction with PyMuPDF.

PyMuPDF is used rather than a plain text extractor because heading detection
needs per-span font size and weight - a PDF has no tags, so the only structural
signal is typography - and because page numbers are required for citations
("المصدر: ... صفحة 12").
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from rag.documents.base import HeadingCandidate, LoadedDocument, PageText, infer_title
from rag.logging_utils import get_logger

logger = get_logger(__name__)

# "1. ", "1.2 ", "1.2.3 " and the Arabic equivalents. Limit numeric
# heading components so phone numbers such as "19888 - Banque Misr" do not
# become headings and cause the contact details to disappear from their section.
NUMBERED = re.compile(r"^\s*(?:\d{1,3}(?:\.\d{1,3}){0,2}[.)]?|[IVXivx]+[.)]|[أ-ي][-.)])\s+\S")
ARABIC_HEADING_WORD = re.compile(r"^\s*(?:الفصل|القسم|الباب|المقدمة|الخاتمة|ملحق)\b")

MAX_HEADING_CHARS = 140


def _table_regions(page) -> list[tuple]:
    """Bounding boxes of tables on the page, plus their rendered rows.

    Tables must be handled separately for two reasons found while testing a
    real help-centre PDF: their header cells are bold and short, so they get
    mistaken for headings, and reading them line-by-line separates each label
    from its value ("Banque Misr" and "19888" end up on different lines).
    """
    regions: list[tuple] = []
    try:
        finder = page.find_tables()
    except Exception:
        return regions
    # Lines outside a table's detected bbox can still be table columns. This
    # happens with RTL PDFs where PyMuPDF detects only the two right-hand
    # columns and leaves a third, left-aligned value column in the prose stream.
    layout_lines: list[tuple[tuple, str]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if text:
                layout_lines.append((tuple(line.get("bbox", (0, 0, 0, 0))), text))

    for table in getattr(finder, "tables", []) or []:
        try:
            rows = table.extract()
        except Exception:
            continue
        rendered = []
        expanded = list(table.bbox)
        row_layout = getattr(table, "rows", [])
        from rag.documents.arabic_repair import _repair_table_cell
        from rag.text_utils import arabic_ratio

        candidates_by_row: list[list[tuple[str, tuple, str]]] = []
        for index in range(len(rows)):
            matches: list[tuple[str, tuple, str]] = []
            if index < len(row_layout):
                _, y0, _, y1 = row_layout[index].bbox
                for line_bbox, line_text in layout_lines:
                    x0, ly0, x1, ly1 = line_bbox
                    cy = (ly0 + ly1) / 2
                    if len(line_text) > 100 or not y0 <= cy <= y1:
                        continue
                    if x1 <= table.bbox[0] + 2:
                        matches.append(("left", line_bbox, line_text))
                    elif x0 >= table.bbox[2] - 2:
                        matches.append(("right", line_bbox, line_text))
            candidates_by_row.append(matches)
        minimum_rows = max(2, (len(rows) + 1) // 2)
        supported_sides = {
            side for side in ("left", "right")
            if sum(any(candidate[0] == side for candidate in row) for row in candidates_by_row)
            >= minimum_rows
        }

        for index, row in enumerate(rows):
            cells = [(c or "").strip().replace("\n", " ") for c in row]
            if index < len(row_layout):
                side_cells = [
                    candidate for candidate in candidates_by_row[index]
                    if candidate[0] in supported_sides
                ]
                # A third column outside the detected table is common in RTL
                # comparison tables. Include it in the same row and suppress
                # it from the prose stream below.
                left = [text for side, _, text in side_cells if side == "left"]
                right = [text for side, _, text in side_cells if side == "right"]
                is_rtl = arabic_ratio(" ".join(left + cells + right)) >= 0.45
                if is_rtl:
                    # PyMuPDF's table-cell extraction can reverse Arabic
                    # characters inside each cell. Nearby third-column values
                    # come from the regular text layer, so they need word-order
                    # repair instead.
                    cells = [
                        cell[::-1] if arabic_ratio(cell) >= 0.65 else _repair_table_cell(cell)
                        for cell in cells
                    ]
                    left = [_repair_table_cell(text) for text in left]
                    right = [_repair_table_cell(text) for text in right]
                    # Put an RTL row in reader order (rightmost cell first).
                    cells = list(reversed(left + cells + right))
                else:
                    cells = left + cells + right
                for _, box, _ in side_cells:
                    expanded[0] = min(expanded[0], box[0])
                    expanded[1] = min(expanded[1], box[1])
                    expanded[2] = max(expanded[2], box[2])
                    expanded[3] = max(expanded[3], box[3])
            if any(cells):
                rendered.append(" | ".join(cells))
        if rendered:
            regions.append((tuple(expanded), "\n".join(rendered)))
    return regions


def _inside(bbox, region) -> bool:
    """True when a line's box lies mostly within a table's box."""
    x0, y0, x1, y1 = bbox
    rx0, ry0, rx1, ry1 = region
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return rx0 <= cx <= rx1 and ry0 <= cy <= ry1


def _spans(page) -> tuple[list[dict], list[str]]:
    """Flatten a page into text lines with size/weight, tables kept separate.

    Returns (prose lines, rendered tables). Lines that fall inside a table are
    dropped from the prose stream so they cannot become heading candidates.
    """
    tables = _table_regions(page)
    boxes = [bbox for bbox, _ in tables]

    out: list[dict] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:      # 0 = text
            continue
        for line in block.get("lines", []):
            text = "".join(s.get("text", "") for s in line.get("spans", []))
            if not text.strip():
                continue
            bbox = line.get("bbox", (0, 0, 0, 0))
            if any(_inside(bbox, box) for box in boxes):
                continue
            sizes = [s.get("size", 0) for s in line.get("spans", []) if s.get("text", "").strip()]
            flags = [s.get("flags", 0) for s in line.get("spans", []) if s.get("text", "").strip()]
            out.append({
                "text": text.strip(),
                "size": round(max(sizes), 1) if sizes else 0.0,
                # PyMuPDF flag bit 4 (value 16) marks bold.
                "bold": any(f & 16 for f in flags),
            })
    return out, [rendered for _, rendered in tables]


def load_pdf(path: str | Path) -> LoadedDocument:
    """Extract text per page plus typographic heading candidates."""
    import pymupdf

    path = Path(path)
    doc = pymupdf.open(path)
    pages: list[PageText] = []
    all_lines: list[tuple[int, dict]] = []

    try:
        for index in range(doc.page_count):
            page = doc[index]
            lines, tables = _spans(page)
            all_lines.extend((index + 1, ln) for ln in lines)
            body = "\n".join(ln["text"] for ln in lines).strip()
            # Tables are appended as pipe-separated rows so each label stays
            # on the same line as its value.
            if tables:
                body = "\n\n".join(filter(None, [body, *tables]))
            pages.append(PageText(page_number=index + 1, text=body))
    finally:
        doc.close()

    _strip_running_headers(pages)
    # Heading candidates are recomputed against the surviving lines so a
    # stripped footer can never be treated as a heading.
    all_lines = [
        (page.page_number, line)
        for page in pages
        for line in _relines(page, all_lines)
    ]
    headings = detect_headings(all_lines)
    for page in pages:
        page.text = _normalize_pdf_formatting(page.text)
    first = pages[0].text if pages else ""
    return LoadedDocument(
        title=infer_title(path, first),
        source=path.name,
        path=str(path.resolve()),
        source_type="pdf",
        pages=pages,
        headings=headings,
        byte_size=path.stat().st_size if path.exists() else 0,
    )


_LIST_MARKER = re.compile(r"^\s*(\d{1,3})[.)]\s*$")


def _normalize_pdf_formatting(text: str) -> str:
    """Repair simple layout artifacts while retaining source wording."""
    lines = text.splitlines()
    normalized: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        marker = _LIST_MARKER.match(line)
        if marker and index + 1 < len(lines) and lines[index + 1].strip():
            line = f"{marker.group(1)}. {lines[index + 1].strip()}"
            index += 1
        # PDF text extraction sometimes loses spaces around an inline dash or
        # keeps a dangling dash before a table's pipe separator.
        line = re.sub(r"\)([-–])\s*(?=\w)", r") \1 ", line)
        line = re.sub(r"\s*[-–]\s*\|", " |", line)
        normalized.append(line)
        index += 1
    return "\n".join(normalized)


def _relines(page: PageText, all_lines: list[tuple[int, dict]]) -> list[dict]:
    """The original span records for the lines that survived stripping."""
    kept = set(page.text.split("\n"))
    return [ln for pn, ln in all_lines if pn == page.page_number and ln["text"] in kept]


def _strip_running_headers(pages: list[PageText], min_pages: int = 3) -> None:
    """Remove lines that repeat on most pages - page headers and footers.

    A help-centre export repeats its source URL at the foot of every page; left
    in, that boilerplate lands inside chunks and dilutes their embeddings.
    Only applied when there are enough pages for repetition to be meaningful.
    """
    if len(pages) < min_pages:
        return
    from collections import Counter

    counts: Counter = Counter()
    for page in pages:
        for line in set(page.text.split("\n")):
            stripped = line.strip()
            if stripped:
                counts[stripped] += 1

    threshold = max(2, int(len(pages) * 0.6))
    boilerplate = {
        line for line, n in counts.items()
        if n >= threshold and len(line) < 200
    }
    if not boilerplate:
        return
    for page in pages:
        page.text = "\n".join(
            line for line in page.text.split("\n")
            if line.strip() not in boilerplate
        ).strip()


def detect_headings(lines: list[tuple[int, dict]]) -> list[HeadingCandidate]:
    """Find headings from typography and numbering. No model involved.

    Body size is the *modal* font size of the document, so a document that is
    entirely 18pt does not have every line treated as a heading.
    """
    if not lines:
        return []

    size_counts = Counter(ln["size"] for _, ln in lines if ln["size"])
    if not size_counts:
        return []
    body_size = size_counts.most_common(1)[0][0]

    candidates: list[HeadingCandidate] = []
    for order, (page_number, line) in enumerate(lines):
        text = line["text"]
        if not (3 <= len(text) <= MAX_HEADING_CHARS):
            continue
        size, bold = line["size"], line["bold"]

        reason = None
        if size >= body_size * 1.15:
            reason = "font_size"
        elif bold and size >= body_size and len(text) <= 90:
            reason = "bold"
        elif NUMBERED.match(text) or ARABIC_HEADING_WORD.match(text):
            reason = "numbering"
        if reason is None:
            continue
        # A "heading" that ends in a sentence terminator is usually body text.
        if text.rstrip().endswith(('.', '،', '؛')) and reason != "numbering":
            continue
        candidates.append(
            HeadingCandidate(text=text, page_number=page_number, level=0,
                             order=order, reason=reason)
        )

    _assign_levels(candidates, lines)
    return candidates


def _assign_levels(candidates: list[HeadingCandidate], lines: list[tuple[int, dict]]) -> None:
    """Map distinct heading font sizes to levels 1..N, largest first."""
    if not candidates:
        return
    by_order = {order: line for order, (_, line) in enumerate(lines)}
    sizes = sorted({by_order[c.order]["size"] for c in candidates}, reverse=True)
    level_of = {size: i + 1 for i, size in enumerate(sizes)}
    for c in candidates:
        c.level = level_of.get(by_order[c.order]["size"], 1)
