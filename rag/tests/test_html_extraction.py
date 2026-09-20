"""HTML section extraction: headings, hierarchy, components, malformed input."""

import pytest

from rag.htmlx.cleaner import parse_html, select_main_content, strip_noise
from rag.htmlx.components import find_components
from rag.htmlx.section_extractor import SectionExtractor

LONG = "This is a sentence with enough characters to clear the minimum section length. " * 2


@pytest.fixture
def extractor():
    return SectionExtractor(min_section_chars=30)


def test_extracts_headings_and_nested_hierarchy(extractor):
    html = f"""
    <html><body><main>
      <h1>Internet</h1><p>{LONG}</p>
      <h2>Internet Packages</h2><p>{LONG}</p>
      <h3>Renewal</h3><p>{LONG}</p>
    </main></body></html>
    """
    sections = extractor.extract(html)
    paths = [s.section_path for s in sections]
    assert ["Internet"] in paths
    assert ["Internet", "Internet Packages"] in paths
    assert ["Internet", "Internet Packages", "Renewal"] in paths

    renewal = next(s for s in sections if s.section_title == "Renewal")
    assert renewal.heading_level == 3
    assert renewal.extraction_method == "html_structure"


def test_sibling_heading_pops_the_stack(extractor):
    """An h2 after a deeper h3 must not inherit that h3 as a parent."""
    html = f"""
    <main>
      <h1>Root</h1><p>{LONG}</p>
      <h2>A</h2><p>{LONG}</p>
      <h3>A1</h3><p>{LONG}</p>
      <h2>B</h2><p>{LONG}</p>
    </main>
    """
    sections = extractor.extract(html)
    b = next(s for s in sections if s.section_title == "B")
    assert b.section_path == ["Root", "B"]


def test_preserves_dom_order(extractor):
    html = f"<main><h2>First</h2><p>{LONG}</p><h2>Second</h2><p>{LONG}</p></main>"
    titles = [s.section_title for s in extractor.extract(html)]
    assert titles == ["First", "Second"]


def test_navigation_and_footer_are_dropped(extractor):
    html = f"""
    <body>
      <nav><h2>Shop</h2><p>{LONG}</p></nav>
      <div class="navigation__item"><h2>Menu</h2><p>{LONG}</p></div>
      <main><h2>Real Content</h2><p>{LONG}</p></main>
      <footer><h2>Footer Links</h2><p>{LONG}</p></footer>
    </body>
    """
    titles = [s.section_title for s in extractor.extract(html)]
    assert titles == ["Real Content"]


def test_accordion_becomes_a_component_section(extractor):
    """Mirrors the real markup: h3 control + aria-controls panel."""
    html = f"""
    <main>
      <h2>More information</h2><p>{LONG}</p>
      <div class="card accordion-item js-accordion-item">
        <div class="card-header">
          <h3 class="accordion-title js-collapser" id="heading-0"
              aria-controls="collapse-0" aria-expanded="true"
              data-toggle="collapse" role="button">
            <span class="accordion-title-heading">Renew Your Bundle</span>
          </h3>
        </div>
        <div id="collapse-0" aria-labelledby="heading-0" class="collapse show">
          <div class="card-body"><p>You can renew your bundle via *880#. {LONG}</p></div>
        </div>
      </div>
    </main>
    """
    sections = extractor.extract(html)
    renew = next(s for s in sections if s.section_title == "Renew Your Bundle")
    assert renew.extraction_method == "html_component"
    assert renew.component_type == "accordion"
    assert "*880#" in renew.text


def test_faq_question_heading_keeps_its_answer(extractor):
    html = f"""
    <main>
      <div class="accordion">
        <h3 class="accordion-title" id="q1" aria-controls="a1" data-toggle="collapse">
          How do I renew my package?
        </h3>
        <div id="a1" aria-labelledby="q1"><p>You can renew it monthly. {LONG}</p></div>
      </div>
    </main>
    """
    sections = extractor.extract(html)
    faq = next(s for s in sections if "renew" in (s.section_title or "").lower())
    assert "monthly" in faq.text


def test_navigation_accordion_is_not_a_section():
    """The corpus' nav menus use the same accordion classes - they must not win."""
    html = """
    <nav class="navigation">
      <a class="js-accordion-heading js-navigation-link navigation__link"
         role="tab" aria-expanded="false" aria-controls="nav-1">Shop</a>
      <div id="nav-1" class="js-accordion-content navigation__list">
        <a href="/x">Buy Plans</a><a href="/y">Phones &amp; Devices</a>
      </div>
    </nav>
    """
    soup = parse_html(html)
    assert find_components(soup) == []


def test_plain_button_is_not_a_section():
    html = """
    <main>
      <button class="btn btn-secondary">Subscribe now</button>
      <button aria-controls="missing-id">Broken target</button>
      <a href="#top">Back to top</a>
    </main>
    """
    assert find_components(parse_html(html)) == []


def test_tabpanel_is_detected_as_tab_component():
    html = f"""
    <main>
      <div id="tab-1" role="tab" aria-controls="panel-1">Fakka Cards</div>
      <div id="panel-1" role="tabpanel" aria-labelledby="tab-1">
        <p>To recharge Fakka card press *858#. {LONG}</p>
      </div>
    </main>
    """
    components = find_components(parse_html(html))
    assert [c.component_type for c in components] == ["tab"]
    assert components[0].label == "Fakka Cards"


def test_malformed_html_does_not_raise(extractor):
    broken = f"<html><body><main><h1>Title<p>{LONG}<div><h2>Sub</h2><p>{LONG}"
    sections = extractor.extract(broken)
    assert sections
    assert all(s.text for s in sections)


def test_empty_html_falls_back_to_text(extractor):
    sections = extractor.extract("", fallback_text=LONG)
    assert len(sections) == 1
    assert sections[0].extraction_method == "no_section"
    assert sections[0].section_title is None
    assert sections[0].section_path is None


def test_never_invents_a_title_when_no_heading_exists(extractor):
    html = f"<main><p>{LONG}</p></main>"
    sections = extractor.extract(html)
    assert all(s.section_title is None for s in sections)
    assert all(s.extraction_method == "no_section" for s in sections)


def test_tables_and_lists_are_kept_readable(extractor):
    html = f"""
    <main><h2>Pricing</h2>
      <table><tr><th>Bundle</th><th>Price</th></tr>
             <tr><td>Flex 70</td><td>70 EGP</td></tr></table>
      <ul><li>First item here</li><li>Second item here</li></ul>
      <p>{LONG}</p>
    </main>
    """
    section = extractor.extract(html)[0]
    assert "Flex 70 | 70 EGP" in section.text
    assert "- First item here" in section.text


def test_h5_promoted_only_when_no_real_headings(extractor):
    html = f"<main><h5>Styled Heading</h5><p>{LONG}</p></main>"
    sections = extractor.extract(html)
    assert sections[0].section_title == "Styled Heading"


def test_strip_noise_removes_scripts_and_styles():
    soup = strip_noise(parse_html("<body><script>x=1</script><style>a{}</style><p>keep</p></body>"))
    assert "x=1" not in soup.get_text()
    assert "keep" in soup.get_text()


def test_select_main_content_prefers_dense_article():
    html = f"<body><main></main><div class='journal-content-article'><p>{LONG}</p></div></body>"
    root = select_main_content(strip_noise(parse_html(html)))
    assert LONG.strip()[:40] in root.get_text(" ", strip=True)
