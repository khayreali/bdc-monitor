import logging
import re
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from bdc_monitor.domain import Section

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

log = logging.getLogger(__name__)


class FilingParser(ABC):
    @abstractmethod
    def parse(self, filepath: Path, accession_number: str) -> list[Section]:
        ...

    def _extract_text(self, filepath: Path) -> str:
        html = filepath.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # normalize non-breaking spaces — EDGAR HTML is full of &nbsp;
        text = text.replace("\xa0", " ")
        # strip standalone page numbers (lines that are just 1-4 digits)
        text = re.sub(r"^\d{1,4}$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text


# use [ \t]+ (horizontal whitespace only) so patterns can't match across
# newlines — this is what keeps TOC entries from being picked up as headings,
# since the TOC splits "Item 1." and "Business" onto separate lines.
# [^\n]*$ allows headings with or without trailing text on the same line.
_ITEM_RE = re.compile(
    r"^Item[ \t]+\d+[A-Z]?\.[ \t]+[A-Za-z][^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_PART_RE = re.compile(
    r"^PART[ \t]+[IVX]+\.?[ \t]+[A-Z][^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_SOI_RE = re.compile(
    r"^Consolidated[ \t]+Schedule[s]?[ \t]+of[ \t]+Investments[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_NOTES_RE = re.compile(
    r"^Notes[ \t]+to[ \t]+(?:Condensed[ \t]+)?(?:Consolidated[ \t]+)?Financial[ \t]+Statements[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)


def _find_section_boundaries(text: str) -> list[tuple[str, int]]:
    """Find section heading positions in the extracted text.

    Returns (heading_text, char_offset) pairs sorted by position.
    Skips TOC entries, "Continued" page headers, and cross-references.
    """
    candidates = []

    for pattern in [_PART_RE, _ITEM_RE, _SOI_RE, _NOTES_RE]:
        for m in pattern.finditer(text):
            heading = m.group(0).strip()
            # skip page headers like "Schedule of Investments — Continued"
            if "continued" in heading.lower():
                continue
            # skip cross-references that use em/en dashes
            # e.g. "ITEM 1A. RISK FACTORS — Risks Related to..."
            if "—" in heading or "–" in heading:
                continue
            # skip overly long lines
            if len(heading) > 150:
                continue
            candidates.append((heading, m.start()))

    candidates.sort(key=lambda x: x[1])

    # dedup: if two headings have very similar prefixes (same first 40 chars),
    # keep only the first one. catches repeated "Consolidated Schedule of
    # Investments as of..." on every page.
    seen = set()
    deduped = []
    for heading, pos in candidates:
        prefix = heading[:40].lower()
        if prefix in seen:
            continue
        seen.add(prefix)
        deduped.append((heading, pos))

    return deduped


def _build_sections(
    text: str, boundaries: list[tuple[str, int]], accession: str
) -> list[Section]:
    sections = []
    for i, (title, start) in enumerate(boundaries):
        end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(text)
        body = text[start:end].strip()
        if len(body) < 100:
            continue
        sections.append(Section(
            filing_accession=accession,
            section_type="other",  # SectionClassifier handles this
            title=title[:200],
            text=body,
            start_idx=start,
            end_idx=end,
        ))
    return sections


class TenQParser(FilingParser):
    """Parser for 10-Q quarterly filings."""

    def parse(self, filepath: Path, accession_number: str) -> list[Section]:
        text = self._extract_text(filepath)

        # skip cover page + TOC. content starts at "PART I. FINANCIAL..." or
        # "PART I - FINANCIAL..." (some filers use a dash)
        content_start = re.search(
            r"^PART[ \t]+I[. \t-]*FINANCIAL", text, re.IGNORECASE | re.MULTILINE
        )
        if content_start:
            text = text[content_start.start():]
        else:
            fallback = re.search(
                r"^Item[ \t]+1\.[ \t]+Financial",
                text,
                re.IGNORECASE | re.MULTILINE,
            )
            if fallback:
                text = text[fallback.start():]

        boundaries = _find_section_boundaries(text)
        sections = _build_sections(text, boundaries, accession_number)

        log.info(f"parsed 10-Q {accession_number}: {len(sections)} sections")
        return sections


class TenKParser(FilingParser):
    """Parser for 10-K annual filings.

    Same general structure as 10-Q but with different item numbering
    (Item 1 = Business, Item 1A = Risk Factors, Item 7 = MD&A, etc.)
    """

    def parse(self, filepath: Path, accession_number: str) -> list[Section]:
        text = self._extract_text(filepath)

        # 10-K has "PART I" alone (no "FINANCIAL INFORMATION"), so use
        # the first Item heading as content start instead
        content_start = re.search(
            r"^Item[ \t]+1\.[ \t]+", text, re.IGNORECASE | re.MULTILINE
        )
        if content_start:
            text = text[content_start.start():]

        boundaries = _find_section_boundaries(text)
        sections = _build_sections(text, boundaries, accession_number)

        log.info(f"parsed 10-K {accession_number}: {len(sections)} sections")
        return sections


class ShareholderLetterParser(FilingParser):
    """Parser for shareholder letters.

    These are much less structured than 10-Q/10-K — usually
    just prose with a few headings. We split on blank-line-separated
    paragraphs that look like headers (short, possibly bold/caps).
    """

    _HEADING_RE = re.compile(
        r"^[A-Z][A-Za-z\s,&:]{5,80}$", re.MULTILINE
    )

    def parse(self, filepath: Path, accession_number: str) -> list[Section]:
        text = self._extract_text(filepath)

        boundaries = []
        for m in self._HEADING_RE.finditer(text):
            heading = m.group(0).strip()
            # skip lines that are all caps and very short (likely labels)
            if len(heading) < 10:
                continue
            boundaries.append((heading, m.start()))

        if not boundaries:
            # no clear headings — return the whole thing as one section
            return [Section(
                filing_accession=accession_number,
                section_type="other",
                title="Shareholder Letter",
                text=text,
                start_idx=0,
                end_idx=len(text),
            )]

        sections = _build_sections(text, boundaries, accession_number)
        log.info(
            f"parsed shareholder letter {accession_number}: {len(sections)} sections"
        )
        return sections


def get_parser(filing_type: str) -> FilingParser:
    """Return the right parser for a filing type."""
    parsers = {
        "10-Q": TenQParser(),
        "10-K": TenKParser(),
        "shareholder_letter": ShareholderLetterParser(),
    }
    parser = parsers.get(filing_type)
    if not parser:
        raise ValueError(f"no parser for filing type: {filing_type}")
    return parser
