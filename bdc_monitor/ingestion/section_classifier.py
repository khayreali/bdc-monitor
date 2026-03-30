import logging
import re

from bdc_monitor.domain import Section

log = logging.getLogger(__name__)

# ordered list — first match wins (checked against section title)
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"schedule[s]?\s+of\s+investments", re.I), "schedule_of_investments"),
    (re.compile(r"management.s\s+discussion", re.I), "mdna"),
    (re.compile(r"tender\s+offer|repurchase\s+offer|offer\s+to\s+purchase", re.I), "tender_offer"),
    (re.compile(r"notes\s+to\s+(?:condensed\s+)?(?:consolidated\s+)?financial", re.I), "notes"),
    (re.compile(r"risk\s+factor", re.I), "risk_factors"),
]

# BDC filings bury repurchase program info inside Item 5, Item 2, etc.
# as sub-headings rather than top-level sections. if a section title doesn't
# match anything above, check the text for these patterns.
_TENDER_TEXT_RE = re.compile(
    r"(?:share|stock)\s+repurchase\s+program|"
    r"discretionary\s+share\s+repurchase|"
    r"repurchase\s+offer[s]?(?:\s|$)",
    re.I,
)


class SectionClassifier:
    """Regex-based section labeling. Falls back to 'other' when nothing matches."""

    def classify(self, section: Section) -> str:
        title = section.title
        for pattern, section_type in _PATTERNS:
            if pattern.search(title):
                return section_type
        # title didn't match — check text for repurchase/tender sub-headings
        if _TENDER_TEXT_RE.search(section.text[:15000]):
            return "tender_offer"
        return "other"

    def classify_all(self, sections: list[Section]) -> list[Section]:
        for section in sections:
            section.section_type = self.classify(section)
        counts = {}
        for s in sections:
            counts[s.section_type] = counts.get(s.section_type, 0) + 1
        log.info(f"classified {len(sections)} sections: {counts}")
        return sections
