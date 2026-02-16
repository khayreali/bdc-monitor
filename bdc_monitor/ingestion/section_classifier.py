import logging
import re

from bdc_monitor.domain import Section

log = logging.getLogger(__name__)

# ordered list — first match wins
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"schedule[s]?\s+of\s+investments", re.I), "schedule_of_investments"),
    (re.compile(r"management.s\s+discussion", re.I), "mdna"),
    (re.compile(r"tender\s+offer|repurchase\s+offer|offer\s+to\s+purchase", re.I), "tender_offer"),
    (re.compile(r"notes\s+to\s+(?:condensed\s+)?(?:consolidated\s+)?financial", re.I), "notes"),
    (re.compile(r"risk\s+factor", re.I), "risk_factors"),
]


class SectionClassifier:
    """Regex-based section labeling. Falls back to 'other' when nothing matches."""

    def classify(self, section: Section) -> str:
        title = section.title
        for pattern, section_type in _PATTERNS:
            if pattern.search(title):
                return section_type
        return "other"

    def classify_all(self, sections: list[Section]) -> list[Section]:
        for section in sections:
            section.section_type = self.classify(section)
        counts = {}
        for s in sections:
            counts[s.section_type] = counts.get(s.section_type, 0) + 1
        log.info(f"classified {len(sections)} sections: {counts}")
        return sections
