import json
import logging
import re

from bdc_monitor.config import BDCS
from bdc_monitor.generation.llm_client import LLMClient

log = logging.getLogger(__name__)

# name fragments that uniquely identify each BDC — ordered longest first
# so "blue owl technology" matches before "blue owl"
_NAME_PATTERNS = [
    ("blue owl credit", "OBDC"),
    ("blue owl capital", "OBDC"),
    ("blue owl technology", "OBDT"),
    ("blackstone private", "BCRED"),
    ("ares capital", "ARCC"),
    ("prospect capital", "PSEC"),
    ("apollo debt", "APOLLO"),
    ("fs kkr", "FSK"),
    ("kkr fs", "KKRFS"),
    # "blue owl" alone is ambiguous (OBDC vs OBDT), let the LLM handle it
]

_TICKERS_STR = ", ".join(BDCS.keys())
_COMPANIES_STR = "\n".join(f"  {t}: {n}" for t, n in BDCS.items())

_SYSTEM = f"""\
Extract structured filters from a question about BDC (Business Development Company) SEC filings.

Known BDCs:
{_COMPANIES_STR}

Return a JSON object with these fields:
- "tickers": list of ticker symbols from the question (empty list if none or "all")
- "period_end": specific quarter-end date as "YYYY-MM-DD" if mentioned (Q1=03-31, Q2=06-30, Q3=09-30, Q4=12-31), or null
- "section_types": list of relevant section types from [schedule_of_investments, mdna, tender_offer, notes, risk_factors], or empty list if unclear

Return ONLY valid JSON, nothing else."""


class QueryRouter:

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def route(self, question: str) -> dict | None:
        # try regex first for common patterns — saves an LLM call
        filters = self._regex_route(question)
        if filters:
            log.info(f"query router (regex): {filters}")
            return filters

        try:
            raw = self.llm.generate(_SYSTEM, question, max_tokens=200)
            # strip markdown fences if the LLM wraps the JSON
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            parsed = json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            log.warning(f"query router failed to parse LLM response: {e}")
            return None

        return self._build_where(parsed)

    def _regex_route(self, question: str) -> dict | None:
        q = question.lower()

        # find tickers mentioned directly
        tickers = [t for t in BDCS if t.lower() in q]
        # also check company name fragments
        for pattern, ticker in _NAME_PATTERNS:
            if pattern in q and ticker not in tickers:
                tickers.append(ticker)

        if not tickers:
            return None

        where = {}
        if len(tickers) == 1:
            where["ticker"] = tickers[0]
        else:
            where["ticker"] = {"$in": tickers}

        # try to extract quarter
        m = re.search(r"Q([1-4])\s*(\d{4})", question, re.I)
        if m:
            quarter, year = int(m.group(1)), m.group(2)
            end_dates = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
            where["period_end"] = f"{year}-{end_dates[quarter]}"

        if len(where) > 1:
            where = {"$and": [{k: v} for k, v in where.items()]}

        log.info(f"query router (regex): {where}")
        return where

    def _build_where(self, parsed: dict) -> dict | None:
        parts = []

        tickers = parsed.get("tickers", [])
        if tickers:
            if len(tickers) == 1:
                parts.append({"ticker": tickers[0]})
            else:
                parts.append({"ticker": {"$in": tickers}})

        period = parsed.get("period_end")
        if period:
            parts.append({"period_end": period})

        section_types = parsed.get("section_types", [])
        if section_types:
            if len(section_types) == 1:
                parts.append({"section_type": section_types[0]})
            else:
                parts.append({"section_type": {"$in": section_types}})

        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return {"$and": parts}
