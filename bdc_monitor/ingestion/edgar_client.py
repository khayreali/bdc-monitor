import asyncio
import logging
import time
from datetime import date
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from bdc_monitor.config import Settings
from bdc_monitor.domain import Filing

log = logging.getLogger(__name__)


class EdgarClient:
    """Pulls filings from SEC EDGAR. Handles rate limits and disk caching."""

    SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
    ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
    COMPANY_SEARCH = "https://www.sec.gov/cgi-bin/browse-edgar"
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    # SEC says max 10 req/s — stay a bit under
    _MIN_INTERVAL = 0.12  # ~8 req/s to stay safe

    def __init__(self, settings: Settings):
        self.settings = settings
        self.filings_dir = settings.filings_dir
        self.filings_dir.mkdir(parents=True, exist_ok=True)
        self._last_req = 0.0
        self._client: httpx.AsyncClient | None = None
        self._tickers: dict | None = None
        self._cik_cache: dict[str, str] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": self.settings.edgar_user_agent},
                follow_redirects=True,
                timeout=30.0,
            )
        return self._client

    async def _get(self, url: str, **kwargs) -> httpx.Response:
        """Rate-limited GET request."""
        now = time.monotonic()
        wait = self._MIN_INTERVAL - (now - self._last_req)
        if wait > 0:
            await asyncio.sleep(wait)

        client = self._get_client()
        self._last_req = time.monotonic()
        resp = await client.get(url, **kwargs)
        resp.raise_for_status()
        return resp

    async def resolve_cik(self, company_name: str, ticker: str = "") -> str:
        """Look up CIK number for a company. Tries ticker JSON first, falls back to EDGAR search."""
        if company_name in self._cik_cache:
            return self._cik_cache[company_name]

        cik = await self._try_tickers_json(ticker, company_name)
        if not cik:
            cik = await self._search_edgar_html(company_name)
        if not cik:
            raise ValueError(f"couldn't resolve CIK for '{company_name}' (ticker={ticker})")

        self._cik_cache[company_name] = cik
        log.info(f"resolved {company_name} -> CIK {cik}")
        return cik

    async def _try_tickers_json(self, ticker: str, name: str) -> str | None:
        """Check SEC's company_tickers.json — works for exchange-listed companies."""
        if self._tickers is None:
            resp = await self._get(self.TICKERS_URL)
            self._tickers = resp.json()

        # exact ticker match
        if ticker:
            for entry in self._tickers.values():
                if entry["ticker"].upper() == ticker.upper():
                    return str(entry["cik_str"])

        # name substring match (case-insensitive)
        name_lower = name.lower()
        for entry in self._tickers.values():
            title = entry["title"].lower()
            if name_lower in title or title in name_lower:
                return str(entry["cik_str"])

        return None

    async def _search_edgar_html(self, company_name: str) -> str | None:
        """Fall back to EDGAR company search page and parse the HTML.

        Two cases: single match (redirects to company page) or
        multiple matches (shows a results table).
        """
        params = {
            "company": company_name,
            "CIK": "",
            "type": "10-K",
            "dateb": "",
            "owner": "include",
            "count": "10",
            "search_text": "",
            "action": "getcompany",
        }
        resp = await self._get(self.COMPANY_SEARCH, params=params)
        soup = BeautifulSoup(resp.text, "lxml")

        # single match — landed on the company page
        company_span = soup.find("span", class_="companyName")
        if company_span:
            cik_link = company_span.find("a")
            if cik_link:
                digits = "".join(c for c in cik_link.text if c.isdigit())
                if digits:
                    return digits.lstrip("0") or "0"

        # multiple matches — results table
        table = soup.find("table", class_="tableFile2")
        if table:
            for row in table.find_all("tr")[1:]:
                cells = row.find_all("td")
                if cells:
                    text = cells[0].get_text(strip=True)
                    # CIKs are all digits, form types contain letters
                    if text.isdigit():
                        return text.lstrip("0") or "0"

        return None

    async def list_filings(
        self, cik: str, filing_types: list[str], since: date, ticker: str
    ) -> list[Filing]:
        """List filings from the submissions API, filtered by type and date."""
        padded = cik.zfill(10)
        url = f"{self.SUBMISSIONS_BASE}/CIK{padded}.json"
        resp = await self._get(url)
        data = resp.json()

        company_name = data.get("name", "")
        recent = data["filings"]["recent"]

        filings = []
        for i in range(len(recent["accessionNumber"])):
            form = recent["form"][i]
            if form not in filing_types:
                continue

            filed = date.fromisoformat(recent["filingDate"][i])
            if filed < since:
                continue

            accession = recent["accessionNumber"][i]
            report_str = recent["reportDate"][i]
            period_end = date.fromisoformat(report_str) if report_str else filed

            primary_doc = recent["primaryDocument"][i]
            accession_nd = accession.replace("-", "")

            filings.append(Filing(
                accession_number=accession,
                cik=cik,
                ticker=ticker,
                company_name=company_name,
                filing_type=form,
                period_end=period_end,
                filed_date=filed,
                url=f"{self.ARCHIVES_BASE}/{cik}/{accession_nd}/{primary_doc}",
            ))

        log.info(f"found {len(filings)} {filing_types} filings for {ticker} since {since}")
        return filings

    async def download_filing(self, filing: Filing) -> Path:
        """Download a filing's primary document. Skips if already on disk."""
        accession_nd = filing.accession_number.replace("-", "")
        dest_dir = self.filings_dir / filing.ticker / accession_nd
        dest_dir.mkdir(parents=True, exist_ok=True)

        filename = filing.url.rsplit("/", 1)[-1]
        filepath = dest_dir / filename

        if filepath.exists():
            log.info(f"cached: {filepath}")
            return filepath

        resp = await self._get(filing.url)
        filepath.write_bytes(resp.content)
        log.info(f"downloaded {filing.filing_type} -> {filepath}")
        return filepath
