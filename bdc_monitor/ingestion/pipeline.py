import logging
from datetime import date

from bdc_monitor.config import BDCS, FILING_TYPES
from bdc_monitor.ingestion.edgar_client import EdgarClient

log = logging.getLogger(__name__)


class IngestionPipeline:
    """Downloads filings from EDGAR and stores metadata."""

    def __init__(self, client: EdgarClient):
        self.client = client

    async def run(self, tickers: list[str], since: date):
        for ticker in tickers:
            company = BDCS.get(ticker)
            if not company:
                log.warning(f"unknown ticker: {ticker}, skipping")
                continue

            try:
                cik = await self.client.resolve_cik(company, ticker)
            except ValueError as e:
                log.error(f"skipping {ticker}: {e}")
                continue

            filings = await self.client.list_filings(cik, FILING_TYPES, since, ticker)

            new = 0
            for filing in filings:
                path = await self.client.download_filing(filing)
                filing.local_path = str(path)
                new += 1

            log.info(f"{ticker}: downloaded {new} filings")
