import logging
from datetime import date
from pathlib import Path

from bdc_monitor.config import BDCS, FILING_TYPES
from bdc_monitor.indexing.metadata_store import MetadataStore
from bdc_monitor.ingestion.edgar_client import EdgarClient
from bdc_monitor.ingestion.parsers import get_parser
from bdc_monitor.ingestion.section_classifier import SectionClassifier

log = logging.getLogger(__name__)


class IngestionPipeline:
    """Downloads filings from EDGAR, parses them into sections, and stores everything."""

    def __init__(self, client: EdgarClient, store: MetadataStore):
        self.client = client
        self.store = store
        self.classifier = SectionClassifier()

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
                if self.store.has_filing(filing.accession_number):
                    # already downloaded — but maybe not parsed yet
                    if not self.store.is_parsed(filing.accession_number):
                        local = filing.local_path
                        if not local:
                            # look it up from the db
                            continue
                        self._parse_filing(
                            Path(local), filing.filing_type, filing.accession_number
                        )
                    continue

                path = await self.client.download_filing(filing)
                filing.local_path = str(path)
                self.store.insert_filing(filing)
                self._parse_filing(path, filing.filing_type, filing.accession_number)
                new += 1

            total = self.store.count_filings(ticker)
            sections = self.store.count_sections(ticker)
            log.info(f"{ticker}: {new} new filings, {total} total, {sections} sections")

    def _parse_filing(self, path: Path, filing_type: str, accession: str):
        try:
            parser = get_parser(filing_type)
            sections = parser.parse(path, accession)
            sections = self.classifier.classify_all(sections)
            self.store.insert_sections(sections)
            self.store.mark_parsed(accession)
        except Exception:
            log.exception(f"failed to parse {accession} ({filing_type})")
