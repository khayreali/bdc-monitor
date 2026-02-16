import sqlite3
from pathlib import Path

from bdc_monitor.domain import Filing, Section


class MetadataStore:

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS filings (
                accession_number TEXT PRIMARY KEY,
                cik TEXT NOT NULL,
                ticker TEXT NOT NULL,
                company_name TEXT NOT NULL,
                filing_type TEXT NOT NULL,
                period_end TEXT NOT NULL,
                filed_date TEXT NOT NULL,
                url TEXT NOT NULL,
                local_path TEXT,
                parsed INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filing_accession TEXT NOT NULL REFERENCES filings(accession_number),
                section_type TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                start_idx INTEGER NOT NULL,
                end_idx INTEGER NOT NULL,
                UNIQUE(filing_accession, start_idx)
            );
        """)

    def has_filing(self, accession_number: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM filings WHERE accession_number = ?",
            (accession_number,),
        ).fetchone()
        return row is not None

    def insert_filing(self, filing: Filing):
        self.conn.execute(
            """INSERT OR IGNORE INTO filings
               (accession_number, cik, ticker, company_name, filing_type,
                period_end, filed_date, url, local_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                filing.accession_number, filing.cik, filing.ticker,
                filing.company_name, filing.filing_type,
                str(filing.period_end), str(filing.filed_date),
                filing.url, filing.local_path,
            ),
        )
        self.conn.commit()

    def mark_parsed(self, accession_number: str):
        self.conn.execute(
            "UPDATE filings SET parsed = 1 WHERE accession_number = ?",
            (accession_number,),
        )
        self.conn.commit()

    def is_parsed(self, accession_number: str) -> bool:
        row = self.conn.execute(
            "SELECT parsed FROM filings WHERE accession_number = ?",
            (accession_number,),
        ).fetchone()
        return bool(row and row[0])

    def insert_sections(self, sections: list[Section]):
        self.conn.executemany(
            """INSERT OR IGNORE INTO sections
               (filing_accession, section_type, title, text, start_idx, end_idx)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (s.filing_accession, s.section_type, s.title,
                 s.text, s.start_idx, s.end_idx)
                for s in sections
            ],
        )
        self.conn.commit()

    def count_filings(self, ticker: str | None = None) -> int:
        if ticker:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM filings WHERE ticker = ?", (ticker,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM filings").fetchone()
        return row[0]

    def count_sections(self, ticker: str | None = None) -> int:
        if ticker:
            row = self.conn.execute(
                """SELECT COUNT(*) FROM sections s
                   JOIN filings f ON s.filing_accession = f.accession_number
                   WHERE f.ticker = ?""",
                (ticker,),
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM sections").fetchone()
        return row[0]

    def close(self):
        self.conn.close()
