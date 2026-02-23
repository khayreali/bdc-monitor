import sqlite3
from pathlib import Path

from bdc_monitor.domain import Chunk, Filing, Section


class MetadataStore:
    """SQLite store for filing and chunk metadata."""

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

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                filing_accession TEXT NOT NULL REFERENCES filings(accession_number),
                section_id INTEGER NOT NULL REFERENCES sections(id),
                ticker TEXT NOT NULL,
                period_end TEXT NOT NULL,
                filing_type TEXT NOT NULL,
                section_type TEXT NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL
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

    def get_unindexed_sections(self) -> list[tuple]:
        """Returns sections from filings that have no chunks yet.

        Each row: (section_id, filing_accession, ticker, period_end,
                    filing_type, section_type, text)
        """
        return self.conn.execute("""
            SELECT s.id, s.filing_accession, f.ticker, f.period_end,
                   f.filing_type, s.section_type, s.text
            FROM sections s
            JOIN filings f ON s.filing_accession = f.accession_number
            WHERE f.accession_number NOT IN (
                SELECT DISTINCT filing_accession FROM chunks
            )
            ORDER BY f.ticker, f.period_end, s.id
        """).fetchall()

    def insert_chunks(self, chunks: list[Chunk]):
        self.conn.executemany(
            """INSERT OR IGNORE INTO chunks
               (chunk_id, filing_accession, section_id, ticker, period_end,
                filing_type, section_type, text, token_count, chunk_index)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (c.chunk_id, c.filing_accession, int(c.chunk_id.split("_")[-2]),
                 c.ticker, str(c.period_end), c.filing_type, c.section_type,
                 c.text, c.token_count, c.chunk_index)
                for c in chunks
            ],
        )
        self.conn.commit()

    def count_chunks(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self):
        self.conn.close()
