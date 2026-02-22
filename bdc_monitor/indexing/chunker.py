import re
from abc import ABC, abstractmethod
from datetime import date

import tiktoken

from bdc_monitor.domain import Chunk

# text-embedding-3-small uses cl100k_base
_ENC = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_ENC.encode(text))


def _make_chunk(
    text: str, filing_accession: str, section_id: int, idx: int,
    ticker: str, period_end: date, filing_type: str, section_type: str,
) -> Chunk:
    tokens = _ENC.encode(text)
    return Chunk(
        chunk_id=f"{filing_accession}_{section_id}_{idx}",
        filing_accession=filing_accession,
        ticker=ticker,
        period_end=period_end,
        filing_type=filing_type,
        section_type=section_type,
        text=text,
        token_count=len(tokens),
        chunk_index=idx,
    )


class Chunker(ABC):
    @abstractmethod
    def chunk(
        self,
        text: str,
        filing_accession: str,
        section_id: int,
        ticker: str,
        period_end: date,
        filing_type: str,
        section_type: str,
    ) -> list[Chunk]:
        ...


class FixedSizeChunker(Chunker):
    """Fixed-size token windows with overlap."""

    def __init__(self, max_tokens: int = 500, overlap: int = 50):
        self.max_tokens = max_tokens
        self.stride = max_tokens - overlap

    def chunk(self, text, filing_accession, section_id, ticker,
              period_end, filing_type, section_type):
        tokens = _ENC.encode(text)
        if not tokens:
            return []

        chunks = []
        pos = 0
        while pos < len(tokens):
            window = tokens[pos : pos + self.max_tokens]
            chunks.append(Chunk(
                chunk_id=f"{filing_accession}_{section_id}_{len(chunks)}",
                filing_accession=filing_accession,
                ticker=ticker,
                period_end=period_end,
                filing_type=filing_type,
                section_type=section_type,
                text=_ENC.decode(window),
                token_count=len(window),
                chunk_index=len(chunks),
            ))
            pos += self.stride

        return chunks
