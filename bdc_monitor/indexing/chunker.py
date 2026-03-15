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


class StructureAwareChunker(Chunker):
    """Splits on paragraph boundaries instead of mid-sentence."""

    def __init__(self, max_tokens: int = 500, overlap_paragraphs: int = 1):
        self.max_tokens = max_tokens
        self.overlap = overlap_paragraphs

    def chunk(self, text, filing_accession, section_id, ticker,
              period_end, filing_type, section_type):
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []

        chunks = []
        current: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = _token_len(para)

            # single paragraph bigger than max — split it on sentences
            if para_tokens > self.max_tokens:
                # flush what we have first
                if current:
                    chunks.append(_make_chunk(
                        "\n\n".join(current), filing_accession, section_id,
                        len(chunks), ticker, period_end, filing_type, section_type,
                    ))
                    current, current_tokens = [], 0

                for sub in self._split_long_block(para):
                    chunks.append(_make_chunk(
                        sub, filing_accession, section_id, len(chunks),
                        ticker, period_end, filing_type, section_type,
                    ))
                continue

            if current_tokens + para_tokens > self.max_tokens and current:
                chunks.append(_make_chunk(
                    "\n\n".join(current), filing_accession, section_id,
                    len(chunks), ticker, period_end, filing_type, section_type,
                ))
                # overlap: keep last N paragraphs
                current = current[-self.overlap:] if self.overlap else []
                current_tokens = sum(_token_len(p) for p in current)

            current.append(para)
            current_tokens += para_tokens

        if current:
            chunks.append(_make_chunk(
                "\n\n".join(current), filing_accession, section_id,
                len(chunks), ticker, period_end, filing_type, section_type,
            ))

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        raw = re.split(r"\n\s*\n", text)
        paras = [p.strip() for p in raw if p.strip()]
        return paras

    def _split_long_block(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # if regex didn't actually split (no sentence boundaries), do token split
        if len(sentences) <= 1:
            return self._token_split(text)

        parts = []
        current = []
        current_tokens = 0
        for sent in sentences:
            st = _token_len(sent)
            # single sentence bigger than max — token split it
            if st > self.max_tokens:
                if current:
                    parts.append(" ".join(current))
                    current, current_tokens = [], 0
                parts.extend(self._token_split(sent))
                continue
            if current_tokens + st > self.max_tokens and current:
                parts.append(" ".join(current))
                current, current_tokens = [], 0
            current.append(sent)
            current_tokens += st
        if current:
            parts.append(" ".join(current))
        return parts

    def _token_split(self, text: str) -> list[str]:
        tokens = _ENC.encode(text)
        parts = []
        stride = self.max_tokens - 50
        pos = 0
        while pos < len(tokens):
            parts.append(_ENC.decode(tokens[pos:pos + self.max_tokens]))
            pos += stride
        return parts


class SectionAwareChunker(Chunker):
    """Splits SOI on page headers, falls back to StructureAwareChunker for everything else."""

    def __init__(self, max_tokens: int = 500):
        self.max_tokens = max_tokens
        self._fallback = StructureAwareChunker(max_tokens=max_tokens)
        # page headers that repeat in SOI
        self._page_re = re.compile(
            r"^Consolidated\s+Schedule[s]?\s+of\s+Investments",
            re.IGNORECASE | re.MULTILINE,
        )

    def chunk(self, text, filing_accession, section_id, ticker,
              period_end, filing_type, section_type):
        # only apply SOI logic to schedule_of_investments sections
        if section_type != "schedule_of_investments":
            return self._fallback.chunk(
                text, filing_accession, section_id, ticker,
                period_end, filing_type, section_type,
            )

        return self._chunk_soi(
            text, filing_accession, section_id, ticker,
            period_end, filing_type, section_type,
        )

    def _chunk_soi(self, text, filing_accession, section_id, ticker,
                   period_end, filing_type, section_type):
        # split on page header repetitions
        pages = self._page_re.split(text)
        # first element is text before the first header (might be empty)
        # rest are page contents

        row_groups = []
        for page in pages:
            page = page.strip()
            if not page or _token_len(page) < 20:
                continue
            row_groups.append(page)

        if not row_groups:
            return self._fallback.chunk(
                text, filing_accession, section_id, ticker,
                period_end, filing_type, section_type,
            )

        # merge small pages, split large ones
        chunks = []
        current_parts: list[str] = []
        current_tokens = 0

        for group in row_groups:
            group_tokens = _token_len(group)

            if group_tokens > self.max_tokens:
                # flush current
                if current_parts:
                    chunks.append(_make_chunk(
                        "\n".join(current_parts), filing_accession, section_id,
                        len(chunks), ticker, period_end, filing_type, section_type,
                    ))
                    current_parts, current_tokens = [], 0

                # split this oversized page by line groups
                for sub in self._split_by_lines(group):
                    chunks.append(_make_chunk(
                        sub, filing_accession, section_id, len(chunks),
                        ticker, period_end, filing_type, section_type,
                    ))
                continue

            if current_tokens + group_tokens > self.max_tokens and current_parts:
                chunks.append(_make_chunk(
                    "\n".join(current_parts), filing_accession, section_id,
                    len(chunks), ticker, period_end, filing_type, section_type,
                ))
                current_parts, current_tokens = [], 0

            current_parts.append(group)
            current_tokens += group_tokens

        if current_parts:
            chunks.append(_make_chunk(
                "\n".join(current_parts), filing_accession, section_id,
                len(chunks), ticker, period_end, filing_type, section_type,
            ))

        return chunks

    def _split_by_lines(self, text: str) -> list[str]:
        lines = text.split("\n")
        parts = []
        current = []
        current_tokens = 0

        for line in lines:
            lt = _token_len(line)
            if current_tokens + lt > self.max_tokens and current:
                parts.append("\n".join(current))
                current, current_tokens = [], 0
            current.append(line)
            current_tokens += lt

        if current:
            parts.append("\n".join(current))
        return parts
