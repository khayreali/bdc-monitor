from datetime import date
from pydantic import BaseModel


class Filing(BaseModel):
    accession_number: str
    cik: str
    ticker: str
    company_name: str
    filing_type: str  # 10-Q, 10-K, shareholder_letter
    period_end: date
    filed_date: date
    url: str
    local_path: str | None = None


class Section(BaseModel):
    filing_accession: str
    section_type: str  # schedule_of_investments, mdna, tender_offer, notes, risk_factors, other
    title: str
    text: str
    start_idx: int
    end_idx: int


class Chunk(BaseModel):
    chunk_id: str
    filing_accession: str
    ticker: str
    period_end: date
    filing_type: str
    section_type: str
    text: str
    token_count: int
    chunk_index: int  # position within the section


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    source: str  # "dense", "sparse", or "hybrid"


class Citation(BaseModel):
    chunk_id: str
    text_span: str
    filing_accession: str
    ticker: str
    period_end: date
    section_type: str


class Answer(BaseModel):
    question: str
    answer_text: str
    citations: list[Citation]
    chunks_used: list[RetrievedChunk]
    metadata: dict = {}


class EvalQuestion(BaseModel):
    question: str
    expected_facts: list[str]
    relevant_filings: list[str]  # accession numbers
    relevant_section_types: list[str]
    difficulty: str  # easy, medium, hard


class EvalResult(BaseModel):
    question: EvalQuestion
    answer: Answer
    retrieval_recall: float
    citation_precision: float
    fact_recall: float
    config_name: str = ""
