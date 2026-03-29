from datetime import date

from bdc_monitor.domain import Chunk, RetrievedChunk
from bdc_monitor.generation.generator import Generator
from bdc_monitor.generation.llm_client import LLMClient
from bdc_monitor.retrieval.context_assembler import ContextAssembler


def _rc(cid, text="some text"):
    chunk = Chunk(
        chunk_id=cid, filing_accession="acc", ticker="T",
        period_end=date(2025, 1, 1), filing_type="10-Q",
        section_type="mdna", text=text, token_count=5, chunk_index=0,
    )
    return RetrievedChunk(chunk=chunk, score=0.8, rank=1, source="dense")


class FakeLLM(LLMClient):
    def __init__(self, response):
        self._response = response

    def generate(self, system, user, max_tokens=2048):
        return self._response


def test_hallucinated_citation_gets_dropped():
    """If the LLM cites a chunk_id that wasn't in the retrieved set,
    the generator should drop it from the citations list."""
    chunks = [_rc("real_chunk_1"), _rc("real_chunk_2")]

    # LLM response references a real chunk and a fake one
    llm = FakeLLM(
        "The NAV was $15 [CITE: real_chunk_1]. "
        "Leverage was 1.2x [CITE: nonexistent_chunk]. "
        "Risk factors noted [CITE: real_chunk_2]."
    )
    gen = Generator(llm, ContextAssembler())
    answer = gen.generate("what is the NAV?", chunks)

    cited_ids = [c.chunk_id for c in answer.citations]
    assert "real_chunk_1" in cited_ids
    assert "real_chunk_2" in cited_ids
    assert "nonexistent_chunk" not in cited_ids
    assert len(answer.citations) == 2


def test_duplicate_citations_are_deduplicated():
    """If the LLM cites the same chunk_id twice, it should appear
    only once in the citations list."""
    chunks = [_rc("chunk_a")]

    llm = FakeLLM(
        "Fact one [CITE: chunk_a]. Also fact two [CITE: chunk_a]."
    )
    gen = Generator(llm, ContextAssembler())
    answer = gen.generate("question", chunks)

    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "chunk_a"


def test_no_citations_returns_empty_list():
    """An LLM response with no [CITE:] markers should produce an
    answer with an empty citations list, not an error."""
    chunks = [_rc("chunk_a")]
    llm = FakeLLM("I don't have enough information to answer this question.")
    gen = Generator(llm, ContextAssembler())
    answer = gen.generate("question", chunks)

    assert answer.citations == []
    assert answer.answer_text == "I don't have enough information to answer this question."
