from datetime import date

from bdc_monitor.domain import Chunk, RetrievedChunk
from bdc_monitor.retrieval.retrievers import HybridRetriever, Retriever


def _chunk(cid, text="placeholder"):
    return Chunk(
        chunk_id=cid, filing_accession="acc", ticker="T",
        period_end=date(2025, 1, 1), filing_type="10-Q",
        section_type="mdna", text=text, token_count=5, chunk_index=0,
    )


class FakeDense(Retriever):
    def __init__(self, results):
        self._results = results

    def retrieve(self, query, top_k=None, where=None):
        return self._results


class FakeSparse(Retriever):
    def __init__(self, results):
        self._results = results

    def retrieve(self, query, top_k=None, where=None):
        return self._results


def test_rrf_ranks_shared_chunk_highest():
    """A chunk appearing in both dense and sparse lists should rank above
    chunks appearing in only one list, even if it ranks lower in each."""
    dense = [
        RetrievedChunk(chunk=_chunk("a"), score=0.9, rank=1, source="dense"),
        RetrievedChunk(chunk=_chunk("b"), score=0.7, rank=2, source="dense"),
        RetrievedChunk(chunk=_chunk("c"), score=0.5, rank=3, source="dense"),
    ]
    sparse = [
        RetrievedChunk(chunk=_chunk("d"), score=4.0, rank=1, source="sparse"),
        RetrievedChunk(chunk=_chunk("c"), score=3.0, rank=2, source="sparse"),
        RetrievedChunk(chunk=_chunk("e"), score=1.0, rank=3, source="sparse"),
    ]

    hybrid = HybridRetriever(FakeDense(dense), FakeSparse(sparse), rrf_k=60, top_k=5)
    results = hybrid.retrieve("test")

    ids = [r.chunk.chunk_id for r in results]
    # c is in both lists (rank 3 dense, rank 2 sparse) — RRF should push it up
    # rrf(c) = 1/(60+3) + 1/(60+2) = 0.01587 + 0.01613 = 0.032
    # rrf(a) = 1/(60+1) = 0.01639
    # rrf(d) = 1/(60+1) = 0.01639
    # c should beat both a and d
    assert ids[0] == "c", f"expected 'c' at rank 1, got '{ids[0]}'"
    assert len(results) == 5

    # verify scores are monotonically decreasing
    scores = [r.score for r in results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_rrf_with_no_overlap():
    """When dense and sparse return completely different chunks,
    all chunks should appear in output with single-source RRF scores."""
    dense = [
        RetrievedChunk(chunk=_chunk("a"), score=0.9, rank=1, source="dense"),
        RetrievedChunk(chunk=_chunk("b"), score=0.7, rank=2, source="dense"),
    ]
    sparse = [
        RetrievedChunk(chunk=_chunk("x"), score=3.0, rank=1, source="sparse"),
        RetrievedChunk(chunk=_chunk("y"), score=2.0, rank=2, source="sparse"),
    ]

    hybrid = HybridRetriever(FakeDense(dense), FakeSparse(sparse), rrf_k=60, top_k=10)
    results = hybrid.retrieve("test")

    ids = [r.chunk.chunk_id for r in results]
    assert set(ids) == {"a", "b", "x", "y"}
    # a and x both have rank 1 in their list, so same RRF score
    # b and y both have rank 2, same RRF score
    # the top 2 should be a and x (in some order), bottom 2 b and y
    assert results[0].score == results[1].score
    assert results[2].score == results[3].score
    assert results[0].score > results[2].score
