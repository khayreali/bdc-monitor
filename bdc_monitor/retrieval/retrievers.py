import logging
from abc import ABC, abstractmethod
from datetime import date

from bdc_monitor.domain import Chunk, RetrievedChunk
from bdc_monitor.indexing.bm25_index import BM25Index
from bdc_monitor.indexing.embedder import EmbeddingModel
from bdc_monitor.indexing.vector_store import VectorStore

log = logging.getLogger(__name__)


class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self, query: str, top_k: int | None = None, where: dict | None = None
    ) -> list[RetrievedChunk]:
        ...


def _chroma_to_chunks(results: dict) -> list[RetrievedChunk]:
    """Convert ChromaDB query results to RetrievedChunk list."""
    if not results["ids"] or not results["ids"][0]:
        return []

    ids = results["ids"][0]
    distances = results["distances"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    chunks = []
    for i, (chunk_id, dist, text, meta) in enumerate(
        zip(ids, distances, documents, metadatas)
    ):
        # parse chunk_index from chunk_id: {accession}_{section_id}_{index}
        parts = chunk_id.rsplit("_", 2)
        chunk_index = int(parts[-1]) if len(parts) >= 2 else 0

        chunk = Chunk(
            chunk_id=chunk_id,
            filing_accession=meta.get("accession_number", ""),
            ticker=meta.get("ticker", ""),
            period_end=date.fromisoformat(meta["period_end"]),
            filing_type=meta.get("filing_type", ""),
            section_type=meta.get("section_type", ""),
            text=text,
            token_count=len(text.split()) * 4 // 3,  # rough estimate
            chunk_index=chunk_index,
        )
        chunks.append(RetrievedChunk(
            chunk=chunk,
            score=1.0 - dist,  # cosine distance -> similarity
            rank=i + 1,
            source="dense",
        ))
    return chunks


class DenseRetriever(Retriever):
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingModel, top_k: int = 20):
        self.vs = vector_store
        self.embedder = embedder
        self.default_top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None, where: dict | None = None):
        k = top_k or self.default_top_k
        embedding = self.embedder.embed([query])[0]
        results = self.vs.query(embedding, top_k=k, where=where)
        return _chroma_to_chunks(results)


class SparseRetriever(Retriever):
    def __init__(self, bm25_index: BM25Index, top_k: int = 20):
        self.bm25 = bm25_index
        self.default_top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None, where: dict | None = None):
        k = top_k or self.default_top_k
        results = self.bm25.query(query, top_k=k, where=where)

        return [
            RetrievedChunk(chunk=chunk, score=score, rank=i + 1, source="sparse")
            for i, (chunk, score) in enumerate(results)
        ]


class HybridRetriever(Retriever):
    """Merges dense + sparse results using reciprocal rank fusion."""

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        rrf_k: int = 60,
        top_k: int = 20,
    ):
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.default_top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None, where: dict | None = None):
        k = top_k or self.default_top_k

        dense_results = self.dense.retrieve(query, top_k=k, where=where)
        sparse_results = self.sparse.retrieve(query, top_k=k, where=where)

        # RRF: score = sum of 1/(k + rank) across both lists
        scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rc in dense_results:
            cid = rc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + 1.0 / (self.rrf_k + rc.rank)
            chunk_map[cid] = rc.chunk  # prefer dense version

        for rc in sparse_results:
            cid = rc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + 1.0 / (self.rrf_k + rc.rank)
            if cid not in chunk_map:
                chunk_map[cid] = rc.chunk

        # sort by RRF score, assign new ranks
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            RetrievedChunk(
                chunk=chunk_map[cid], score=score, rank=i + 1, source="hybrid"
            )
            for i, (cid, score) in enumerate(ranked)
        ]
