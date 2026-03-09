import logging

from bdc_monitor.domain import RetrievedChunk

log = logging.getLogger(__name__)


class Reranker:
    """Cohere rerank API, with local bge-reranker-base as fallback."""

    def __init__(self, cohere_api_key: str = "", top_k: int = 10):
        self.top_k = top_k
        self._cohere = None
        self._cross_encoder = None

        if cohere_api_key:
            import cohere
            self._cohere = cohere.ClientV2(api_key=cohere_api_key)
            log.info("reranker: using Cohere API")
        else:
            log.info("reranker: using local cross-encoder (BAAI/bge-reranker-base)")
            self._load_cross_encoder()

    def _load_cross_encoder(self):
        from sentence_transformers import CrossEncoder
        self._cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        if not chunks:
            return []

        if self._cohere:
            return self._rerank_cohere(query, chunks, k)
        return self._rerank_local(query, chunks, k)

    def _rerank_cohere(self, query, chunks, k):
        docs = [rc.chunk.text for rc in chunks]
        try:
            resp = self._cohere.rerank(
                query=query,
                documents=docs,
                model="rerank-english-v3.0",
                top_n=k,
            )
        except Exception:
            log.exception("cohere rerank failed, falling back to local")
            if not self._cross_encoder:
                self._load_cross_encoder()
            return self._rerank_local(query, chunks, k)

        reranked = []
        for i, result in enumerate(resp.results):
            rc = chunks[result.index]
            reranked.append(RetrievedChunk(
                chunk=rc.chunk,
                score=result.relevance_score,
                rank=i + 1,
                source=rc.source,
            ))
        return reranked

    def _rerank_local(self, query, chunks, k):
        pairs = [(query, rc.chunk.text) for rc in chunks]
        scores = self._cross_encoder.predict(pairs)

        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievedChunk(
                chunk=rc.chunk,
                score=float(score),
                rank=i + 1,
                source=rc.source,
            )
            for i, (rc, score) in enumerate(scored[:k])
        ]
