import logging
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from bdc_monitor.domain import Chunk

log = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    """BM25 sparse retrieval index. Filters via post-filtering since
    BM25 doesn't support metadata natively."""

    def __init__(self, persist_path: Path):
        self.persist_path = persist_path
        self.chunks: list[Chunk] = []
        self.tokenized: list[list[str]] = []
        self._index: BM25Okapi | None = None

        if persist_path.exists():
            self._load()

    def _load(self):
        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.tokenized = data["tokenized"]
        if self.tokenized:
            self._index = BM25Okapi(self.tokenized)
        log.info(f"loaded BM25 index with {len(self.chunks)} chunks")

    def save(self):
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "tokenized": self.tokenized}, f)

    def add_chunks(self, chunks: list[Chunk]):
        for c in chunks:
            self.chunks.append(c)
            self.tokenized.append(_tokenize(c.text))
        # rebuild — rank_bm25 doesn't support incremental adds
        if self.tokenized:
            self._index = BM25Okapi(self.tokenized)

    def query(
        self,
        text: str,
        top_k: int = 20,
        where: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        if self._index is None or not self.chunks:
            return []

        scores = self._index.get_scores(_tokenize(text))

        # pair chunks with scores, apply metadata filter, sort
        results = list(zip(self.chunks, scores))
        if where:
            results = [
                (c, s) for c, s in results if _matches_filter(c, where)
            ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return len(self.chunks)


def _matches_filter(chunk: Chunk, where: dict) -> bool:
    for key, val in where.items():
        if key == "$and":
            return all(_matches_filter(chunk, sub) for sub in val)
        if key == "$or":
            return any(_matches_filter(chunk, sub) for sub in val)

        attr = getattr(chunk, key, None)
        if attr is None and key == "accession_number":
            attr = chunk.filing_accession
        attr_str = str(attr) if attr is not None else None

        if isinstance(val, dict):
            if "$eq" in val and attr_str != str(val["$eq"]):
                return False
            if "$in" in val and attr_str not in [str(v) for v in val["$in"]]:
                return False
        else:
            if attr_str != str(val):
                return False
    return True
