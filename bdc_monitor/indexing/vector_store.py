from pathlib import Path

import chromadb

from bdc_monitor.domain import Chunk


class VectorStore:
    """Thin wrapper over ChromaDB for dense retrieval."""

    def __init__(self, chroma_dir: Path, collection_name: str = "bdc_chunks"):
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]):
        if not chunks:
            return
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "ticker": c.ticker,
                    "period_end": str(c.period_end),
                    "filing_type": c.filing_type,
                    "section_type": c.section_type,
                    "accession_number": c.filing_accession,
                }
                for c in chunks
            ],
        )

    def query(
        self,
        embedding: list[float],
        top_k: int = 20,
        where: dict | None = None,
    ) -> dict:
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def count(self) -> int:
        return self.collection.count()
