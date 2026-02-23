import logging
import time
from datetime import date

from bdc_monitor.domain import Chunk
from bdc_monitor.indexing.bm25_index import BM25Index
from bdc_monitor.indexing.chunker import Chunker
from bdc_monitor.indexing.embedder import EmbeddingModel
from bdc_monitor.indexing.metadata_store import MetadataStore
from bdc_monitor.indexing.vector_store import VectorStore

log = logging.getLogger(__name__)

BATCH_SIZE = 50


class IndexingPipeline:
    """Chunks sections, embeds them, and writes to vector + BM25 stores."""

    def __init__(
        self,
        store: MetadataStore,
        chunker: Chunker,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ):
        self.store = store
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25_index

    def run(self) -> dict:
        sections = self.store.get_unindexed_sections()
        if not sections:
            log.info("nothing to index")
            return {"sections": 0, "chunks": 0}

        log.info(f"indexing {len(sections)} sections")

        total_chunks = 0
        batch: list[Chunk] = []

        for i, row in enumerate(sections):
            section_id, accession, ticker, period_end, filing_type, section_type, text = row

            chunks = self.chunker.chunk(
                text=text,
                filing_accession=accession,
                section_id=section_id,
                ticker=ticker,
                period_end=date.fromisoformat(period_end),
                filing_type=filing_type,
                section_type=section_type,
            )
            batch.extend(chunks)

            if len(batch) >= BATCH_SIZE:
                self._flush(batch)
                total_chunks += len(batch)
                batch = []
                time.sleep(0.5)

            if (i + 1) % 50 == 0:
                log.info(f"  processed {i + 1}/{len(sections)} sections, {total_chunks} chunks so far")

        if batch:
            self._flush(batch)
            total_chunks += len(batch)

        self.bm25.save()
        log.info(f"done: {len(sections)} sections -> {total_chunks} chunks")
        return {"sections": len(sections), "chunks": total_chunks}

    def _flush(self, chunks: list[Chunk]):
        try:
            embeddings = self.embedder.embed([c.text for c in chunks])
            self.vector_store.add_chunks(chunks, embeddings)
        except Exception:
            log.exception(f"embedding/vector store failed for batch of {len(chunks)}")
            return

        self.bm25.add_chunks(chunks)
        self.store.insert_chunks(chunks)
