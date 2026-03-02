import logging
import re

from bdc_monitor.domain import Answer, RetrievedChunk
from bdc_monitor.generation.llm_client import LLMClient
from bdc_monitor.retrieval.context_assembler import ContextAssembler

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a financial analyst assistant specializing in Business Development Companies (BDCs).
Answer the user's question using ONLY the provided context from SEC filings.

Rules:
1. Cite your sources using [CITE: chunk_id] inline, placed right after the claim they support.
2. Only use chunk_ids that appear in the context. Do not fabricate citations.
3. If the context doesn't contain enough information, say so.
4. Be precise with numbers, dates, and financial figures."""

_CITE_RE = re.compile(r"\[CITE:\s*([^\]]+?)\s*\]")


class Generator:

    def __init__(self, llm: LLMClient, assembler: ContextAssembler):
        self.llm = llm
        self.assembler = assembler

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        context, citation_map = self.assembler.assemble(chunks)

        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        # extract and validate cited chunk_ids
        cited_ids = _CITE_RE.findall(raw)
        seen = set()
        citations = []
        for cid in cited_ids:
            if cid in seen:
                continue
            seen.add(cid)
            if cid in citation_map:
                citations.append(citation_map[cid])
            else:
                log.warning(f"hallucinated citation: {cid}")

        return Answer(
            question=question,
            answer_text=raw,
            citations=citations,
            chunks_used=chunks,
        )
