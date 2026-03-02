from bdc_monitor.domain import Citation, RetrievedChunk


class ContextAssembler:
    def assemble(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[str, dict[str, Citation]]:
        """Returns (context_string, citation_map keyed by chunk_id)."""
        parts = []
        citation_map: dict[str, Citation] = {}

        for rc in chunks:
            c = rc.chunk
            header = f"[CITE: {c.chunk_id}]"
            source = f"Source: {c.ticker} | {c.filing_type} | {c.period_end} | {c.section_type}"
            parts.append(f"{header}\n{source}\n{c.text}")

            citation_map[c.chunk_id] = Citation(
                chunk_id=c.chunk_id,
                text_span="",
                filing_accession=c.filing_accession,
                ticker=c.ticker,
                period_end=c.period_end,
                section_type=c.section_type,
            )

        context = "\n\n".join(parts)
        return context, citation_map
