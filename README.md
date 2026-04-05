# BDC Redemption Monitor

A local RAG pipeline that pulls SEC filings for 8 non-traded and listed BDCs, indexes them with hybrid retrieval, and answers natural-language questions about redemption rates, leverage, manager commentary, and asset quality — with inline citations to specific filings.

I built this because I wanted to understand how RAG systems actually work, not just call `langchain.RetrievalQA` and hope for the best. I have an accounting background, and the BDC space is interesting because there's real tension between what the numbers show (Schedule of Investments, leverage ratios) and what managers write in their MD&A. I wanted a tool that could surface those gaps.

The part I'm most proud of is the ablation runner. Anyone can build a RAG pipeline — the harder question is whether different configurations actually produce different results and why.

## does it work?

I ran the ablation multiple times as the project evolved — first with local embeddings (all-MiniLM-L6-v2, 384d) while debugging my OpenAI billing, then with text-embedding-3-small (1536d) after getting that sorted. These are the final numbers on a complete index of 32,791 chunks:

| Config | Retrieval Recall | Citation Precision | Fact Recall | Avg |
|--------|:---:|:---:|:---:|:---:|
| dense_only | 0.675 | 0.312 | 0.475 | 0.487 |
| sparse_only | 0.725 | 0.508 | 0.375 | 0.536 |
| hybrid | **0.725** | **0.515** | **0.425** | **0.555** |
| hybrid_reranked | 0.700 | 0.296 | 0.400 | 0.465 |

Hybrid wins overall (0.555 avg), with the best retrieval recall and citation precision. Sparse alone is surprisingly close (0.536) — BM25 does well on financial text because a lot of the signal is exact terms. Dense-only trails because it misses keyword-specific matches that BM25 catches.

The reranker doesn't help. I investigated this across several ablation runs. With tight trimming (20 candidates → 10 output) it dramatically improved citation precision but tanked retrieval recall. With looser settings (40 → 20) the retrieval improved but citation precision dropped. It's a genuine tradeoff, not a bug — the local cross-encoder (bge-reranker-base) just isn't good enough on financial text to be a net win. I'd want to try Cohere's reranker or a finance-domain model before writing off reranking entirely.

Easy questions (single-filing, single-fact) work well across configs. The failures are in medium and hard questions — cross-period trends and cross-issuer comparisons need more context than 20 chunks can provide.

With 20 eval questions the numbers shift ±5% between runs from LLM non-determinism. I flagged 10 questions that need hand-review. A real eval would need 60+ questions with a second labeler.

## what it does

Pulls 10-Qs and 10-Ks from SEC EDGAR for these BDCs:
- Blue Owl Capital Corp (OBDC)
- Blackstone Private Credit Fund (BCRED)
- Ares Capital Corporation (ARCC)
- KKR FS Income Trust
- Apollo Debt Solutions BDC
- Prospect Capital Corporation (PSEC)
- FS KKR Capital Corp (FSK)
- Blue Owl Technology Finance Corp

Parses each filing into sections (Schedule of Investments, MD&A, Risk Factors, Notes, etc.), chunks them, embeds them, and indexes into ChromaDB + BM25. Then you ask questions and get answers with citations to specific filing sections.

## setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo>
cd bdc-monitor
uv sync
cp .env.example .env
# fill in your API keys in .env
```

You need `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` for the LLM. For embeddings, it'll use `OPENAI_API_KEY` if set, otherwise falls back to a local model (all-MiniLM-L6-v2 via sentence-transformers). `COHERE_API_KEY` is optional — without it the reranker uses a local cross-encoder.

## usage

```bash
# 1. pull filings from EDGAR (takes a few minutes, rate-limited at 10 req/s)
bdc ingest --since 2024-01-01

# 2. chunk and index everything (~32k chunks with fixed chunker)
bdc index --chunker fixed

# 3. ask questions
bdc ask "What was OBDC's NAV per share in Q3 2025?"
bdc ask "Compare leverage ratios across all 8 BDCs"

# 4. run eval against the question set
bdc eval --config configs/hybrid_reranked.yaml

# 5. run ablation across all configs
bdc ablation --configs "configs/*.yaml"
```

The `--chunker` flag accepts `fixed` (500-token windows), `structure` (paragraph-aware), or `section` (specialized for Schedule of Investments tables). You'd re-index between chunker changes.

## architecture

Four layers, dependencies flow downward:

```
evaluation/     Evaluator, AblationRunner
generation/     LLMClient, Generator, RAGPipeline
retrieval/      Retrievers (Dense/Sparse/Hybrid), Reranker, QueryRouter, ContextAssembler
indexing/       Chunkers, Embedder, VectorStore (ChromaDB), BM25Index, MetadataStore
ingestion/      EdgarClient, FilingParsers, SectionClassifier
domain/         Pydantic models (Filing, Section, Chunk, Answer, etc.)
```

Ingestion is async — the EDGAR client rate-limits at 10 req/s and caches downloads to disk. Three separate parsers for 10-Q, 10-K, and shareholder letters because the HTML structures are genuinely different. The section classifier uses regex to label sections (schedule_of_investments, mdna, risk_factors, etc.) — I planned an LLM fallback for ambiguous ones but haven't implemented it. Regex handles about 90% of sections.

Three chunking strategies for the ablation: fixed-size token windows, paragraph-aware, and a specialized one for Schedule of Investments that splits on page headers. Every chunk goes into both ChromaDB (dense, cosine) and a BM25 index (sparse). Metadata on every chunk so the retriever can filter before scoring.

The hybrid retriever merges dense + sparse results with reciprocal rank fusion (k=60). A query router extracts entities from the question (ticker, quarter, topic) and builds metadata filters. The reranker uses Cohere's API when available, otherwise a local cross-encoder.

The LLM gets chunks wrapped with `[CITE: chunk_id]` markers and is told to cite inline. After generation, every cited chunk_id gets validated against the actual retrieved set — hallucinated citations get stripped.

## the eval harness

Three metrics:

- retrieval_recall — did we retrieve chunks from the right filing sections? Mechanical check.
- citation_precision — do the cited chunks actually support the claims? LLM-as-judge.
- fact_recall — how many expected facts appear in the answer? LLM-as-judge.

The ablation runner takes the same questions, runs them against each pipeline config, and prints a comparison table with per-difficulty breakdowns. Results also get dumped to JSON (`eval/results/`).

## tradeoffs

Chunking: fixed-size windows are consistent but split table rows in half. The structure-aware chunker respects paragraphs, great for MD&A prose, useless for the Schedule of Investments since that's one giant table extracted as flat text. The section-aware chunker splits SOI on page breaks — crude but it lines up with how EDGAR paginates things.

Hybrid retrieval: BM25 is surprisingly good for financial queries. A lot of the signal is in exact terms — "non-accrual," "tender offer," "5% cap." Dense retrieval misses exact matches more often than I expected. The 20-point retrieval recall gap between dense-only and hybrid in the ablation confirms this.

Query routing: I almost didn't build this. But without it, asking about OBDC's Q3 2025 NAV pulls chunks from random BDCs and random quarters. A regex + LLM entity extractor that builds metadata filters — maybe 50 lines of code — improved retrieval noticeably.

## limitations and what's next

The eval set is 20 questions labeled by me. Out of those, 8 got fact_recall=0 across every config. Some of that is the system's fault (can't cover 8 BDCs in 20 chunks), but some is my fault — the expected facts are vague descriptions like "leverage ratio for each quarter" rather than specific numbers. A real eval needs 60-100 questions with concrete expected facts and a second labeler. That's the obvious first thing to fix.

The corpus is 8 BDCs over ~8 quarters (2024-2025). Expanding back to 2022 would cover the full arc of the BDC redemption crisis — when multiple non-traded BDCs started capping redemptions at 5% and investors got antsy. I can only answer questions about the tail end of that story right now.

The reranker story was more complicated than I expected. With local embeddings it was clearly making things worse — dropping retrieval recall by 20 points. I assumed the local embeddings were giving the cross-encoder bad candidates. But even after switching to OpenAI embeddings, the reranker doesn't clearly beat plain hybrid overall. What it DOES do is dramatically improve citation precision (0.539 vs 0.386 in v3) — the reranker is genuinely good at promoting chunks that support the claims they're cited for. But it occasionally demotes a relevant chunk below the cutoff, killing retrieval recall for that question. I bumped the candidate pool from 20→40 and the output from 10→20 which helped, but it's still a tradeoff rather than a free win. I think the right call for production would be a Cohere reranker instead of the local cross-encoder, plus a bigger candidate pool.

The section classifier originally produced zero "tender_offer" sections. Turns out BDC filings don't have standalone tender offer headings — the repurchase program info lives inside Item 5 or Item 2 as a sub-heading like "Share Repurchase Program" or "Stock Repurchase Program." I fixed this by having the classifier also scan the first 15k chars of section text for these patterns, not just the title. Now 50 sections classify as tender_offer. The fix didn't directly improve the repurchase-related eval questions though, because the retriever finds the same text either way — the label change just makes section-type filtering work correctly. If I did this again I'd train a classifier on labeled examples instead of guessing at regex.

I'd also build a real table parser for the Schedule of Investments. Right now it's flat text — a chunk might have "IRI Group Holdings, Inc.\nFirst lien senior secured loan\nS+\n4.25\n%\n12/2029\n$\n42,404" with zero column structure. Preserving the table schema would make portfolio composition questions much more answerable.

One last thing: EDGAR's HTML is wildly inconsistent across filers. OBDC puts a period between "PART I." and "FINANCIAL INFORMATION," BCRED uses a hyphen, and some filers use non-breaking spaces where normal spaces should go. I spent more time on HTML parsing edge cases than any other part of this project.
