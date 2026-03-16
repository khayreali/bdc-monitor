import asyncio
import logging
from datetime import date
from typing import Optional

import typer

from bdc_monitor.config import BDCS, load_settings

app = typer.Typer(help="BDC Redemption Monitor — pull filings, index, and query BDC data")


def _get_embedder(settings, force_local: bool = False):
    if settings.openai_api_key and not force_local:
        from bdc_monitor.indexing.embedder import OpenAIEmbedder
        return OpenAIEmbedder(api_key=settings.openai_api_key)
    from bdc_monitor.indexing.embedder import LocalEmbedder
    typer.echo("using local embeddings (all-MiniLM-L6-v2)")
    return LocalEmbedder()


@app.command()
def ingest(
    since: str = typer.Option("2024-01-01", help="Earliest filing date (YYYY-MM-DD)"),
    bdcs: Optional[str] = typer.Option(None, help="Comma-separated tickers, e.g. OBDC,ARCC"),
):
    """Pull filings from SEC EDGAR and store locally."""
    logging.basicConfig(level=logging.INFO)

    settings = load_settings()
    tickers = [t.strip() for t in bdcs.split(",")] if bdcs else list(BDCS.keys())
    since_date = date.fromisoformat(since)

    asyncio.run(_run_ingest(settings, tickers, since_date))


async def _run_ingest(settings, tickers, since_date):
    from bdc_monitor.indexing.metadata_store import MetadataStore
    from bdc_monitor.ingestion.edgar_client import EdgarClient
    from bdc_monitor.ingestion.pipeline import IngestionPipeline

    store = MetadataStore(settings.db_path)
    async with EdgarClient(settings) as client:
        pipeline = IngestionPipeline(client, store)
        await pipeline.run(tickers, since_date)
    store.close()


@app.command()
def index(
    chunker: str = typer.Option("fixed", help="Chunking strategy: fixed, structure, section"),
    local_embeddings: bool = typer.Option(False, "--local-embeddings", help="Force local embeddings instead of OpenAI"),
):
    """Chunk and index ingested filings."""
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()

    from bdc_monitor.indexing.bm25_index import BM25Index
    from bdc_monitor.indexing.chunker import (
        FixedSizeChunker,
        SectionAwareChunker,
        StructureAwareChunker,
    )
    from bdc_monitor.indexing.metadata_store import MetadataStore
    from bdc_monitor.indexing.pipeline import IndexingPipeline
    from bdc_monitor.indexing.vector_store import VectorStore

    store = MetadataStore(settings.db_path)

    chunkers = {
        "fixed": FixedSizeChunker,
        "structure": StructureAwareChunker,
        "section": SectionAwareChunker,
    }
    if chunker not in chunkers:
        typer.echo(f"unknown chunker '{chunker}', options: {', '.join(chunkers)}")
        raise typer.Exit(1)
    chunker_impl = chunkers[chunker]()

    embedder = _get_embedder(settings, force_local=local_embeddings)
    vs = VectorStore(chroma_dir=settings.chroma_dir)
    bm25 = BM25Index(persist_path=settings.data_dir / "bm25_index.pkl")

    pipeline = IndexingPipeline(store, chunker_impl, embedder, vs, bm25)
    stats = pipeline.run()

    typer.echo(f"indexed {stats['sections']} sections -> {stats['chunks']} chunks")
    typer.echo(f"vector store: {vs.count()} | bm25: {bm25.count()}")
    store.close()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural language question about BDC filings"),
):
    """Ask a question about BDC filings."""
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()

    from bdc_monitor.generation.generator import Generator
    from bdc_monitor.generation.llm_client import AnthropicClient, OpenAIClient
    from bdc_monitor.generation.rag_pipeline import RAGPipeline
    from bdc_monitor.indexing.bm25_index import BM25Index
    from bdc_monitor.indexing.vector_store import VectorStore
    from bdc_monitor.retrieval.context_assembler import ContextAssembler
    from bdc_monitor.retrieval.query_router import QueryRouter
    from bdc_monitor.retrieval.reranker import Reranker
    from bdc_monitor.retrieval.retrievers import (
        DenseRetriever,
        HybridRetriever,
        SparseRetriever,
    )

    vs = VectorStore(chroma_dir=settings.chroma_dir)
    if vs.count() == 0:
        typer.echo("error: no indexed chunks. run 'bdc index' first.")
        raise typer.Exit(1)

    embedder = _get_embedder(settings)
    bm25 = BM25Index(persist_path=settings.data_dir / "bm25_index.pkl")

    dense = DenseRetriever(vs, embedder, top_k=settings.top_k)
    sparse = SparseRetriever(bm25, top_k=settings.top_k)
    retriever = HybridRetriever(dense, sparse, rrf_k=settings.rrf_k, top_k=settings.top_k)

    if settings.default_llm == "anthropic":
        if not settings.anthropic_api_key:
            typer.echo("error: ANTHROPIC_API_KEY not set in .env")
            raise typer.Exit(1)
        llm = AnthropicClient(settings.anthropic_api_key, settings.anthropic_model)
    else:
        if not settings.openai_api_key:
            typer.echo("error: OPENAI_API_KEY not set in .env")
            raise typer.Exit(1)
        llm = OpenAIClient(settings.openai_api_key, settings.openai_model)

    assembler = ContextAssembler()
    generator = Generator(llm, assembler)
    router = QueryRouter(llm)
    reranker = Reranker(
        cohere_api_key=settings.cohere_api_key,
        top_k=settings.rerank_top_k,
    )
    pipeline = RAGPipeline(
        retriever=retriever, generator=generator,
        reranker=reranker, query_router=router,
    )

    answer = pipeline.ask(question)

    typer.echo(f"\n{answer.answer_text}\n")
    if answer.citations:
        typer.echo("Sources:")
        for i, c in enumerate(answer.citations, 1):
            typer.echo(f"  [{i}] {c.ticker} | {c.period_end} | {c.section_type} ({c.chunk_id[:40]}...)")
    typer.echo(f"\n({len(answer.chunks_used)} chunks retrieved, {len(answer.citations)} cited)")


@app.command("eval")
def run_eval(
    questions: str = typer.Option("eval/questions.yaml", help="Path to eval questions YAML"),
    config: str = typer.Option("configs/hybrid_reranked.yaml", help="Pipeline config YAML"),
):
    """Run evaluation against a question set."""
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()

    from bdc_monitor.evaluation.ablation import _build_llm, build_pipeline, load_config
    from bdc_monitor.evaluation.evaluator import Evaluator, load_eval_questions
    from bdc_monitor.indexing.bm25_index import BM25Index
    from bdc_monitor.indexing.vector_store import VectorStore

    eqs = load_eval_questions(questions)
    cfg = load_config(config)
    name = cfg.get("name", "default")

    embedder = _get_embedder(settings)
    vs = VectorStore(chroma_dir=settings.chroma_dir)
    bm25 = BM25Index(persist_path=settings.data_dir / "bm25_index.pkl")

    pipeline = build_pipeline(cfg, settings, embedder, vs, bm25)
    judge = _build_llm(settings.default_llm, settings)
    evaluator = Evaluator(pipeline, judge)

    results = evaluator.run(eqs, config_name=name)

    n = len(results)
    avg_ret = sum(r.retrieval_recall for r in results) / n
    avg_cit = sum(r.citation_precision for r in results) / n
    avg_fact = sum(r.fact_recall for r in results) / n

    typer.echo(f"\n=== {name} ({n} questions) ===")
    typer.echo(f"  retrieval_recall:   {avg_ret:.3f}")
    typer.echo(f"  citation_precision: {avg_cit:.3f}")
    typer.echo(f"  fact_recall:        {avg_fact:.3f}")


@app.command()
def ablation(
    questions: str = typer.Option("eval/questions.yaml", help="Path to eval questions YAML"),
    configs: str = typer.Option("configs/*.yaml", help="Glob pattern for config files"),
):
    """Run ablation study across multiple pipeline configs."""
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()

    from glob import glob
    from pathlib import Path

    from bdc_monitor.evaluation.ablation import AblationRunner
    from bdc_monitor.evaluation.evaluator import load_eval_questions

    eqs = load_eval_questions(questions)
    config_paths = sorted(glob(configs))
    if not config_paths:
        typer.echo(f"no config files found matching '{configs}'")
        raise typer.Exit(1)

    typer.echo(f"running ablation: {len(eqs)} questions x {len(config_paths)} configs")
    for p in config_paths:
        typer.echo(f"  - {p}")

    runner = AblationRunner(settings)
    all_results = runner.run(eqs, config_paths)
    AblationRunner.print_comparison(all_results)

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"eval/results/ablation_{ts}.json")
    AblationRunner.save_results(all_results, out_path)
    typer.echo(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    app()
