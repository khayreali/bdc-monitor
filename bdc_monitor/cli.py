import asyncio
import logging
from datetime import date
from typing import Optional

import typer

from bdc_monitor.config import BDCS, load_settings

app = typer.Typer(help="BDC Redemption Monitor — pull filings, index, and query BDC data")


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
):
    """Chunk and index ingested filings."""
    logging.basicConfig(level=logging.INFO)
    typer.echo("not implemented yet")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural language question about BDC filings"),
):
    """Ask a question about BDC filings."""
    logging.basicConfig(level=logging.INFO)
    typer.echo("not implemented yet")


@app.command("eval")
def run_eval(
    questions: str = typer.Option("eval/questions.yaml", help="Path to eval questions YAML"),
    config: str = typer.Option("configs/hybrid_reranked.yaml", help="Pipeline config YAML"),
):
    """Run evaluation against a question set."""
    logging.basicConfig(level=logging.INFO)
    typer.echo("not implemented yet")


@app.command()
def ablation(
    questions: str = typer.Option("eval/questions.yaml", help="Path to eval questions YAML"),
    configs: str = typer.Option("configs/*.yaml", help="Glob pattern for config files"),
):
    """Run ablation study across multiple pipeline configs."""
    logging.basicConfig(level=logging.INFO)
    typer.echo("not implemented yet")


if __name__ == "__main__":
    app()
