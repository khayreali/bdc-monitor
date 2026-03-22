import json
import logging
from pathlib import Path

import yaml

from bdc_monitor.config import Settings
from bdc_monitor.domain import EvalQuestion, EvalResult
from bdc_monitor.evaluation.evaluator import Evaluator
from bdc_monitor.generation.generator import Generator
from bdc_monitor.generation.llm_client import AnthropicClient, LLMClient, OpenAIClient
from bdc_monitor.generation.rag_pipeline import RAGPipeline
from bdc_monitor.indexing.bm25_index import BM25Index
from bdc_monitor.indexing.embedder import EmbeddingModel
from bdc_monitor.indexing.vector_store import VectorStore
from bdc_monitor.retrieval.context_assembler import ContextAssembler
from bdc_monitor.retrieval.query_router import QueryRouter
from bdc_monitor.retrieval.reranker import Reranker
from bdc_monitor.retrieval.retrievers import (
    DenseRetriever,
    HybridRetriever,
    SparseRetriever,
)

log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_pipeline(
    config: dict, settings: Settings,
    embedder: EmbeddingModel, vs: VectorStore, bm25: BM25Index,
) -> RAGPipeline:
    top_k = config.get("top_k", settings.top_k)
    rerank_top_k = config.get("rerank_top_k", settings.rerank_top_k)

    dense = DenseRetriever(vs, embedder, top_k=top_k)
    sparse = SparseRetriever(bm25, top_k=top_k)

    retriever_type = config.get("retriever", "hybrid")
    if retriever_type == "dense":
        retriever = dense
    elif retriever_type == "sparse":
        retriever = sparse
    else:
        retriever = HybridRetriever(dense, sparse, rrf_k=settings.rrf_k, top_k=top_k)

    llm_type = config.get("llm", settings.default_llm)
    llm = _build_llm(llm_type, settings)

    reranker = None
    if config.get("reranker", False):
        reranker = Reranker(cohere_api_key=settings.cohere_api_key, top_k=rerank_top_k)

    router = QueryRouter(llm)

    assembler = ContextAssembler()
    generator = Generator(llm, assembler)

    return RAGPipeline(
        retriever=retriever, generator=generator,
        reranker=reranker, query_router=router,
    )


def _build_llm(llm_type: str, settings: Settings) -> LLMClient:
    if llm_type == "anthropic":
        return AnthropicClient(settings.anthropic_api_key, settings.anthropic_model)
    return OpenAIClient(settings.openai_api_key, settings.openai_model)


class AblationRunner:

    def __init__(self, settings: Settings, embedder: EmbeddingModel | None = None):
        self.settings = settings
        if embedder is None:
            if settings.openai_api_key:
                from bdc_monitor.indexing.embedder import OpenAIEmbedder
                embedder = OpenAIEmbedder(api_key=settings.openai_api_key)
            else:
                from bdc_monitor.indexing.embedder import LocalEmbedder
                embedder = LocalEmbedder()
        self.embedder = embedder
        self.vs = VectorStore(chroma_dir=settings.chroma_dir)
        self.bm25 = BM25Index(persist_path=settings.data_dir / "bm25_index.pkl")

    def run(
        self, questions: list[EvalQuestion], config_paths: list[str]
    ) -> dict[str, list[EvalResult]]:
        judge = _build_llm(self.settings.default_llm, self.settings)
        all_results = {}

        for path in config_paths:
            config = load_config(path)
            name = config.get("name", Path(path).stem)
            log.info(f"\n=== Config: {name} ===")

            pipeline = build_pipeline(
                config, self.settings, self.embedder, self.vs, self.bm25
            )
            evaluator = Evaluator(pipeline, judge)
            results = evaluator.run(questions, config_name=name)
            all_results[name] = results

        return all_results

    @staticmethod
    def save_results(all_results: dict[str, list[EvalResult]], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {}
        for name, results in all_results.items():
            n = len(results)
            out[name] = {
                "n_questions": n,
                "retrieval_recall": sum(r.retrieval_recall for r in results) / n,
                "citation_precision": sum(r.citation_precision for r in results) / n,
                "fact_recall": sum(r.fact_recall for r in results) / n,
                "per_question": [
                    {
                        "question": r.question.question,
                        "difficulty": r.question.difficulty,
                        "retrieval_recall": r.retrieval_recall,
                        "citation_precision": r.citation_precision,
                        "fact_recall": r.fact_recall,
                    }
                    for r in results
                ],
            }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        log.info(f"saved results to {path}")

    @staticmethod
    def print_comparison(all_results: dict[str, list[EvalResult]]):
        print()
        header = f"{'Config':<25s} {'Retrieval':>10s} {'Citation':>10s} {'Fact':>10s} {'Avg':>10s}"
        print(header)
        print("-" * len(header))

        for name, results in all_results.items():
            if not results:
                continue
            n = len(results)
            avg_ret = sum(r.retrieval_recall for r in results) / n
            avg_cit = sum(r.citation_precision for r in results) / n
            avg_fact = sum(r.fact_recall for r in results) / n
            avg_all = (avg_ret + avg_cit + avg_fact) / 3

            print(f"{name:<25s} {avg_ret:>10.3f} {avg_cit:>10.3f} {avg_fact:>10.3f} {avg_all:>10.3f}")

        print()

        # per-difficulty breakdown
        difficulties = ["easy", "medium", "hard"]
        for diff in difficulties:
            print(f"\n{diff.upper()} questions:")
            print(f"{'Config':<25s} {'Retrieval':>10s} {'Citation':>10s} {'Fact':>10s}")
            print("-" * 55)
            for name, results in all_results.items():
                subset = [r for r in results if r.question.difficulty == diff]
                if not subset:
                    continue
                n = len(subset)
                avg_ret = sum(r.retrieval_recall for r in subset) / n
                avg_cit = sum(r.citation_precision for r in subset) / n
                avg_fact = sum(r.fact_recall for r in subset) / n
                print(f"{name:<25s} {avg_ret:>10.3f} {avg_cit:>10.3f} {avg_fact:>10.3f}")
