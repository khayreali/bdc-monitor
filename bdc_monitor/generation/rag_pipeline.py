from bdc_monitor.domain import Answer
from bdc_monitor.generation.generator import Generator
from bdc_monitor.retrieval.retrievers import Retriever


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        reranker=None,
        query_router=None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.query_router = query_router

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        where = None
        if self.query_router:
            where = self.query_router.route(question)

        chunks = self.retriever.retrieve(question, top_k=top_k, where=where)

        if self.reranker and chunks:
            chunks = self.reranker.rerank(question, chunks)

        return self.generator.generate(question, chunks)
