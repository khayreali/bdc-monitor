import logging
import time
from abc import ABC, abstractmethod

import openai

log = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbedder(EmbeddingModel):

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        for attempt in range(6):
            try:
                resp = self.client.embeddings.create(input=texts, model=self.model)
                return [item.embedding for item in resp.data]
            except openai.RateLimitError:
                wait = min(2 ** attempt, 60)
                log.warning(f"rate limited, retrying in {wait}s (attempt {attempt + 1}/6)...")
                time.sleep(wait)

        raise RuntimeError("embedding failed after 6 attempts")


class LocalEmbedder(EmbeddingModel):
    """Fallback when no OpenAI key is available."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embeddings]
