from abc import ABC, abstractmethod

import anthropic
import openai


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        ...


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        return resp.content[0].text


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
