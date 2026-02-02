from pathlib import Path

from pydantic_settings import BaseSettings


# hardcoded BDC list — we look up CIKs from EDGAR at runtime
BDCS = {
    "OBDC": "Blue Owl Credit Income Corp",
    "BCRED": "Blackstone Private Credit Fund",
    "ARCC": "Ares Capital Corporation",
    "KKRFS": "KKR FS Income Trust",
    "APOLLO": "Apollo Debt Solutions BDC",
    "PSEC": "Prospect Capital Corporation",
    "FSK": "FS KKR Capital Corp",
    "OBDT": "Blue Owl Technology Finance Corp",
}

FILING_TYPES = ["10-Q", "10-K"]

SECTION_TYPES = [
    "schedule_of_investments",
    "mdna",
    "tender_offer",
    "notes",
    "risk_factors",
    "other",
]


class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    cohere_api_key: str = ""
    edgar_user_agent: str = "BDC Monitor research@example.com"
    data_dir: Path = Path("./data")

    # retrieval defaults
    top_k: int = 20
    rerank_top_k: int = 10
    rrf_k: int = 60

    # llm defaults
    default_llm: str = "anthropic"  # or "openai"
    anthropic_model: str = "claude-sonnet-4-20250514"
    openai_model: str = "gpt-4o"

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def filings_dir(self) -> Path:
        return self.data_dir / "filings"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "metadata.db"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


def load_settings() -> Settings:
    return Settings()
