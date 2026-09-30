"""Central configuration. All secrets and switches come from environment variables."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent      # .../backend
PROJECT_ROOT = BACKEND_DIR.parent                          # repo root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", extra="ignore"
    )

    # LLM (provider-agnostic: any OpenAI-compatible endpoint)
    llm_provider: str = "groq"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-20b"
    llm_fallback_model: str = "openai/gpt-oss-120b"
    groq_api_key: str = ""

    # Optional integrations
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "conversational-product-recommender"
    tavily_api_key: str = ""

    # App
    cors_origins: str = "http://localhost:5173"

    # Data paths
    catalog_csv: Path = PROJECT_ROOT / "data" / "amazon_dataset_synthetic_100.csv"
    processed_products: Path = BACKEND_DIR / "app" / "data" / "processed_products.json"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def tavily_enabled(self) -> bool:
        return bool(self.tavily_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()