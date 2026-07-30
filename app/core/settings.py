from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "otsentinel_chunks"

    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "auto"
    embedding_batch_size: int = Field(
        default=4,
        ge=1,
    )

    ollama_base_url: str = "http://localhost:11434"
    generator_model_name: str = "llama3.2:3b"
    generation_temperature: float = 0.0
    min_retrieval_score: float = Field(default=0.35, ge=0.0, le=1.0)



@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance."""

    return Settings()
