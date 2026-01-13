import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)
    
    # Pinecone configuration
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_environment: str
    
    # OpenAI configuration
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    
    # Application defaults
    default_namespace: str = "prod"
    
    # Retrieval defaults
    default_top_k: int = 20
    rerank_top_n: int = 4
    
    def validate(self) -> None:
        """Validate that all required settings are present."""
        missing = []
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if not self.pinecone_index_name:
            missing.append("PINECONE_INDEX_NAME")
        if not self.pinecone_environment:
            missing.append("PINECONE_ENVIRONMENT")
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please set these in your .env file or environment. See .env.example for reference."
            )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.validate()
    return _settings

