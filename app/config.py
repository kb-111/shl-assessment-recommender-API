"""
Centralized configuration via pydantic-settings + dotenv.
All env vars are read here; nothing else imports os.environ directly.
"""

# class Settings(BaseSettings):
#     gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
#     catalog_path: str = Field("data/shl_catalog.json", env="CATALOG_PATH")
#     faiss_index_path: str = Field("data/shl_faiss.index", env="FAISS_INDEX_PATH")
#     faiss_meta_path: str = Field("data/shl_faiss_meta.json", env="FAISS_META_PATH")
#     log_level: str = Field("INFO", env="LOG_LEVEL")
#     embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
#     top_k_retrieval: int = 20      # candidates fetched from FAISS before reranking
#     top_k_final: int = 10          # hard cap per API contract
#     # gemini_model: str = "gemini-1.5-flash" 
#     gemini_model = "models/gemini-1.5-flash-001"
#     request_timeout: int = 25      # keep well under 30s evaluator cap

#     model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    gemini_api_key: str = Field(validation_alias="GEMINI_API_KEY")

    catalog_path: str = Field(
        default="data/shl_catalog.json",
        validation_alias="CATALOG_PATH"
    )

    faiss_index_path: str = Field(
        default="data/shl_faiss.index",
        validation_alias="FAISS_INDEX_PATH"
    )

    faiss_meta_path: str = Field(
        default="data/shl_faiss_meta.json",
        validation_alias="FAISS_META_PATH"
    )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    top_k_retrieval: int = 20
    top_k_final: int = 10

    gemini_model: str = Field(
    default="models/gemini-3.1-flash-lite",
    validation_alias="GEMINI_MODEL"
    )

    request_timeout: int = 25

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


settings = Settings()