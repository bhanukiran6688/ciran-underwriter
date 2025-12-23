"""
Centralized configuration loader.

- Reads environment variables (via python-dotenv if a .env is present).
- Validates required fields.
- Exposes a cached `get_settings()` for app-wide use.
"""

import os
from functools import lru_cache
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv


def _to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


class Settings(BaseModel):
    # Server
    PORT: int = Field(default=8000)
    LOG_LEVEL: str = Field(default="INFO")

    # --- LLM (required) ---
    GOOGLE_API_KEY: str
    GOOGLE_MODEL_NAME: str = Field(default="gemini-2.5-flash")
    # You can tweak generation behavior here if desired
    LLM_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=2048, ge=256, le=8192)

    class Config:
        frozen = True


def _build_settings_from_env() -> Settings:
    # Load .env if present (no-op if not)
    load_dotenv(override=False)

    missing: list[str] = []
    
    def req(name: str) -> str:
        v = os.getenv(name)
        if not v:
            missing.append(name)
            return ""
        return v

    settings = Settings(
        PORT=int(os.getenv("PORT", "8000")),
        LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
        GOOGLE_API_KEY=req("GOOGLE_API_KEY"),
        GOOGLE_MODEL_NAME=os.getenv("GOOGLE_MODEL_NAME", "gemini-2.5-flash"),
        LLM_TEMPERATURE=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        LLM_MAX_OUTPUT_TOKENS=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))
    )

    if missing:
        raise ValidationError.from_exception_data(
            title="Missing required environment variables",
            line_errors=[
                {
                    "type": "value_error.missing",
                    "loc": (name,),
                    "msg": f"Environment var '{name}' is required but not set.",
                    "input": None,
                }
                for name in missing
            ],
        )

    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get a cached Settings instance."""
    return _build_settings_from_env()
