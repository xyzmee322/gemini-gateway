from __future__ import annotations

from functools import lru_cache
import re
from typing import Any, Literal
from urllib.parse import urlsplit

from cryptography.fernet import Fernet
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PLACEHOLDER_SECRET_PATTERN = re.compile(
    r"(^change-me\b|^dev[-_]|dev-only|local-token|placeholder|replace-with|generate-with)",
    re.IGNORECASE,
)


class GeminiGatewaySettings(BaseSettings):
    """Настройки внутреннего Gemini gateway."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="GEMINI_GATEWAY_",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = "gemini-gateway"
    environment: Literal["development", "staging", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = Field(default=8010, ge=1, le=65535)
    postgres_dsn: str
    postgres_sync_dsn: str
    encryption_key: SecretStr
    hmac_key: SecretStr = Field(min_length=32)
    internal_auth_token: SecretStr = Field(min_length=16)
    upstream_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    default_request_timeout_seconds: float = Field(default=35.0, ge=1.0, le=180.0)
    cooldown_jitter_percent: int = Field(default=15, ge=0, le=50)
    route_attempts_ttl_days: int = Field(default=30, ge=1, le=365)
    retention_interval_seconds: float = Field(default=3600.0, ge=1.0, le=86_400.0)
    require_seeded_routes: bool = False
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_embeddings_fallback_enabled: bool = False
    openrouter_embeddings_fallback_model: str = "google/gemini-embedding-2"

    @field_validator("postgres_dsn")
    @classmethod
    def validate_async_dsn(cls, value: str) -> str:
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError("postgres_dsn must use postgresql+asyncpg://")
        return value

    @field_validator("postgres_sync_dsn")
    @classmethod
    def validate_sync_dsn(cls, value: str) -> str:
        if not value.startswith("postgresql+psycopg://"):
            raise ValueError("postgres_sync_dsn must use postgresql+psycopg://")
        return value

    @field_validator("encryption_key")
    @classmethod
    def validate_fernet_key(cls, value: SecretStr) -> SecretStr:
        Fernet(value.get_secret_value().encode("ascii"))
        return value

    @field_validator("openrouter_api_key", mode="before")
    @classmethod
    def normalize_optional_openrouter_key(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("openrouter_api_key")
    @classmethod
    def validate_optional_openrouter_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        raw_value = value.get_secret_value().strip()
        if len(raw_value) < 16:
            raise ValueError("openrouter_api_key must be at least 16 characters")
        return SecretStr(raw_value)

    @field_validator("openrouter_base_url")
    @classmethod
    def normalize_openrouter_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed_url = urlsplit(normalized)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
            raise ValueError("openrouter_base_url must be an absolute URL")
        return normalized

    @field_validator("openrouter_embeddings_fallback_model")
    @classmethod
    def validate_openrouter_fallback_model(cls, value: str) -> str:
        normalized = value.strip()
        if normalized != "google/gemini-embedding-2":
            raise ValueError("openrouter fallback is restricted to google/gemini-embedding-2")
        return normalized

    @model_validator(mode="after")
    def reject_placeholders_outside_dev(self) -> "GeminiGatewaySettings":
        if self.openrouter_embeddings_fallback_enabled and self.openrouter_api_key is None:
            raise ValueError("openrouter fallback requires openrouter_api_key")

        secrets = [self.hmac_key, self.internal_auth_token, self.encryption_key]
        if self.openrouter_api_key is not None:
            secrets.append(self.openrouter_api_key)

        if self.environment == "development":
            return self

        for secret in secrets:
            if _PLACEHOLDER_SECRET_PATTERN.search(secret.get_secret_value()):
                raise ValueError("gateway secrets must not use placeholders outside development")
        return self


@lru_cache(maxsize=1)
def get_gateway_settings() -> GeminiGatewaySettings:
    """Возвращает кешированные настройки gateway."""

    return GeminiGatewaySettings()
