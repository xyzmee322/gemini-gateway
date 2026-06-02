from __future__ import annotations

from functools import lru_cache
import re
from typing import Literal

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

    @model_validator(mode="after")
    def reject_placeholders_outside_dev(self) -> "GeminiGatewaySettings":
        if self.environment == "development":
            return self

        secrets = (self.hmac_key, self.internal_auth_token, self.encryption_key)
        for secret in secrets:
            if _PLACEHOLDER_SECRET_PATTERN.search(secret.get_secret_value()):
                raise ValueError("gateway secrets must not use placeholders outside development")
        return self


@lru_cache(maxsize=1)
def get_gateway_settings() -> GeminiGatewaySettings:
    """Возвращает кешированные настройки gateway."""

    return GeminiGatewaySettings()
