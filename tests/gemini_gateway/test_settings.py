from __future__ import annotations

import pytest
from cryptography.fernet import Fernet
from pydantic import ValidationError

from gemini_gateway.config import GeminiGatewaySettings


def test_gateway_settings_load_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")

    settings = GeminiGatewaySettings()

    assert settings.postgres_dsn.startswith("postgresql+asyncpg://")
    assert settings.internal_auth_token.get_secret_value() == "secret-token-value"


def test_gateway_settings_reject_placeholder_outside_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_ENVIRONMENT", "production")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "change-me")
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "change-me")

    with pytest.raises(ValidationError):
        GeminiGatewaySettings()


def test_gateway_settings_reject_example_placeholders_outside_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_ENVIRONMENT", "production")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "replace-with-random-string-min-32")
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "replace-with-internal-gateway-token-min-16")

    with pytest.raises(ValidationError):
        GeminiGatewaySettings()


def test_gateway_settings_reject_dev_local_token_outside_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_ENVIRONMENT", "production")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "change-me-local-token")

    with pytest.raises(ValidationError):
        GeminiGatewaySettings()


def test_gateway_settings_reject_wrong_dsn_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")

    with pytest.raises(ValidationError):
        GeminiGatewaySettings()
