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
    assert settings.max_route_attempts == 5


def test_gateway_settings_load_max_route_attempts_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_MAX_ROUTE_ATTEMPTS", "7")

    settings = GeminiGatewaySettings()

    assert settings.max_route_attempts == 7


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


def test_gateway_settings_load_openrouter_embedding_fallback_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_DIRECT_ONLY_ENABLED", "true")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_API_KEY", "sk-or-openrouter-secret-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/")

    settings = GeminiGatewaySettings()

    assert settings.openrouter_embeddings_fallback_enabled is True
    assert settings.openrouter_embeddings_direct_only_enabled is True
    assert settings.openrouter_api_key is not None
    assert settings.openrouter_api_key.get_secret_value() == "sk-or-openrouter-secret-value"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.openrouter_embeddings_fallback_model == "google/gemini-embedding-2"


def test_gateway_settings_reject_enabled_openrouter_fallback_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_API_KEY", "")

    with pytest.raises(ValidationError):
        GeminiGatewaySettings()


def test_gateway_settings_reject_enabled_openrouter_direct_only_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED", "false")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_DIRECT_ONLY_ENABLED", "true")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_API_KEY", "")

    with pytest.raises(ValidationError):
        GeminiGatewaySettings()


def test_gateway_settings_allows_blank_openrouter_key_when_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED", "false")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_DIRECT_ONLY_ENABLED", "false")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_API_KEY", "")

    settings = GeminiGatewaySettings()

    assert settings.openrouter_embeddings_fallback_enabled is False
    assert settings.openrouter_api_key is None


def test_gateway_settings_reject_openrouter_placeholder_outside_dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_ENVIRONMENT", "production")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_API_KEY", "replace-with-openrouter-key")

    with pytest.raises(ValidationError):
        GeminiGatewaySettings()


def test_gateway_settings_reject_unsupported_openrouter_fallback_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED", "false")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_DIRECT_ONLY_ENABLED", "false")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_API_KEY", "")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_MODEL", "other/model")

    with pytest.raises(ValidationError, match="openrouter fallback is restricted"):
        GeminiGatewaySettings()


def test_gateway_settings_reject_openrouter_base_url_without_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_API_KEY", "sk-or-openrouter-secret-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_BASE_URL", "https://?x=1")

    with pytest.raises(ValidationError):
        GeminiGatewaySettings()


def test_gateway_settings_load_openlux_chat_fallback_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_API_KEY", "sk-openlux-secret-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_BASE_URL", "https://api.openlux.ai/v1/")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_CHAT_MODE", "fallback")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_CHAT_MODEL", "gemini-3.7-flash")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_PRICING_GROUP", "Anti-Gemini-1")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_MAX_STREAM_BYTES", "262144")

    settings = GeminiGatewaySettings()

    assert settings.openlux_api_key is not None
    assert settings.openlux_api_key.get_secret_value() == "sk-openlux-secret-value"
    assert settings.openlux_base_url == "https://api.openlux.ai/v1"
    assert settings.openlux_chat_mode == "fallback"
    assert settings.openlux_chat_model == "gemini-3.7-flash"
    assert settings.openlux_pricing_group == "Anti-Gemini-1"
    assert settings.openlux_max_stream_bytes == 262144


def test_gateway_settings_load_openlux_embedding_direct_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_API_KEY", "sk-openlux-secret-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_EMBEDDINGS_DIRECT_ONLY_ENABLED", "true")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_EMBEDDINGS_MODEL", "gemini-embedding-2-preview")

    settings = GeminiGatewaySettings()

    assert settings.openlux_embeddings_direct_only_enabled is True
    assert settings.openlux_embeddings_model == "gemini-embedding-2-preview"


def test_gateway_settings_reject_enabled_openlux_embeddings_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_API_KEY", "")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_CHAT_MODE", "off")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_EMBEDDINGS_DIRECT_ONLY_ENABLED", "true")

    with pytest.raises(ValidationError, match="OpenLux features require openlux_api_key"):
        GeminiGatewaySettings()


def test_gateway_settings_reject_enabled_openlux_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_CHAT_MODE", "fallback")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_API_KEY", "")

    with pytest.raises(ValidationError, match="openlux chat requires openlux_api_key"):
        GeminiGatewaySettings()


def test_gateway_settings_rejects_openlux_model_other_than_gemini_3_7_flash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENLUX_CHAT_MODEL", "gpt-5.5-pro")

    with pytest.raises(ValidationError, match="openlux chat is restricted to gemini-3.7-flash"):
        GeminiGatewaySettings()
