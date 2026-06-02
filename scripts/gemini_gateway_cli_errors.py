from __future__ import annotations

from typing import Any

from pydantic import ValidationError

_GATEWAY_ENV_BY_SETTINGS_FIELD = {
    "postgres_dsn": "GEMINI_GATEWAY_POSTGRES_DSN",
    "postgres_sync_dsn": "GEMINI_GATEWAY_POSTGRES_SYNC_DSN",
    "encryption_key": "GEMINI_GATEWAY_ENCRYPTION_KEY",
    "hmac_key": "GEMINI_GATEWAY_HMAC_KEY",
    "internal_auth_token": "GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN",
}


def gateway_cli_error_payload(error: Exception, *, fallback_error: str) -> dict[str, Any]:
    """Строит безопасную JSON-ошибку для gateway diagnostic CLI."""

    missing_env = _missing_gateway_env_names(error)
    if missing_env:
        env_list = ", ".join(missing_env)
        return {
            "status": "error",
            "reason": "gateway_env_missing",
            "error": (
                f"{fallback_error} Не найдены настройки Gemini Gateway. "
                f"Запустите команду в контейнере gemini-gateway или задайте {env_list}."
            ),
            "missing_env": missing_env,
        }

    return {"status": "error", "reason": "gateway_cli_error", "error": fallback_error}


def _missing_gateway_env_names(error: Exception) -> list[str]:
    if not isinstance(error, ValidationError):
        return []

    missing_env: list[str] = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        if item.get("type") != "missing":
            continue
        location = item.get("loc")
        if not location:
            continue
        env_name = _GATEWAY_ENV_BY_SETTINGS_FIELD.get(str(location[0]))
        if env_name is not None:
            missing_env.append(env_name)

    return sorted(set(missing_env))
