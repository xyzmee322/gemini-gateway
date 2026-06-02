from __future__ import annotations

COMMON_PUBLIC_MESSAGES: dict[str, str] = {
    "rate_limited": "Слишком много запросов, попробуй чуть позже",
    "auth_failed": "Сервис временно недоступен",
    "no_route": "Сервис временно недоступен",
    "network_timeout": "Сервис отвечает слишком долго, попробуй позже",
    "provider_unavailable": "Сервис временно недоступен",
    "proxy_failed": "Сервис временно недоступен",
    "content_filtered": "Не могу ответить на этот запрос",
    "invalid_response": "Не получилось обработать ответ модели",
    "request_failed": "Что-то пошло не так, попробуй позже",
}
COMMON_RETRYABLE_REASONS = frozenset(
    {
        "rate_limited",
        "quota_exhausted",
        "no_route",
        "provider_unavailable",
        "proxy_failed",
        "network_timeout",
        "request_failed",
    }
)


def public_message_for_reason(reason: str, *, overrides: dict[str, str] | None = None) -> str:
    """Возвращает стабильный публичный текст без технических деталей."""
    if overrides is not None and reason in overrides:
        return overrides[reason]
    return COMMON_PUBLIC_MESSAGES.get(reason, COMMON_PUBLIC_MESSAGES["request_failed"])


def retryable_by_default(reason: str) -> bool:
    """Единый дефолт retryability для observability и gateway/LLM ошибок."""
    return reason in COMMON_RETRYABLE_REASONS
