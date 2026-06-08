from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from core.error_taxonomy import COMMON_PUBLIC_MESSAGES, retryable_by_default
from core.number_parsing import parse_optional_int as _optional_int, parse_positive_int as _optional_positive_int
from core.secret_redaction import redact_secrets_in_text
from core.string_parsing import parse_optional_string as _optional_string
from gemini_gateway.contracts import GatewayErrorReason, GatewayErrorResponse

_PUBLIC_MESSAGES: dict[GatewayErrorReason, str] = {
    "no_route": "Сейчас нет доступного маршрута для Gemini, попробуй позже",
    "cooldown_active": "Маршруты Gemini временно охлаждаются, попробуй позже",
    "rate_limited": COMMON_PUBLIC_MESSAGES["rate_limited"],
    "quota_exhausted": "Квота Gemini временно исчерпана, попробуй позже",
    "auth_failed": COMMON_PUBLIC_MESSAGES["auth_failed"],
    "proxy_failed": COMMON_PUBLIC_MESSAGES["provider_unavailable"],
    "network_timeout": COMMON_PUBLIC_MESSAGES["network_timeout"],
    "provider_unavailable": COMMON_PUBLIC_MESSAGES["provider_unavailable"],
    "content_filtered": COMMON_PUBLIC_MESSAGES["content_filtered"],
    "invalid_response": COMMON_PUBLIC_MESSAGES["invalid_response"],
    "request_failed": "Не удалось обработать запрос",
    "unauthorized": "Недостаточно прав для выполнения запроса",
    "bad_request": "Некорректный запрос",
}

_GATEWAY_RETRYABLE_OVERRIDES = frozenset({"no_route", "cooldown_active"})
_WHITESPACE_PATTERN = re.compile(r"\s+")
_PUBLIC_PROVIDER_REASON_MARKERS = (
    "content_filter",
    "content filtered",
    "content_filtered",
    "prohibited",
    "safety",
)
_STABLE_PROVIDER_REASONS = frozenset(
    {
        "connect_timeout",
        "read_timeout",
        "write_timeout",
        "pool_timeout",
        "timeout",
    }
)
_AUTH_ERROR_MARKERS = (
    "api key",
    "api_key",
    "api-key",
    "unauthorized",
    "invalid credential",
    "invalid credentials",
)
_MAX_PUBLIC_PROVIDER_REASON_LENGTH = 160


class GatewayError(RuntimeError):
    """Доменная ошибка gateway с безопасным публичным текстом."""

    def __init__(
        self,
        *,
        reason: GatewayErrorReason,
        request_id: str,
        public_message: str | None = None,
        retryable: bool | None = None,
        provider_status_code: int | None = None,
        provider_code: int | None = None,
        provider_message_safe: str | None = None,
        provider_message: str | None = None,
        retry_after_seconds: int | None = None,
        provider_called: bool | None = None,
        status_code: int | None = None,
        error_code: str | None = None,
        quota_scope: str | None = None,
        quota_reset_at: Any | None = None,
        eligible_routes_count: int | None = None,
        exhausted_routes_count: int | None = None,
        disabled_routes_count: int | None = None,
        cooldown_scope: str | None = None,
        cooldown_level: int | None = None,
        sleep_until: Any | None = None,
        route_label: str | None = None,
        project_label: str | None = None,
        key_label: str | None = None,
        proxy_label: str | None = None,
        transport_mode: str | None = None,
    ) -> None:
        self.reason = reason
        self.request_id = request_id
        self.public_message = public_message or public_message_for_reason(reason)
        self.retryable = retryable if retryable is not None else _is_retryable_reason(reason)
        self.provider_status_code = provider_status_code if provider_status_code is not None else provider_code
        self.provider_code = self.provider_status_code
        self.provider_message_safe = sanitize_gateway_error_message(provider_message_safe or provider_message)
        self.provider_message = self.provider_message_safe
        self.retry_after_seconds = _optional_positive_int(retry_after_seconds)
        self.provider_called = provider_called if provider_called is not None else self.provider_status_code is not None
        self.status_code = status_code or status_code_for_reason(reason, self.provider_status_code)
        self.error_code = _optional_string(error_code) or _default_error_code(reason)
        self.quota_scope = quota_scope if quota_scope in {"minute", "day"} else None
        self.quota_reset_at = quota_reset_at
        self.eligible_routes_count = _optional_non_negative_int(eligible_routes_count)
        self.exhausted_routes_count = _optional_non_negative_int(exhausted_routes_count)
        self.disabled_routes_count = _optional_non_negative_int(disabled_routes_count)
        self.cooldown_scope = cooldown_scope
        self.cooldown_level = _optional_non_negative_int(cooldown_level)
        self.sleep_until = sleep_until
        self.route_label = _optional_string(route_label)
        self.project_label = _optional_string(project_label)
        self.key_label = _optional_string(key_label)
        self.proxy_label = _optional_string(proxy_label)
        self.transport_mode = _optional_string(transport_mode)
        super().__init__(f"{reason}: {self.public_message}")

    def to_response(self) -> GatewayErrorResponse:
        return GatewayErrorResponse(
            request_id=self.request_id,
            error=self.public_message,
            reason=self.reason,
            error_code=self.error_code,
            retryable=self.retryable,
            retry_after_seconds=self.retry_after_seconds,
            quota_scope=self.quota_scope,
            quota_reset_at=_serialize_gateway_time(self.quota_reset_at),
            eligible_routes_count=self.eligible_routes_count,
            exhausted_routes_count=self.exhausted_routes_count,
            disabled_routes_count=self.disabled_routes_count,
            cooldown_scope=self.cooldown_scope,
            cooldown_level=self.cooldown_level,
            sleep_until=_serialize_gateway_time(self.sleep_until),
            provider_reason=public_provider_reason(self.provider_message_safe),
            provider_status_code=self.provider_status_code,
            route_label=self.route_label,
            project_label=self.project_label,
            key_label=self.key_label,
            proxy_label=self.proxy_label,
            transport_mode=self.transport_mode,
        )


def public_message_for_reason(reason: GatewayErrorReason | str) -> str:
    return _PUBLIC_MESSAGES.get(reason, _PUBLIC_MESSAGES["request_failed"])


def _is_retryable_reason(reason: GatewayErrorReason | str) -> bool:
    return reason in _GATEWAY_RETRYABLE_OVERRIDES or retryable_by_default(str(reason))


def status_code_for_reason(reason: GatewayErrorReason | str, provider_code: int | None = None) -> int:
    if reason == "unauthorized":
        return 401
    if reason == "bad_request":
        return 400
    if reason == "invalid_response":
        return 502
    if reason in {"rate_limited", "quota_exhausted", "cooldown_active", "no_route"}:
        return 429
    if reason == "content_filtered":
        return 400
    if provider_code in {408, 429, 500, 502, 503, 504}:
        return provider_code
    return 503


def classify_upstream_error(
    *,
    request_id: str,
    status_code: int | None,
    message: str | None,
    retry_after_seconds: int | None = None,
) -> GatewayError:
    text = (message or "").lower()
    reason: GatewayErrorReason
    retryable: bool | None = None
    if status_code in {401, 403} or _looks_like_auth_error(text):
        reason = "auth_failed"
        retryable = False
    elif status_code == 429:
        reason = "quota_exhausted" if "quota" in text else "rate_limited"
    elif status_code is not None and status_code >= 500:
        reason = "provider_unavailable"
    elif "safety" in text or "content_filter" in text:
        reason = "content_filtered"
        retryable = False
    elif status_code == 400:
        reason = "invalid_response"
        retryable = False
    else:
        reason = "request_failed"
    return GatewayError(
        reason=reason,
        request_id=request_id,
        retryable=retryable,
        provider_status_code=status_code,
        provider_message_safe=sanitize_gateway_error_message(message),
        retry_after_seconds=retry_after_seconds,
    )


def reason_from_gemini_status(
    *,
    status_code: int,
    provider_message: str | None,
    supports_content_filter: bool = True,
) -> GatewayErrorReason:
    """Единая классификация HTTP-ошибок Gemini без cooldown на malformed 400."""

    message = (provider_message or "").lower()
    if status_code == 429:
        return "rate_limited"
    if status_code in {401, 403} or _looks_like_auth_error(message):
        return "auth_failed"
    if status_code == 400:
        if supports_content_filter and _looks_like_content_filter(message):
            return "content_filtered"
        return "invalid_response"
    if status_code in {408, 504}:
        return "network_timeout"
    if status_code >= 500:
        return "provider_unavailable"
    return "invalid_response"


def safe_error_payload(request_id: str, error: Exception) -> dict[str, Any]:
    if isinstance(error, GatewayError):
        return error.to_response().model_dump(exclude_none=True)
    return GatewayError(reason="request_failed", request_id=request_id).to_response().model_dump(exclude_none=True)


def public_provider_reason(message: str | None) -> str | None:
    """Возвращает стабильную безопасную причину без эха provider-текста."""

    safe_message = sanitize_gateway_error_message(message)
    if not safe_message or len(safe_message) > _MAX_PUBLIC_PROVIDER_REASON_LENGTH:
        return None
    normalized = safe_message.lower()
    if normalized in _STABLE_PROVIDER_REASONS:
        return normalized
    if any(marker in normalized for marker in _PUBLIC_PROVIDER_REASON_MARKERS):
        return "content_filtered"
    return None


def sanitize_gateway_error_message(message: str | None, *, max_length: int = 500) -> str | None:
    """Очищает техническую причину перед БД и логами."""

    if not message:
        return None

    safe_message = _WHITESPACE_PATTERN.sub(" ", message).strip()
    safe_message = redact_secrets_in_text(safe_message)
    if len(safe_message) <= max_length:
        return safe_message
    return safe_message[: max_length - 3].rstrip() + "..."


def _looks_like_content_filter(message: str) -> bool:
    markers = ("safety", "content_filter", "content filter", "blocked", "prohibited")
    return any(marker in message for marker in markers)


def _looks_like_auth_error(message: str) -> bool:
    return any(marker in message for marker in _AUTH_ERROR_MARKERS)


def _optional_non_negative_int(value: Any) -> int | None:
    parsed = _optional_int(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _default_error_code(reason: GatewayErrorReason | str) -> str | None:
    if reason in {"quota_exhausted", "cooldown_active"}:
        return str(reason)
    return None


def _serialize_gateway_time(value: Any) -> str | None:
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        text = aware.astimezone(UTC).isoformat()
        return text.replace("+00:00", "Z")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
