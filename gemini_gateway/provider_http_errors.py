from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import json

import httpx

from gemini_gateway.errors import GatewayError, reason_from_gemini_status

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def build_gateway_error_from_response(
    *,
    response: httpx.Response,
    request_id: str,
    supports_content_filter: bool,
) -> GatewayError:
    """Собирает безопасную gateway-ошибку из HTTP-ответа Gemini."""

    provider_message = extract_provider_message(response)
    return GatewayError(
        reason=reason_from_gemini_status(
            status_code=response.status_code,
            provider_message=provider_message,
            supports_content_filter=supports_content_filter,
        ),
        retryable=response.status_code in _RETRYABLE_STATUS_CODES,
        provider_status_code=response.status_code,
        provider_message_safe=provider_message,
        retry_after_seconds=parse_retry_after(response.headers.get("Retry-After")),
        request_id=request_id,
        status_code=response.status_code if response.status_code == 429 else None,
    )


def extract_provider_message(response: httpx.Response) -> str | None:
    """Достаёт короткое provider-сообщение без выброса raw JSON наружу."""

    try:
        payload = response.json()
    except json.JSONDecodeError:
        return response.text[:500] if response.text else None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("status") or error.get("code")
            return str(message) if message is not None else None
        if isinstance(error, str):
            return error
    return None


def parse_retry_after(value: str | None) -> int | None:
    """Парсит Retry-After seconds/date в неотрицательное число секунд."""

    if not value:
        return None
    stripped = value.strip()
    try:
        return max(0, int(stripped))
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0, round((retry_at - datetime.now(tz=UTC)).total_seconds()))
