from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field


ProviderOperationType = Literal["chat", "embedding", "tts"]
ProviderPayloadKind = Literal["text", "media", "tts"]
ProviderTimeoutStage = Literal["request", "response_headers", "response_body", "unknown"]


class ProviderPayloadSummary(BaseModel):
    """Безопасная сводка payload без текста, base64 и секретов."""

    operation_type: ProviderOperationType
    payload_kind: ProviderPayloadKind
    media_count: int = Field(ge=0)
    image_count: int = Field(ge=0)


class ProviderTimingSummary(BaseModel):
    """Безопасные замеры HTTP-вызова провайдера."""

    operation_type: ProviderOperationType
    payload_kind: ProviderPayloadKind
    request_bytes: int = Field(ge=0)
    response_bytes: int | None = Field(default=None, ge=0)
    media_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    provider_total_ms: int = Field(ge=0)
    request_prepare_ms: int = Field(ge=0)
    # Включает ожидание сети, прокси и загрузки запроса до получения headers.
    response_headers_ms: int | None = Field(default=None, ge=0)
    response_body_ms: int | None = Field(default=None, ge=0)
    response_parse_ms: int | None = Field(default=None, ge=0)
    timeout_kind: str | None = None
    timeout_stage: ProviderTimeoutStage | None = None


@dataclass(frozen=True)
class TimedJsonResponse:
    response: httpx.Response
    payload: dict[str, Any] | None
    body_bytes: bytes
    timing: ProviderTimingSummary


async def send_timed_json(
    *,
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str] | None,
    payload: dict[str, Any],
    payload_summary: ProviderPayloadSummary,
    timeout_seconds: float,
) -> TimedJsonResponse:
    total_started_at = perf_counter()
    prepare_started_at = perf_counter()
    request_bytes = _json_bytes(payload)
    request = client.build_request(
        method,
        url,
        headers=headers,
        content=request_bytes,
        timeout=timeout_seconds,
    )
    request_prepare_ms = _elapsed_ms(prepare_started_at)

    response: httpx.Response | None = None
    response_headers_ms: int | None = None
    response_body_ms: int | None = None
    response_parse_ms: int | None = None

    try:
        headers_started_at = perf_counter()
        response = await client.send(request, stream=True)
        response_headers_ms = _elapsed_ms(headers_started_at)

        body_started_at = perf_counter()
        body_bytes = await response.aread()
        response_body_ms = _elapsed_ms(body_started_at)

        parse_started_at = perf_counter()
        response_payload = _json_object_from_bytes(body_bytes)
        response_parse_ms = _elapsed_ms(parse_started_at)
    except httpx.TimeoutException as exc:
        await _close_response_safely(response)
        timeout_stage = _resolve_timeout_stage(exc, response)
        timing = _build_timing_summary(
            payload_summary=payload_summary,
            request_bytes=len(request_bytes),
            response_bytes=None,
            provider_total_ms=_elapsed_ms(total_started_at),
            request_prepare_ms=request_prepare_ms,
            response_headers_ms=response_headers_ms,
            response_body_ms=response_body_ms,
            response_parse_ms=response_parse_ms,
            timeout_kind=_timeout_kind(exc),
            timeout_stage=timeout_stage,
        )
        attach_timing_to_error(exc, timing)
        raise
    except httpx.HTTPError as exc:
        await _close_response_safely(response)
        timing = _build_timing_summary(
            payload_summary=payload_summary,
            request_bytes=len(request_bytes),
            response_bytes=None,
            provider_total_ms=_elapsed_ms(total_started_at),
            request_prepare_ms=request_prepare_ms,
            response_headers_ms=response_headers_ms,
            response_body_ms=response_body_ms,
            response_parse_ms=response_parse_ms,
            timeout_kind=None,
            timeout_stage=None,
        )
        attach_timing_to_error(exc, timing)
        raise

    timing = _build_timing_summary(
        payload_summary=payload_summary,
        request_bytes=len(request_bytes),
        response_bytes=len(body_bytes),
        provider_total_ms=_elapsed_ms(total_started_at),
        request_prepare_ms=request_prepare_ms,
        response_headers_ms=response_headers_ms,
        response_body_ms=response_body_ms,
        response_parse_ms=response_parse_ms,
        timeout_kind=None,
        timeout_stage=None,
    )
    return TimedJsonResponse(
        response=response,
        payload=response_payload,
        body_bytes=body_bytes,
        timing=timing,
    )


def classify_chat_payload(payload: dict[str, Any]) -> ProviderPayloadSummary:
    media_count, image_count = _count_media(payload.get("messages"))
    return _build_payload_summary(
        operation_type="chat",
        media_count=media_count,
        image_count=image_count,
    )


def classify_embedding_payload(payload: dict[str, Any]) -> ProviderPayloadSummary:
    media_count, image_count = _count_media(payload)
    return _build_payload_summary(
        operation_type="embedding",
        media_count=media_count,
        image_count=image_count,
    )


def classify_tts_payload(payload: dict[str, Any]) -> ProviderPayloadSummary:
    return ProviderPayloadSummary(
        operation_type="tts",
        payload_kind="tts",
        media_count=0,
        image_count=0,
    )


def timing_to_dict(value: ProviderTimingSummary | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, ProviderTimingSummary):
        return _sanitize_timing_dict(value.model_dump())
    return _sanitize_timing_dict(value)


def provider_timing_columns(value: Any) -> dict[str, Any]:
    """Возвращает безопасные aggregate-поля provider timing для БД и логов."""

    timing = timing_to_dict(value)
    return {
        "operation_type": _optional_timing_string(timing.get("operation_type")),
        "payload_kind": _optional_timing_string(timing.get("payload_kind")),
        "request_bytes": _optional_non_negative_int(timing.get("request_bytes")),
        "response_bytes": _optional_non_negative_int(timing.get("response_bytes")),
        "media_count": _optional_non_negative_int(timing.get("media_count")),
        "image_count": _optional_non_negative_int(timing.get("image_count")),
        "provider_total_ms": _optional_non_negative_int(timing.get("provider_total_ms")),
        "request_prepare_ms": _optional_non_negative_int(timing.get("request_prepare_ms")),
        "response_headers_ms": _optional_non_negative_int(timing.get("response_headers_ms")),
        "response_body_ms": _optional_non_negative_int(timing.get("response_body_ms")),
        "response_parse_ms": _optional_non_negative_int(timing.get("response_parse_ms")),
        "timeout_kind": _optional_timing_string(timing.get("timeout_kind")),
        "timeout_stage": _optional_timing_string(timing.get("timeout_stage")),
    }


def attach_timing_to_error(error: Exception, timing: ProviderTimingSummary | None) -> None:
    if timing is None:
        return
    setattr(error, "provider_timing", timing)


async def _close_response_safely(response: httpx.Response | None) -> None:
    if response is None:
        return
    try:
        await response.aclose()
    except Exception:
        return


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _json_object_from_bytes(value: bytes) -> dict[str, Any] | None:
    try:
        parsed_value = json.loads(value)
    except ValueError:
        return None
    if isinstance(parsed_value, dict):
        return parsed_value
    return None


def _build_timing_summary(
    *,
    payload_summary: ProviderPayloadSummary,
    request_bytes: int,
    response_bytes: int | None,
    provider_total_ms: int,
    request_prepare_ms: int,
    response_headers_ms: int | None,
    response_body_ms: int | None,
    response_parse_ms: int | None,
    timeout_kind: str | None,
    timeout_stage: ProviderTimeoutStage | None,
) -> ProviderTimingSummary:
    return ProviderTimingSummary(
        operation_type=payload_summary.operation_type,
        payload_kind=payload_summary.payload_kind,
        request_bytes=request_bytes,
        response_bytes=response_bytes,
        media_count=payload_summary.media_count,
        image_count=payload_summary.image_count,
        provider_total_ms=provider_total_ms,
        request_prepare_ms=request_prepare_ms,
        response_headers_ms=response_headers_ms,
        response_body_ms=response_body_ms,
        response_parse_ms=response_parse_ms,
        timeout_kind=timeout_kind,
        timeout_stage=timeout_stage,
    )


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((perf_counter() - started_at) * 1000))


def _resolve_timeout_stage(
    error: httpx.TimeoutException,
    response: httpx.Response | None,
) -> ProviderTimeoutStage:
    if isinstance(error, httpx.ReadTimeout):
        return "response_body" if response is not None else "response_headers"
    if isinstance(error, (httpx.ConnectTimeout, httpx.PoolTimeout, httpx.WriteTimeout)):
        return "request"
    return "unknown"


def _timeout_kind(error: httpx.TimeoutException) -> str:
    if isinstance(error, httpx.ConnectTimeout):
        return "connect_timeout"
    if isinstance(error, httpx.ReadTimeout):
        return "read_timeout"
    if isinstance(error, httpx.WriteTimeout):
        return "write_timeout"
    if isinstance(error, httpx.PoolTimeout):
        return "pool_timeout"
    return "timeout"


def _sanitize_timing_dict(value: dict[str, Any]) -> dict[str, Any]:
    safe_value: dict[str, Any] = {}
    for field_name in ProviderTimingSummary.model_fields:
        if field_name not in value:
            continue
        safe_field_value = _sanitize_timing_field(field_name, value[field_name])
        if safe_field_value is _UNSAFE_FIELD:
            continue
        safe_value[field_name] = safe_field_value
    return safe_value


def _sanitize_timing_field(field_name: str, value: Any) -> Any:
    if field_name == "operation_type":
        return value if value in _SAFE_OPERATION_TYPES else _UNSAFE_FIELD
    if field_name == "payload_kind":
        return value if value in _SAFE_PAYLOAD_KINDS else _UNSAFE_FIELD
    if field_name in _NON_NEGATIVE_TIMING_FIELDS:
        return _sanitize_optional_non_negative_int(field_name, value)
    if field_name == "timeout_kind":
        return _sanitize_timeout_kind(value)
    if field_name == "timeout_stage":
        return value if value is None or value in _SAFE_TIMEOUT_STAGES else _UNSAFE_FIELD
    return _UNSAFE_FIELD


def _sanitize_optional_non_negative_int(field_name: str, value: Any) -> int | None | object:
    if value is None and field_name in _OPTIONAL_NON_NEGATIVE_TIMING_FIELDS:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return _UNSAFE_FIELD
    return value


def _sanitize_timeout_kind(value: Any) -> str | None | object:
    if value is None:
        return None
    if isinstance(value, str) and value in _SAFE_TIMEOUT_KINDS:
        return value
    if isinstance(value, str):
        return "timeout"
    return _UNSAFE_FIELD


def _optional_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_timing_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _build_payload_summary(
    *,
    operation_type: ProviderOperationType,
    media_count: int,
    image_count: int,
) -> ProviderPayloadSummary:
    return ProviderPayloadSummary(
        operation_type=operation_type,
        payload_kind="media" if media_count > 0 else "text",
        media_count=media_count,
        image_count=image_count,
    )


def _count_media(value: Any) -> tuple[int, int]:
    if isinstance(value, list):
        return _sum_media_counts(_count_media(item) for item in value)

    if not isinstance(value, dict):
        return 0, 0

    current_count = _count_current_media_part(value)
    nested_count = _sum_media_counts(
        _count_media(item) for key, item in value.items() if key not in _MEDIA_PAYLOAD_KEYS
    )
    return current_count[0] + nested_count[0], current_count[1] + nested_count[1]


def _sum_media_counts(counts: Any) -> tuple[int, int]:
    media_count = 0
    image_count = 0
    for media_item_count, image_item_count in counts:
        media_count += media_item_count
        image_count += image_item_count
    return media_count, image_count


def _count_current_media_part(value: dict[str, Any]) -> tuple[int, int]:
    mime_type = _extract_mime_type(value)
    part_type = value.get("type")
    has_image_url = "image_url" in value
    has_inline_data = "inlineData" in value or "inline_data" in value
    has_file_data = "fileData" in value or "file_data" in value

    if part_type == "image_url" or has_image_url or _is_image_mime_type(mime_type):
        return 1, 1
    if has_inline_data or has_file_data or isinstance(part_type, str) and part_type.startswith("media"):
        return 1, 0
    return 0, 0


def _extract_mime_type(value: dict[str, Any]) -> str | None:
    raw_mime_type = value.get("mimeType") or value.get("mime_type")
    if isinstance(raw_mime_type, str):
        return raw_mime_type

    for key in ("inlineData", "inline_data", "fileData", "file_data"):
        nested = value.get(key)
        if isinstance(nested, dict):
            nested_mime_type = nested.get("mimeType") or nested.get("mime_type")
            if isinstance(nested_mime_type, str):
                return nested_mime_type
    return None


def _is_image_mime_type(value: str | None) -> bool:
    return value is not None and value.lower().startswith("image/")


_MEDIA_PAYLOAD_KEYS = frozenset(
    {
        "fileData",
        "file_data",
        "image_url",
        "inlineData",
        "inline_data",
    }
)

_SAFE_TIMEOUT_KINDS = frozenset(
    {
        "connect_timeout",
        "pool_timeout",
        "read_timeout",
        "timeout",
        "write_timeout",
    }
)

_SAFE_OPERATION_TYPES = frozenset({"chat", "embedding", "tts"})
_SAFE_PAYLOAD_KINDS = frozenset({"text", "media", "tts"})
_SAFE_TIMEOUT_STAGES = frozenset({"request", "response_headers", "response_body", "unknown"})
_OPTIONAL_NON_NEGATIVE_TIMING_FIELDS = frozenset(
    {
        "response_body_ms",
        "response_bytes",
        "response_headers_ms",
        "response_parse_ms",
    }
)
_NON_NEGATIVE_TIMING_FIELDS = frozenset(
    {
        "image_count",
        "media_count",
        "provider_total_ms",
        "request_bytes",
        "request_prepare_ms",
    }
) | _OPTIONAL_NON_NEGATIVE_TIMING_FIELDS
_UNSAFE_FIELD = object()
