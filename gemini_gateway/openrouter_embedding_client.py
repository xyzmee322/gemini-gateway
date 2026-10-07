from __future__ import annotations

import math
import re
from typing import Any

import httpx

from gemini_gateway.contracts import GatewayEmbeddingInputPart, GatewayEmbeddingRequest, GatewayEmbeddingResponse
from gemini_gateway.contracts import GatewayErrorReason
from gemini_gateway.errors import GatewayError
from gemini_gateway.gemini_client import _timeout_error_kind
from gemini_gateway.provider_http_errors import extract_provider_message, parse_retry_after
from gemini_gateway.value_extractors import first_int_value, first_string_value

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 524, 529})
_CONTENT_FILTER_MARKERS = (
    "safety",
    "moderation",
    "flagged",
    "content_filter",
    "content filter",
    "blocked",
    "prohibited",
)
_MIME_TYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_AUDIO_FORMATS = {
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/wav": "wav",
    "audio/webm": "webm",
    "audio/x-wav": "wav",
}
_FILE_EXTENSIONS = {
    "application/json": "json",
    "application/octet-stream": "bin",
    "application/pdf": "pdf",
    "text/csv": "csv",
    "text/html": "html",
    "text/markdown": "md",
    "text/plain": "txt",
}


def _data_url_mime_type(url: str) -> str | None:
    normalized_url = url.strip()
    if not normalized_url.lower().startswith("data:"):
        return None
    metadata = normalized_url[5:].split(",", maxsplit=1)[0]
    mime_type = metadata.split(";", maxsplit=1)[0].strip().lower()
    return mime_type or "image/jpeg"


class OpenRouterEmbeddingClient:
    """HTTP-клиент OpenRouter embeddings без proxy и service wiring."""

    def __init__(
        self,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport

    async def embed(
        self,
        *,
        request: GatewayEmbeddingRequest,
        api_key: str,
    ) -> GatewayEmbeddingResponse:
        payload = _embedding_payload(request)
        try:
            client_kwargs: dict[str, Any] = {"timeout": self._timeout, "trust_env": False}
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=request.timeout_seconds,
                )
        except httpx.TimeoutException as exc:
            raise GatewayError(
                reason="network_timeout",
                retryable=True,
                provider_message_safe=_timeout_error_kind(exc),
                request_id=request.request_id,
            ) from exc
        except httpx.HTTPError as exc:
            raise GatewayError(
                reason="provider_unavailable",
                retryable=True,
                provider_message_safe="transport_error",
                request_id=request.request_id,
            ) from exc

        if response.status_code >= 400:
            provider_message = extract_provider_message(response)
            raise GatewayError(
                reason=_openrouter_error_reason(
                    status_code=response.status_code,
                    provider_message=provider_message,
                ),
                retryable=response.status_code in _RETRYABLE_STATUS_CODES,
                provider_status_code=response.status_code,
                provider_message_safe=provider_message,
                retry_after_seconds=parse_retry_after(response.headers.get("Retry-After")),
                request_id=request.request_id,
            )

        try:
            raw_response = response.json()
        except ValueError as exc:
            raise GatewayError(
                reason="invalid_response",
                retryable=False,
                provider_message_safe="OpenRouter embedding response must be JSON",
                request_id=request.request_id,
                provider_called=True,
            ) from exc
        if not isinstance(raw_response, dict):
            raise GatewayError(
                reason="invalid_response",
                retryable=False,
                provider_message_safe="OpenRouter embedding response payload must be a JSON object",
                request_id=request.request_id,
                provider_called=True,
            )
        _raise_for_embedded_error(raw_response, request_id=request.request_id)

        return parse_openai_embedding_response(request=request, raw_response=raw_response)


def _embedding_payload(request: GatewayEmbeddingRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model,
        "input": [{"content": [_content_part(part, request_id=request.request_id) for part in request.input]}],
        "dimensions": request.dimensions,
        "encoding_format": "float",
    }
    if request.chat_id is not None:
        payload["user"] = str(request.chat_id)
    return payload


def _content_part(part: GatewayEmbeddingInputPart, *, request_id: str) -> dict[str, Any]:
    if part.type == "text":
        return {"type": "text", "text": part.text}
    if part.type == "inline_data":
        return _inline_data_content_part(part.inline_data, request_id=request_id)
    image_url = part.image_url
    if not isinstance(image_url, dict):
        raise GatewayError(
            reason="bad_request",
            retryable=False,
            provider_message_safe="image_url part requires object payload",
            request_id=request_id,
            provider_called=False,
        )
    url = str(image_url.get("url") or "").strip()
    mime_type = _data_url_mime_type(url)
    if mime_type is not None and not mime_type.startswith("image/"):
        raise GatewayError(
            reason="bad_request",
            retryable=False,
            provider_message_safe="image_url embedding part requires image data URL",
            request_id=request_id,
            provider_called=False,
        )
    return {"type": "image_url", "image_url": image_url}


def _inline_data_content_part(inline_data: Any, *, request_id: str) -> dict[str, Any]:
    if not isinstance(inline_data, dict):
        raise _invalid_inline_data_error(request_id)

    mime_type = str(inline_data.get("mime_type") or "").strip().lower()
    data = str(inline_data.get("data") or "").strip()
    if not _MIME_TYPE_PATTERN.fullmatch(mime_type) or not data:
        raise _invalid_inline_data_error(request_id)

    data_url = f"data:{mime_type};base64,{data}"
    if mime_type.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": data_url}}
    if mime_type.startswith("audio/"):
        return {
            "type": "input_audio",
            "input_audio": {
                "data": data_url,
                "format": _audio_format(mime_type),
            },
        }
    if mime_type.startswith("video/"):
        return {"type": "video_url", "video_url": {"url": data_url}}
    return {
        "type": "file",
        "file": {
            "filename": f"embedding-input.{_file_extension(mime_type)}",
            "file_data": data_url,
        },
    }


def _audio_format(mime_type: str) -> str:
    return _AUDIO_FORMATS.get(mime_type, _safe_mime_subtype(mime_type, default="audio"))


def _file_extension(mime_type: str) -> str:
    return _FILE_EXTENSIONS.get(mime_type, _safe_mime_subtype(mime_type, default="bin"))


def _safe_mime_subtype(mime_type: str, *, default: str) -> str:
    subtype = mime_type.partition("/")[2].removeprefix("x-").split("+", maxsplit=1)[0]
    normalized = re.sub(r"[^a-z0-9]", "", subtype)
    return normalized or default


def _invalid_inline_data_error(request_id: str) -> GatewayError:
    return GatewayError(
        reason="bad_request",
        retryable=False,
        provider_message_safe="inline_data embedding part requires valid MIME type and base64 data",
        request_id=request_id,
        provider_called=False,
    )


def _raise_for_embedded_error(raw_response: dict[str, Any], *, request_id: str) -> None:
    error_payload = raw_response.get("error")
    if not isinstance(error_payload, dict):
        return

    status_code = first_int_value(error_payload, "code", "status_code") or 502
    if status_code < 400 or status_code > 599:
        status_code = 502
    provider_message = first_string_value(error_payload, "message", "detail", "status")
    raise GatewayError(
        reason=_openrouter_error_reason(
            status_code=status_code,
            provider_message=provider_message,
        ),
        retryable=status_code in _RETRYABLE_STATUS_CODES,
        provider_status_code=status_code,
        provider_message_safe=provider_message,
        request_id=request_id,
        provider_called=True,
    )


def parse_openai_embedding_response(
    *,
    request: GatewayEmbeddingRequest,
    raw_response: dict[str, Any],
    provider_name: str = "OpenRouter",
) -> GatewayEmbeddingResponse:
    values = _embedding_values(raw_response)
    if not values:
        raise GatewayError(
            reason="invalid_response",
            retryable=False,
            provider_message_safe=f"{provider_name} embedding response does not contain embedding values",
            request_id=request.request_id,
            provider_called=True,
        )
    if len(values) != request.dimensions:
        raise GatewayError(
            reason="invalid_response",
            retryable=False,
            provider_message_safe=f"{provider_name} embedding response dimensions do not match requested dimensions",
            request_id=request.request_id,
            provider_called=True,
        )

    return GatewayEmbeddingResponse(
        request_id=request.request_id,
        generation_id=first_string_value(raw_response, "id", "response_id"),
        model=first_string_value(raw_response, "model") or request.model,
        embedding=values,
        dimensions=len(values),
        usage=_usage_from_response(raw_response.get("usage")),
        raw_response=raw_response,
        provider_specific_fields={},
    )


def _embedding_values(raw_response: dict[str, Any]) -> list[float] | None:
    data = raw_response.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    embedding = first.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        return None

    values: list[float] = []
    for value in embedding:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        normalized_value = float(value)
        if not math.isfinite(normalized_value):
            return None
        values.append(normalized_value)
    return values


def _usage_from_response(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}

    result: dict[str, int] = {}
    for output_key, aliases in {
        "prompt_tokens": ("prompt_tokens", "promptTokens", "prompt_token_count"),
        "completion_tokens": ("completion_tokens", "completionTokens", "candidates_token_count"),
        "total_tokens": ("total_tokens", "totalTokens", "total_token_count"),
    }.items():
        value = first_int_value(usage, *aliases)
        if value is not None:
            result[output_key] = value
    return result


def _openrouter_error_reason(*, status_code: int, provider_message: str | None) -> GatewayErrorReason:
    message = (provider_message or "").lower()
    if status_code == 402:
        return "quota_exhausted"
    if status_code == 429:
        return "rate_limited"
    if status_code == 403 and any(marker in message for marker in _CONTENT_FILTER_MARKERS):
        return "content_filtered"
    if status_code in {401, 403}:
        return "auth_failed"
    if status_code in {408, 504, 524}:
        return "network_timeout"
    if status_code >= 500:
        return "provider_unavailable"
    return "invalid_response"
