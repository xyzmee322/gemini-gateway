from __future__ import annotations

from typing import Any

import httpx

from gemini_gateway.contracts import GatewayTTSRequest, GatewayTTSResponse
from gemini_gateway.errors import GatewayError
from gemini_gateway.gemini_client import (
    _raise_for_embedded_provider_error,
    _require_proxy_url,
    _timeout_error_kind,
    _transport_error_reason,
    _to_gemini_model_name,
)
from gemini_gateway.http_client_pool import GatewayHttpClientPool
from gemini_gateway.provider_http_errors import build_gateway_error_from_response
from gemini_gateway.value_extractors import first_int_value, first_string_value


class GeminiTTSClient:
    """HTTP-клиент native Gemini text-to-speech generateContent."""

    def __init__(
        self,
        *,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        client_pool: GatewayHttpClientPool | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport
        self._client_pool = client_pool

    async def synthesize(
        self,
        *,
        request: GatewayTTSRequest,
        api_key: str,
        proxy_url: str | None,
    ) -> GatewayTTSResponse:
        _require_proxy_url(proxy_url=proxy_url, request_id=request.request_id)
        payload = _tts_payload(request)
        try:
            if self._client_pool is not None:
                response = await self._client_pool.get(proxy_url=proxy_url).post(
                    f"{self._base_url}/models/{_to_gemini_model_name(request.model)}:generateContent",
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=request.timeout_seconds,
                )
            else:
                client_kwargs: dict[str, Any] = {"timeout": self._timeout, "trust_env": False}
                if self._transport is not None:
                    client_kwargs["transport"] = self._transport
                elif proxy_url is not None:
                    client_kwargs["proxy"] = proxy_url
                async with httpx.AsyncClient(**client_kwargs) as client:
                    response = await client.post(
                        f"{self._base_url}/models/{_to_gemini_model_name(request.model)}:generateContent",
                        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                        json=payload,
                        timeout=request.timeout_seconds,
                    )
        except httpx.ProxyError as exc:
            raise GatewayError(
                reason="proxy_failed",
                retryable=True,
                provider_message_safe=str(exc),
                request_id=request.request_id,
            ) from exc
        except httpx.TimeoutException as exc:
            raise GatewayError(
                reason="network_timeout",
                retryable=True,
                provider_message_safe=_timeout_error_kind(exc),
                request_id=request.request_id,
            ) from exc
        except httpx.HTTPError as exc:
            raise GatewayError(
                reason=_transport_error_reason(proxy_url),
                retryable=True,
                provider_message_safe=str(exc),
                request_id=request.request_id,
            ) from exc

        if response.status_code >= 400:
            raise build_gateway_error_from_response(
                response=response,
                request_id=request.request_id,
                supports_content_filter=False,
            )

        try:
            raw_response = response.json()
        except ValueError as exc:
            raise GatewayError(
                reason="invalid_response",
                retryable=False,
                provider_message_safe=str(exc),
                request_id=request.request_id,
                provider_called=True,
            ) from exc
        if not isinstance(raw_response, dict):
            raise GatewayError(
                reason="invalid_response",
                retryable=False,
                provider_message_safe="Gemini TTS response payload must be a JSON object",
                request_id=request.request_id,
                provider_called=True,
            )
        _raise_for_embedded_provider_error(raw_response=raw_response, request_id=request.request_id)
        return _to_gateway_tts_response(request=request, raw_response=raw_response)


def _tts_payload(request: GatewayTTSRequest) -> dict[str, Any]:
    return {
        "contents": [{"parts": [{"text": request.text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": request.voice_name,
                    }
                }
            },
        },
    }


def _to_gateway_tts_response(*, request: GatewayTTSRequest, raw_response: dict[str, Any]) -> GatewayTTSResponse:
    inline_data = _first_inline_data(raw_response)
    if inline_data is None:
        raise GatewayError(
            reason="invalid_response",
            retryable=False,
            provider_message_safe="Gemini TTS response does not contain inline audio",
            request_id=request.request_id,
            provider_called=True,
        )

    audio_base64 = first_string_value(inline_data, "data")
    if not audio_base64:
        raise GatewayError(
            reason="invalid_response",
            retryable=False,
            provider_message_safe="Gemini TTS inline audio is empty",
            request_id=request.request_id,
            provider_called=True,
        )

    return GatewayTTSResponse(
        request_id=request.request_id,
        generation_id=first_string_value(raw_response, "responseId", "response_id"),
        model=request.model,
        audio_base64=audio_base64,
        audio_mime_type=first_string_value(inline_data, "mimeType", "mime_type") or "audio/wav",
        usage=_usage_from_metadata(raw_response.get("usageMetadata") or raw_response.get("usage_metadata") or {}),
        finish_reason=_first_finish_reason(raw_response),
        raw_response=raw_response,
        provider_specific_fields=_provider_specific_fields(raw_response),
    )


def _first_inline_data(raw_response: dict[str, Any]) -> dict[str, Any] | None:
    candidates = raw_response.get("candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            inline_data = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline_data, dict):
                return inline_data
    return None


def _usage_from_metadata(metadata: Any) -> dict[str, int | None]:
    if not isinstance(metadata, dict):
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": first_int_value(metadata, "promptTokenCount", "prompt_token_count"),
        "completion_tokens": first_int_value(metadata, "candidatesTokenCount", "candidates_token_count"),
        "total_tokens": first_int_value(metadata, "totalTokenCount", "total_token_count"),
    }


def _provider_specific_fields(raw_response: dict[str, Any]) -> dict[str, Any]:
    del raw_response
    return {}


def _first_finish_reason(raw_response: dict[str, Any]) -> str | None:
    candidates = raw_response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    return first_string_value(first, "finishReason", "finish_reason")
