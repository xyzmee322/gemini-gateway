from __future__ import annotations

import re
from typing import Any

import httpx

from gemini_gateway.contracts import GatewayEmbeddingRequest, GatewayEmbeddingResponse
from gemini_gateway.errors import GatewayError
from gemini_gateway.gemini_client import (
    _raise_for_embedded_provider_error,
    _require_proxy_url,
    _timeout_error_kind,
    _to_gemini_model_name,
    _transport_error_reason,
)
from gemini_gateway.http_client_pool import GatewayHttpClientPool
from gemini_gateway.provider_http_errors import build_gateway_error_from_response
from gemini_gateway.value_extractors import first_int_value, first_string_value

_DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[^;,]+)?(?:;base64)?,(?P<data>.+)$", re.DOTALL)


class GeminiEmbeddingClient:
    """HTTP-клиент native Gemini embedContent через обязательный proxy route."""

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

    async def embed(
        self,
        *,
        request: GatewayEmbeddingRequest,
        api_key: str,
        proxy_url: str | None,
    ) -> GatewayEmbeddingResponse:
        _require_proxy_url(proxy_url=proxy_url, request_id=request.request_id)
        payload = _embedding_payload(request)
        try:
            if self._client_pool is not None:
                response = await self._client_pool.get(proxy_url=proxy_url).post(
                    f"{self._base_url}/models/{_to_gemini_model_name(request.model)}:embedContent",
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
                        f"{self._base_url}/models/{_to_gemini_model_name(request.model)}:embedContent",
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
                supports_content_filter=True,
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
                provider_message_safe="Gemini embedding response payload must be a JSON object",
                request_id=request.request_id,
                provider_called=True,
            )
        _raise_for_embedded_provider_error(raw_response=raw_response, request_id=request.request_id)
        return _to_gateway_embedding_response(request=request, raw_response=raw_response)


def _embedding_payload(request: GatewayEmbeddingRequest) -> dict[str, Any]:
    return {
        "content": {
            "parts": [
                _native_part(part.model_dump(mode="python"), request_id=request.request_id) for part in request.input
            ]
        },
        "outputDimensionality": request.dimensions,
    }


def _native_part(part: dict[str, Any], *, request_id: str) -> dict[str, Any]:
    if part.get("type") == "text":
        return {"text": str(part.get("text") or "").strip()}
    return _image_part(part.get("image_url"), request_id=request_id)


def _image_part(image_url: Any, *, request_id: str) -> dict[str, Any]:
    if not isinstance(image_url, dict):
        raise GatewayError(
            reason="bad_request",
            retryable=False,
            provider_message_safe="image_url part requires object payload",
            request_id=request_id,
            provider_called=False,
        )
    url = str(image_url.get("url") or "").strip()
    data_url = _parse_data_url(url)
    if data_url is not None:
        return {"inlineData": data_url}

    raise GatewayError(
        reason="bad_request",
        retryable=False,
        provider_message_safe="image_url embedding parts currently require data URLs",
        request_id=request_id,
        provider_called=False,
    )


def _parse_data_url(url: str) -> dict[str, str] | None:
    match = _DATA_URL_PATTERN.match(url)
    if match is None:
        return None
    data = match.group("data").strip()
    if not data:
        return None
    return {
        "mimeType": match.group("mime") or "image/jpeg",
        "data": data,
    }


def _to_gateway_embedding_response(
    *,
    request: GatewayEmbeddingRequest,
    raw_response: dict[str, Any],
) -> GatewayEmbeddingResponse:
    values = _embedding_values(raw_response)
    if not values:
        raise GatewayError(
            reason="invalid_response",
            retryable=False,
            provider_message_safe="Gemini embedding response does not contain embedding values",
            request_id=request.request_id,
            provider_called=True,
        )
    if len(values) != request.dimensions:
        raise GatewayError(
            reason="invalid_response",
            retryable=False,
            provider_message_safe="Gemini embedding response dimensions do not match requested dimensions",
            request_id=request.request_id,
            provider_called=True,
        )

    return GatewayEmbeddingResponse(
        request_id=request.request_id,
        generation_id=first_string_value(raw_response, "responseId", "response_id"),
        model=request.model,
        embedding=values,
        dimensions=len(values),
        usage=_usage_from_metadata(raw_response.get("usageMetadata") or raw_response.get("usage_metadata") or {}),
        raw_response=raw_response,
        provider_specific_fields={},
    )


def _embedding_values(raw_response: dict[str, Any]) -> list[float] | None:
    embedding = raw_response.get("embedding")
    if not isinstance(embedding, dict):
        return None
    values = embedding.get("values")
    if not isinstance(values, list) or not values:
        return None
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        normalized.append(float(value))
    return normalized


def _usage_from_metadata(metadata: Any) -> dict[str, int | None]:
    if not isinstance(metadata, dict):
        return {"total_tokens": None}
    return {"total_tokens": first_int_value(metadata, "totalTokenCount", "total_token_count")}
