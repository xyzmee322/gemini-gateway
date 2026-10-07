from __future__ import annotations

from typing import Any

import httpx

from gemini_gateway.contracts import GatewayEmbeddingRequest, GatewayEmbeddingResponse
from gemini_gateway.embedding_client import (
    build_native_embedding_payload,
    parse_native_embedding_response,
)
from gemini_gateway.errors import GatewayError
from gemini_gateway.gemini_client import (
    _attach_provider_timing_to_gateway_error,
    _provider_timing_from_exception,
    _timeout_error_kind,
)
from gemini_gateway.openrouter_embedding_client import (
    parse_openai_embedding_response,
)
from gemini_gateway.provider_http_errors import (
    build_gateway_error_from_response,
    extract_provider_message,
    parse_retry_after,
)
from gemini_gateway.provider_observability import (
    classify_embedding_payload,
    send_timed_json,
    timing_to_dict,
)
from gemini_gateway.value_extractors import first_int_value

_PROVIDER_REQUEST_ID_HEADERS = ("x-api-request-id", "x-request-id", "request-id")


class OpenLuxEmbeddingClient:
    """Прямой OpenLux-клиент для text и native media embeddings."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.openlux.ai/v1",
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._native_base_url = self._base_url.removesuffix("/v1")
        self._timeout = timeout
        self._transport = transport

    async def embed(
        self,
        *,
        request: GatewayEmbeddingRequest,
        api_key: str,
        model: str,
    ) -> GatewayEmbeddingResponse:
        compatible = _supports_compatible_dimensions(request)
        payload = (
            _compatible_text_payload(request=request, model=model)
            if compatible
            else build_native_embedding_payload(request)
        )
        url = (
            f"{self._base_url}/embeddings"
            if compatible
            else f"{self._native_base_url}/v1beta/models/{model}:embedContent"
        )
        client_kwargs: dict[str, Any] = {"timeout": self._timeout, "trust_env": False}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                result = await send_timed_json(
                    client=client,
                    method="POST",
                    url=url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    payload=payload,
                    payload_summary=classify_embedding_payload(payload),
                    timeout_seconds=request.timeout_seconds,
                )
        except httpx.TimeoutException as exc:
            raise GatewayError(
                reason="network_timeout",
                retryable=True,
                provider_message_safe=_timeout_error_kind(exc),
                request_id=request.request_id,
                provider_called=True,
                provider_timing=timing_to_dict(_provider_timing_from_exception(exc)),
            ) from exc
        except httpx.HTTPError as exc:
            raise GatewayError(
                reason="provider_unavailable",
                retryable=True,
                provider_message_safe="transport_error",
                request_id=request.request_id,
                provider_called=True,
                provider_timing=timing_to_dict(_provider_timing_from_exception(exc)),
            ) from exc

        response = result.response
        provider_timing = timing_to_dict(result.timing)
        if response.status_code >= 400:
            error = _openlux_embedding_http_error(response=response, request_id=request.request_id)
            _attach_provider_timing_to_gateway_error(error, provider_timing)
            raise error

        raw_response = result.payload
        if raw_response is None:
            raise GatewayError(
                reason="invalid_response",
                retryable=True,
                provider_message_safe="OpenLux embedding response must be a JSON object",
                request_id=request.request_id,
                provider_called=True,
                provider_timing=provider_timing,
            )

        try:
            parsed = (
                parse_openai_embedding_response(
                    request=request,
                    raw_response=raw_response,
                    provider_name="OpenLux",
                )
                if compatible
                else parse_native_embedding_response(
                    request=request,
                    raw_response=raw_response,
                    provider_timing=provider_timing,
                )
            )
        except GatewayError as error:
            error.retryable = error.reason == "invalid_response"
            _attach_provider_timing_to_gateway_error(error, provider_timing)
            raise

        updates: dict[str, Any] = {
            "model": request.model,
            "provider_request_id": _provider_request_id(response.headers),
            "provider_timing": provider_timing,
        }
        if not compatible:
            updates["usage"] = _native_usage(raw_response)
        return parsed.model_copy(update=updates)


def _supports_compatible_dimensions(request: GatewayEmbeddingRequest) -> bool:
    """OpenLux compatible endpoint игнорирует dimensions и выдаёт только 3072."""
    return request.dimensions == 3072 and len(request.input) == 1 and request.input[0].type == "text"


def _compatible_text_payload(*, request: GatewayEmbeddingRequest, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": request.input[0].text,
        "dimensions": request.dimensions,
        "encoding_format": "float",
    }


def _native_usage(raw_response: dict[str, Any]) -> dict[str, int]:
    metadata = raw_response.get("usageMetadata") or raw_response.get("usage_metadata")
    if not isinstance(metadata, dict):
        return {}
    prompt_tokens = first_int_value(metadata, "promptTokenCount", "prompt_token_count")
    total_tokens = first_int_value(metadata, "totalTokenCount", "total_token_count") or prompt_tokens
    usage: dict[str, int] = {}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens
    return usage


def _provider_request_id(headers: httpx.Headers) -> str | None:
    for header_name in _PROVIDER_REQUEST_ID_HEADERS:
        value = headers.get(header_name)
        if value and value.strip():
            return value.strip()[:160]
    return None


def _openlux_embedding_http_error(*, response: httpx.Response, request_id: str) -> GatewayError:
    provider_message = extract_provider_message(response)
    normalized_message = (provider_message or "").lower()
    if response.status_code == 402 or (
        response.status_code == 403
        and any(marker in normalized_message for marker in ("quota", "balance", "insufficient"))
    ):
        return GatewayError(
            reason="quota_exhausted",
            retryable=True,
            provider_status_code=response.status_code,
            provider_message_safe=provider_message,
            retry_after_seconds=parse_retry_after(response.headers.get("Retry-After")),
            request_id=request_id,
            provider_called=True,
        )
    return build_gateway_error_from_response(
        response=response,
        request_id=request_id,
        supports_content_filter=True,
    )
