from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from time import perf_counter
from typing import Any, TypeVar

from core.number_parsing import parse_optional_int as _safe_int
from core.wide_events import build_wide_event
from gemini_gateway.contracts import (
    GatewayChatRequest,
    GatewayChatResponse,
    GatewayEmbeddingRequest,
    GatewayEmbeddingResponse,
    GatewayRouteRequest,
    GatewayRouteMetadata,
    GatewayTTSRequest,
    GatewayTTSResponse,
    GatewayProviderResponse,
    RouteLease,
)
from gemini_gateway.embedding_client import GeminiEmbeddingClient
from gemini_gateway.errors import GatewayError, public_provider_reason
from gemini_gateway.gemini_client import GeminiOpenAIClient
from gemini_gateway.repository import (
    ATTEMPTED_ROUTES_EXHAUSTED_ERROR_CODE,
    InMemoryRouteRepository,
)
from gemini_gateway.provider_observability import provider_timing_columns
from gemini_gateway.tts_client import GeminiTTSClient

_LOGGER = logging.getLogger(__name__)
ResponseT = TypeVar("ResponseT", bound=GatewayProviderResponse)
_OPENROUTER_EMBEDDING_FALLBACK_MODEL = "google/gemini-embedding-2"
_OPENROUTER_EMBEDDING_FALLBACK_REASONS = frozenset(
    {
        "no_route",
        "cooldown_active",
        "quota_exhausted",
        "rate_limited",
        "proxy_failed",
        "network_timeout",
        "provider_unavailable",
        "request_failed",
    }
)
_OPENROUTER_EMBEDDING_FALLBACK_STAGE = "openrouter_embedding_fallback"
_OPENROUTER_FALLBACK_PROVIDER = "openrouter"
_DEFAULT_MAX_ROUTE_ATTEMPTS = 5


class CompletionService:
    """Оркестрирует lease маршрута, Gemini вызов и учет результата."""

    def __init__(
        self,
        *,
        repository: Any,
        gemini_client: Any,
        tts_client: Any | None = None,
        embedding_client: Any | None = None,
        openrouter_embedding_client: Any | None = None,
        openrouter_api_key: str | None = None,
        openrouter_embeddings_fallback_enabled: bool = False,
        openrouter_embeddings_fallback_model: str = _OPENROUTER_EMBEDDING_FALLBACK_MODEL,
        max_route_attempts: int = _DEFAULT_MAX_ROUTE_ATTEMPTS,
        service_name: str = "gemini-gateway",
        environment: str = "development",
    ) -> None:
        self._repository = repository
        self._gemini_client = gemini_client
        self._tts_client = tts_client
        self._embedding_client = embedding_client
        self._openrouter_embedding_client = openrouter_embedding_client
        self._openrouter_api_key = _normalize_optional_secret(openrouter_api_key)
        self._openrouter_embeddings_fallback_enabled = openrouter_embeddings_fallback_enabled
        self._openrouter_embeddings_fallback_model = openrouter_embeddings_fallback_model
        self._max_route_attempts = max(1, int(max_route_attempts))
        self._service_name = service_name
        self._environment = environment

    async def complete(self, request: GatewayChatRequest | dict[str, Any]) -> GatewayChatResponse:
        gateway_request = _ensure_request(request)
        return await self._execute_with_route_retries(
            request=gateway_request,
            provider_call=lambda attempt_request, lease: self._gemini_client.complete(
                request=attempt_request,
                api_key=lease.api_key,
                proxy_url=lease.proxy_url,
            ),
        )

    async def synthesize_speech(self, request: GatewayTTSRequest | dict[str, Any]) -> GatewayTTSResponse:
        gateway_request = _ensure_tts_request(request)
        if self._tts_client is None:
            raise GatewayError(
                reason="provider_unavailable",
                retryable=True,
                request_id=gateway_request.request_id,
            )
        return await self._execute_with_route_retries(
            request=gateway_request,
            provider_call=lambda attempt_request, lease: self._tts_client.synthesize(
                request=attempt_request,
                api_key=lease.api_key,
                proxy_url=lease.proxy_url,
            ),
        )

    async def embed(self, request: GatewayEmbeddingRequest | dict[str, Any]) -> GatewayEmbeddingResponse:
        gateway_request = _ensure_embedding_request(request)
        if self._embedding_client is None:
            raise GatewayError(
                reason="provider_unavailable",
                retryable=True,
                request_id=gateway_request.request_id,
            )
        try:
            return await self._execute_with_route_retries(
                request=gateway_request,
                provider_call=lambda attempt_request, lease: self._embed_with_gemini_route(attempt_request, lease),
            )
        except GatewayError as error:
            if not self._should_use_openrouter_embedding_fallback(request=gateway_request, error=error):
                raise
            return await self._execute_openrouter_embedding_fallback(gateway_request)

    async def _embed_with_gemini_route(
        self,
        request: GatewayEmbeddingRequest,
        lease: RouteLease,
    ) -> GatewayEmbeddingResponse:
        if self._embedding_client is None:
            raise GatewayError(
                reason="provider_unavailable",
                retryable=True,
                request_id=request.request_id,
            )
        return await self._embedding_client.embed(
            request=request,
            api_key=lease.api_key,
            proxy_url=lease.proxy_url,
        )

    def _should_use_openrouter_embedding_fallback(
        self,
        *,
        request: GatewayEmbeddingRequest,
        error: GatewayError,
    ) -> bool:
        return (
            self._openrouter_embeddings_fallback_enabled
            and self._openrouter_embedding_client is not None
            and self._openrouter_api_key is not None
            and request.model == self._openrouter_embeddings_fallback_model
            and error.retryable
            and error.reason in _OPENROUTER_EMBEDDING_FALLBACK_REASONS
        )

    async def _execute_openrouter_embedding_fallback(
        self,
        request: GatewayEmbeddingRequest,
    ) -> GatewayEmbeddingResponse:
        started_at = perf_counter()
        client = self._openrouter_embedding_client
        api_key = self._openrouter_api_key
        if client is None or api_key is None:
            raise GatewayError(
                reason="provider_unavailable",
                retryable=True,
                request_id=request.request_id,
            )

        try:
            response = await client.embed(request=request, api_key=api_key)
            response = _attach_direct_route_metadata(response)
            latency_ms = _elapsed_ms(started_at)
            self._log_openrouter_embedding_fallback_success(request, response, latency_ms)
            return response
        except GatewayError as error:
            latency_ms = _elapsed_ms(started_at)
            _set_request_id(error, request.request_id)
            _attach_direct_error_route_metadata(error)
            self._log_openrouter_embedding_fallback_failure(request, error, latency_ms)
            raise
        except Exception as exc:
            latency_ms = _elapsed_ms(started_at)
            error = GatewayError(
                reason="request_failed",
                retryable=True,
                provider_message_safe=str(exc),
                request_id=request.request_id,
                status_code=500,
            )
            _attach_direct_error_route_metadata(error)
            self._log_openrouter_embedding_fallback_failure(request, error, latency_ms)
            raise error from exc

    async def _execute_with_route_retries(
        self,
        *,
        request: GatewayRouteRequest,
        provider_call: Callable[[GatewayRouteRequest, RouteLease], Awaitable[ResponseT]],
    ) -> ResponseT:
        last_error: GatewayError | None = None

        for attempt in range(self._max_route_attempts):
            attempt_request = _request_with_retry_count(request, attempt)
            try:
                return await self._execute_with_route(
                    request=attempt_request,
                    provider_call=lambda lease, attempt_request=attempt_request: provider_call(attempt_request, lease),
                )
            except GatewayError as error:
                last_error = error
                if not _should_retry_with_next_route(error):
                    raise
                if attempt + 1 >= self._max_route_attempts:
                    raise

        if last_error is not None:
            raise last_error
        raise GatewayError(reason="request_failed", retryable=True, request_id=request.request_id)

    async def _execute_with_route(
        self,
        *,
        request: GatewayRouteRequest,
        provider_call: Callable[[RouteLease], Awaitable[ResponseT]],
    ) -> ResponseT:
        started_at = perf_counter()
        lease: RouteLease | None = None
        provider_called = False

        try:
            lease = await self._repository.acquire_route(request)
            response = await provider_call(lease)
            provider_called = True
            response = _attach_route_metadata(response, lease)
            latency_ms = _elapsed_ms(started_at)
            await _maybe_await(self._repository.record_success(lease, response, latency_ms))
            self._log_success(request, lease, response, latency_ms)
            return response
        except GatewayError as error:
            latency_ms = _elapsed_ms(started_at)
            _set_request_id(error, request.request_id)
            _attach_error_route_metadata(error, lease)
            provider_called = provider_called or bool(getattr(error, "provider_called", False))
            await _maybe_await(self._repository.record_failure(lease, error, latency_ms, provider_called))
            self._log_failure(request, lease, error, latency_ms, provider_called=provider_called)
            raise
        except Exception as exc:
            latency_ms = _elapsed_ms(started_at)
            error = GatewayError(
                reason="request_failed",
                retryable=True,
                provider_message_safe=str(exc),
                request_id=request.request_id,
                status_code=500,
            )
            _attach_error_route_metadata(error, lease)
            await _maybe_await(self._repository.record_failure(lease, error, latency_ms, provider_called))
            self._log_failure(request, lease, error, latency_ms, provider_called=provider_called)
            raise error from exc

    async def health_check(self, *, require_routes: bool = False) -> dict[str, Any]:
        health_check = getattr(self._repository, "health_check", None)
        if health_check is None:
            return {"ok": True, "checks": {"database": True, "schema": True, "routes": not require_routes}}
        return await _maybe_await(health_check(require_routes=require_routes))

    def _log_success(
        self,
        request: GatewayRouteRequest,
        lease: RouteLease,
        response: GatewayProviderResponse,
        latency_ms: int,
    ) -> None:
        event = self._base_event(request=request, lease=lease, latency_ms=latency_ms, status="success")
        event.update(_success_log_fields(response))
        _LOGGER.info("gemini_gateway_request", extra=event)

    def _log_failure(
        self,
        request: GatewayRouteRequest,
        lease: RouteLease | None,
        error: GatewayError,
        latency_ms: int,
        provider_called: bool,
    ) -> None:
        event = self._base_event(request=request, lease=lease, latency_ms=latency_ms, status="error")
        event.update(_failure_log_fields(error, _gateway_failure_stage(lease=lease, provider_called=provider_called)))
        _LOGGER.warning("gemini_gateway_request", extra=event)

    def _log_openrouter_embedding_fallback_success(
        self,
        request: GatewayEmbeddingRequest,
        response: GatewayEmbeddingResponse,
        latency_ms: int,
    ) -> None:
        event = self._base_event(
            request=request,
            lease=None,
            latency_ms=latency_ms,
            status="success",
            route_metadata=_openrouter_fallback_route_metadata(),
        )
        event.update(_success_log_fields(response))
        event["fallback_provider"] = _OPENROUTER_FALLBACK_PROVIDER
        _LOGGER.info("gemini_gateway_request", extra=event)

    def _log_openrouter_embedding_fallback_failure(
        self,
        request: GatewayEmbeddingRequest,
        error: GatewayError,
        latency_ms: int,
    ) -> None:
        event = self._base_event(
            request=request,
            lease=None,
            latency_ms=latency_ms,
            status="error",
            route_metadata=_openrouter_fallback_route_metadata(),
        )
        event.update(_failure_log_fields(error, _OPENROUTER_EMBEDDING_FALLBACK_STAGE))
        event["fallback_provider"] = _OPENROUTER_FALLBACK_PROVIDER
        _LOGGER.warning("gemini_gateway_request", extra=event)

    def _base_event(
        self,
        *,
        request: GatewayRouteRequest,
        lease: RouteLease | None,
        latency_ms: int,
        status: str,
        route_metadata: Any | None = None,
    ) -> dict[str, Any]:
        route = (
            _route_metadata_to_dict(route_metadata)
            if route_metadata is not None
            else _route_metadata_from_lease(lease)
        )
        return build_wide_event(
            event="gemini_gateway_request",
            service=self._service_name,
            environment=self._environment,
            request_id=request.request_id,
            source_service=request.source_service,
            chat_id=request.chat_id,
            telegram_message_id=request.telegram_message_id,
            model=request.model,
            route_label=route.get("route_label"),
            project_label=route.get("project_label"),
            key_label=route.get("key_label"),
            proxy_label=route.get("proxy_label"),
            transport_mode=route.get("transport_mode"),
            status=status,
            duration_ms=latency_ms,
            retry_count=getattr(request, "retry_count", 0),
        )


def create_default_service(
    *,
    service_name: str = "gemini-gateway",
    environment: str = "development",
) -> CompletionService:
    return CompletionService(
        repository=InMemoryRouteRepository([]),
        gemini_client=GeminiOpenAIClient(),
        tts_client=GeminiTTSClient(),
        embedding_client=GeminiEmbeddingClient(),
        service_name=service_name,
        environment=environment,
    )


def _ensure_request(request: GatewayChatRequest | dict[str, Any]) -> GatewayChatRequest:
    if isinstance(request, GatewayChatRequest):
        return request
    if hasattr(GatewayChatRequest, "model_validate"):
        return GatewayChatRequest.model_validate(request)
    return GatewayChatRequest(**request)


def _ensure_tts_request(request: GatewayTTSRequest | dict[str, Any]) -> GatewayTTSRequest:
    if isinstance(request, GatewayTTSRequest):
        return request
    if hasattr(GatewayTTSRequest, "model_validate"):
        return GatewayTTSRequest.model_validate(request)
    return GatewayTTSRequest(**request)


def _ensure_embedding_request(request: GatewayEmbeddingRequest | dict[str, Any]) -> GatewayEmbeddingRequest:
    if isinstance(request, GatewayEmbeddingRequest):
        return request
    if hasattr(GatewayEmbeddingRequest, "model_validate"):
        return GatewayEmbeddingRequest.model_validate(request)
    return GatewayEmbeddingRequest(**request)


def _request_with_retry_count(request: GatewayRouteRequest, retry_count: int) -> GatewayRouteRequest:
    if getattr(request, "retry_count", 0) == retry_count:
        return request
    if hasattr(request, "model_copy"):
        return request.model_copy(update={"retry_count": retry_count})
    return request.copy(update={"retry_count": retry_count})


def _should_retry_with_next_route(error: GatewayError) -> bool:
    if not error.retryable:
        return False
    if error.error_code == ATTEMPTED_ROUTES_EXHAUSTED_ERROR_CODE:
        return False
    return _gateway_error_has_route_metadata(error)


def _attach_route_metadata(response: ResponseT, lease: RouteLease) -> ResponseT:
    return _attach_response_route_metadata(response, _route_metadata_from_lease(lease))


def _attach_direct_route_metadata(response: ResponseT) -> ResponseT:
    return _attach_response_route_metadata(response, _openrouter_fallback_route_metadata())


def _attach_response_route_metadata(response: ResponseT, route_metadata: Any) -> ResponseT:
    route = _route_metadata_to_dict(route_metadata)
    updates = {
        "route": route_metadata,
        "route_label": route["route_label"],
        "project_label": route["project_label"],
        "key_label": route["key_label"],
        "proxy_label": route["proxy_label"],
        "transport_mode": route["transport_mode"],
    }
    if hasattr(response, "model_copy"):
        return response.model_copy(update=updates)
    for key, value in updates.items():
        setattr(response, key, value)
    return response


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))


def _set_request_id(error: GatewayError, request_id: str) -> None:
    if getattr(error, "request_id", None) is None:
        error.request_id = request_id


def _attach_error_route_metadata(error: GatewayError, lease: RouteLease | None) -> None:
    if lease is None:
        return
    _attach_error_metadata(error, _route_metadata_from_lease(lease), overwrite=False)


def _attach_direct_error_route_metadata(error: GatewayError) -> None:
    _attach_error_metadata(error, _openrouter_fallback_route_metadata(), overwrite=True)


def _attach_error_metadata(error: GatewayError, route_metadata: Any, *, overwrite: bool) -> None:
    for field_name, value in _route_metadata_to_dict(route_metadata).items():
        if field_name == "route":
            continue
        if overwrite:
            setattr(error, field_name, value)
            continue
        if getattr(error, field_name, None) is None:
            setattr(error, field_name, value)


def _route_metadata_from_lease(lease: RouteLease | None) -> dict[str, Any]:
    if lease is None:
        return {}
    return {
        "project_label": lease.project_label,
        "route_label": lease.route_label,
        "key_label": lease.key_label,
        "proxy_label": lease.proxy_label,
        "transport_mode": lease.transport_mode,
    }


def _openrouter_fallback_route_metadata() -> GatewayRouteMetadata:
    return GatewayRouteMetadata(
        project_label="openrouter-fallback",
        route_label="openrouter-embedding-fallback",
        key_label="openrouter-api-key",
        proxy_label=None,
        transport_mode="direct",
    )


def _route_metadata_to_dict(route_metadata: Any) -> dict[str, Any]:
    if isinstance(route_metadata, GatewayRouteMetadata):
        return route_metadata.model_dump(mode="python")
    if isinstance(route_metadata, dict):
        return dict(route_metadata)
    return {
        "project_label": route_metadata.project_label,
        "route_label": route_metadata.route_label,
        "key_label": route_metadata.key_label,
        "proxy_label": route_metadata.proxy_label,
        "transport_mode": route_metadata.transport_mode,
    }


def _gateway_error_has_route_metadata(error: GatewayError) -> bool:
    return any(
        getattr(error, field_name, None) is not None
        for field_name in ("route_label", "project_label", "key_label", "transport_mode")
    )


def _normalize_optional_secret(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _success_log_fields(response: GatewayProviderResponse) -> dict[str, Any]:
    usage = response.usage or {}
    fields = {
        "prompt_tokens": _safe_int(usage.get("prompt_tokens")),
        "completion_tokens": _safe_int(usage.get("completion_tokens")),
        "total_tokens": _safe_int(usage.get("total_tokens")),
        "generation_id": response.generation_id,
        "finish_reason": response.finish_reason,
        "error_type": None,
        "error_message": None,
        "error_code": None,
        "retryable": None,
        "cooldown_scope": None,
        "cooldown_level": None,
        "sleep_until": None,
        "quota_scope": None,
        "quota_reset_at": None,
        "eligible_routes_count": None,
        "exhausted_routes_count": None,
        "disabled_routes_count": None,
    }
    fields.update(provider_timing_columns(getattr(response, "provider_timing", None)))
    return fields


def _failure_log_fields(error: GatewayError, failed_stage: str) -> dict[str, Any]:
    fields = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "finish_reason": None,
        "reason": error.reason,
        "failed_stage": failed_stage,
        "error_type": error.reason,
        "error_code": getattr(error, "error_code", None),
        "error_message": error.public_message,
        "provider_reason": public_provider_reason(error.provider_message_safe),
        "provider_status_code": error.provider_status_code,
        "retryable": error.retryable,
        "retry_after_seconds": error.retry_after_seconds,
        "cooldown_scope": getattr(error, "cooldown_scope", None),
        "cooldown_level": getattr(error, "cooldown_level", None),
        "sleep_until": _serialize_datetime(getattr(error, "sleep_until", None)),
        "quota_scope": getattr(error, "quota_scope", None),
        "quota_reset_at": _serialize_datetime(getattr(error, "quota_reset_at", None)),
        "eligible_routes_count": getattr(error, "eligible_routes_count", None),
        "exhausted_routes_count": getattr(error, "exhausted_routes_count", None),
        "disabled_routes_count": getattr(error, "disabled_routes_count", None),
    }
    fields.update(provider_timing_columns(getattr(error, "provider_timing", None)))
    return fields


def _serialize_datetime(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _gateway_failure_stage(*, lease: RouteLease | None, provider_called: bool) -> str:
    if lease is None:
        return "route_acquisition"
    if provider_called:
        return "provider_call"
    return "route_transport"
