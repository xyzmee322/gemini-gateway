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
    GatewayTTSRequest,
    GatewayTTSResponse,
    GatewayProviderResponse,
    RouteLease,
)
from gemini_gateway.embedding_client import GeminiEmbeddingClient
from gemini_gateway.errors import GatewayError, public_provider_reason
from gemini_gateway.gemini_client import GeminiOpenAIClient
from gemini_gateway.repository import InMemoryRouteRepository
from gemini_gateway.tts_client import GeminiTTSClient

_LOGGER = logging.getLogger(__name__)
ResponseT = TypeVar("ResponseT", bound=GatewayProviderResponse)


class CompletionService:
    """Оркестрирует lease маршрута, Gemini вызов и учет результата."""

    def __init__(
        self,
        *,
        repository: Any,
        gemini_client: Any,
        tts_client: Any | None = None,
        embedding_client: Any | None = None,
        service_name: str = "gemini-gateway",
        environment: str = "development",
    ) -> None:
        self._repository = repository
        self._gemini_client = gemini_client
        self._tts_client = tts_client
        self._embedding_client = embedding_client
        self._service_name = service_name
        self._environment = environment

    async def complete(self, request: GatewayChatRequest | dict[str, Any]) -> GatewayChatResponse:
        gateway_request = _ensure_request(request)
        return await self._execute_with_route(
            request=gateway_request,
            provider_call=lambda lease: self._gemini_client.complete(
                request=gateway_request,
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
        return await self._execute_with_route(
            request=gateway_request,
            provider_call=lambda lease: self._tts_client.synthesize(
                request=gateway_request,
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
        return await self._execute_with_route(
            request=gateway_request,
            provider_call=lambda lease: self._embedding_client.embed(
                request=gateway_request,
                api_key=lease.api_key,
                proxy_url=lease.proxy_url,
            ),
        )

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
        usage = response.usage or {}
        event.update(
            {
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
        )
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
        event.update(
            {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "finish_reason": None,
                "reason": error.reason,
                "failed_stage": _gateway_failure_stage(lease=lease, provider_called=provider_called),
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
        )
        _LOGGER.warning("gemini_gateway_request", extra=event)

    def _base_event(
        self,
        *,
        request: GatewayRouteRequest,
        lease: RouteLease | None,
        latency_ms: int,
        status: str,
    ) -> dict[str, Any]:
        return build_wide_event(
            event="gemini_gateway_request",
            service=self._service_name,
            environment=self._environment,
            request_id=request.request_id,
            source_service=request.source_service,
            chat_id=request.chat_id,
            telegram_message_id=request.telegram_message_id,
            model=request.model,
            route_label=lease.route_label if lease else None,
            project_label=lease.project_label if lease else None,
            key_label=lease.key_label if lease else None,
            proxy_label=lease.proxy_label if lease else None,
            transport_mode=lease.transport_mode if lease else None,
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


def _attach_route_metadata(response: ResponseT, lease: RouteLease) -> ResponseT:
    updates = {
        "route": {
            "project_label": lease.project_label,
            "route_label": lease.route_label,
            "key_label": lease.key_label,
            "proxy_label": lease.proxy_label,
            "transport_mode": lease.transport_mode,
        },
        "route_label": lease.route_label,
        "project_label": lease.project_label,
        "key_label": lease.key_label,
        "proxy_label": lease.proxy_label,
        "transport_mode": lease.transport_mode,
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
    route_fields = {
        "route_label": lease.route_label,
        "project_label": lease.project_label,
        "key_label": lease.key_label,
        "proxy_label": lease.proxy_label,
        "transport_mode": lease.transport_mode,
    }
    for field_name, value in route_fields.items():
        if getattr(error, field_name, None) is None:
            setattr(error, field_name, value)


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
