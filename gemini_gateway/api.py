from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from math import ceil
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from core.string_parsing import parse_optional_string as _optional_string
from gemini_gateway.contracts import (
    GatewayChatRequest,
    GatewayEmbeddingRequest,
    GatewayErrorReason,
    GatewayRouteRequest,
    GatewayTTSRequest,
)
from gemini_gateway.errors import GatewayError, public_message_for_reason, public_provider_reason
from gemini_gateway.service import create_default_service

_LOGGER = logging.getLogger(__name__)


ReadinessCheck = Callable[[], Awaitable[dict[str, Any]] | dict[str, Any]]


def create_app(
    *,
    auth_token: str,
    completion_service: Any | None = None,
    readiness_check: ReadinessCheck | None = None,
    service_name: str = "gemini-gateway",
    environment: str = "development",
) -> FastAPI:
    app = FastAPI(title="Gemini Gateway", docs_url=None, redoc_url=None)
    event_environment = environment or "development"
    event_service_name = service_name or "gemini-gateway"
    service = completion_service or create_default_service(
        service_name=event_service_name,
        environment=event_environment,
    )

    @app.get("/health/live")
    async def health_live() -> JSONResponse:
        return JSONResponse(status_code=200, content={"status": "ok"})

    @app.get("/health")
    async def health() -> JSONResponse:
        try:
            readiness = await _resolve_readiness(readiness_check)
        except Exception as exc:
            _log_gateway_api_error(
                event="gemini_gateway_health_error",
                service_name=event_service_name,
                environment=event_environment,
                request_id=None,
                error=exc,
                endpoint="health",
                reason="readiness_check_failed",
                response_reason="health_unready",
                retryable=True,
                failed_stage="readiness_check",
            )
            return JSONResponse(status_code=503, content={"status": "unready", "checks": {}})

        if not readiness.get("ok", False):
            return JSONResponse(
                status_code=503,
                content={"status": "unready", "checks": readiness.get("checks", {})},
            )
        return JSONResponse(status_code=200, content={"status": "ok", "checks": readiness.get("checks", {})})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
        if not _is_authorized(authorization=authorization, auth_token=auth_token):
            return _error_response(
                request_id=None,
                status_code=401,
                reason="unauthorized",
                retryable=False,
            )

        body = await _safe_body(request)
        request_id = _request_id_from_body(body)

        try:
            gateway_request = _parse_gateway_request(body)
        except ValidationError:
            return _error_response(
                request_id=request_id,
                status_code=400,
                reason="bad_request",
                retryable=False,
            )

        try:
            response = await service.complete(gateway_request)
        except GatewayError as error:
            return _error_response(
                request_id=error.request_id or gateway_request.request_id,
                status_code=getattr(error, "status_code", 503),
                reason=error.reason,
                retryable=error.retryable,
                public_message=error.public_message,
                provider_reason=public_provider_reason(error.provider_message_safe),
                provider_status_code=error.provider_status_code,
                retry_after_seconds=_gateway_error_retry_after_seconds(error),
                route_context=_gateway_error_route_context(error),
            )
        except Exception as exc:
            _log_gateway_api_error(
                event="gemini_gateway_api_error",
                service_name=event_service_name,
                environment=event_environment,
                request_id=gateway_request.request_id,
                error=exc,
                endpoint="chat_completions",
                gateway_request=gateway_request,
                reason="unhandled_exception",
                response_reason="request_failed",
                retryable=True,
                failed_stage="chat_completion_handler",
            )
            return _error_response(
                request_id=gateway_request.request_id,
                status_code=500,
                reason="request_failed",
                retryable=True,
            )

        return JSONResponse(status_code=200, content=_dump(response))

    @app.post("/v1/embeddings")
    async def embeddings(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
        if not _is_authorized(authorization=authorization, auth_token=auth_token):
            return _error_response(
                request_id=None,
                status_code=401,
                reason="unauthorized",
                retryable=False,
            )

        body = await _safe_body(request)
        request_id = _request_id_from_body(body)

        try:
            gateway_request = _parse_embedding_request(body)
        except ValidationError:
            return _error_response(
                request_id=request_id,
                status_code=400,
                reason="bad_request",
                retryable=False,
            )

        try:
            response = await service.embed(gateway_request)
        except GatewayError as error:
            return _error_response(
                request_id=error.request_id or gateway_request.request_id,
                status_code=getattr(error, "status_code", 503),
                reason=error.reason,
                retryable=error.retryable,
                public_message=error.public_message,
                provider_reason=public_provider_reason(error.provider_message_safe),
                provider_status_code=error.provider_status_code,
                retry_after_seconds=_gateway_error_retry_after_seconds(error),
                route_context=_gateway_error_route_context(error),
            )
        except Exception as exc:
            _log_gateway_api_error(
                event="gemini_gateway_embeddings_api_error",
                service_name=event_service_name,
                environment=event_environment,
                request_id=gateway_request.request_id,
                error=exc,
                endpoint="embeddings",
                gateway_request=gateway_request,
                reason="unhandled_exception",
                response_reason="request_failed",
                retryable=True,
                failed_stage="embeddings_handler",
            )
            return _error_response(
                request_id=gateway_request.request_id,
                status_code=500,
                reason="request_failed",
                retryable=True,
            )

        return JSONResponse(status_code=200, content=_dump(response))

    @app.post("/v1/audio/speech")
    async def audio_speech(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
        if not _is_authorized(authorization=authorization, auth_token=auth_token):
            return _error_response(
                request_id=None,
                status_code=401,
                reason="unauthorized",
                retryable=False,
            )

        body = await _safe_body(request)
        request_id = _request_id_from_body(body)

        try:
            gateway_request = _parse_tts_request(body)
        except ValidationError:
            return _error_response(
                request_id=request_id,
                status_code=400,
                reason="bad_request",
                retryable=False,
            )

        try:
            response = await service.synthesize_speech(gateway_request)
        except GatewayError as error:
            return _error_response(
                request_id=error.request_id or gateway_request.request_id,
                status_code=getattr(error, "status_code", 503),
                reason=error.reason,
                retryable=error.retryable,
                public_message=error.public_message,
                provider_reason=public_provider_reason(error.provider_message_safe),
                provider_status_code=error.provider_status_code,
                retry_after_seconds=_gateway_error_retry_after_seconds(error),
                route_context=_gateway_error_route_context(error),
            )
        except Exception as exc:
            _log_gateway_api_error(
                event="gemini_gateway_tts_api_error",
                service_name=event_service_name,
                environment=event_environment,
                request_id=gateway_request.request_id,
                error=exc,
                endpoint="audio_speech",
                gateway_request=gateway_request,
                reason="unhandled_exception",
                response_reason="request_failed",
                retryable=True,
                failed_stage="tts_handler",
            )
            return _error_response(
                request_id=gateway_request.request_id,
                status_code=500,
                reason="request_failed",
                retryable=True,
            )

        return JSONResponse(status_code=200, content=_dump(response))

    return app


def _log_gateway_api_error(
    *,
    event: str,
    service_name: str,
    environment: str,
    request_id: str | None,
    error: Exception,
    endpoint: str,
    reason: str,
    response_reason: str,
    retryable: bool,
    failed_stage: str,
    gateway_request: GatewayRouteRequest | None = None,
) -> None:
    _LOGGER.error(
        event,
        extra={
            "event": event,
            "service": service_name,
            "environment": environment,
            "status": "error",
            "reason": reason,
            "response_reason": response_reason,
            "retryable": retryable,
            "error_message": public_message_for_reason(response_reason),
            "failed_stage": failed_stage,
            "request_id": request_id,
            "endpoint": endpoint,
            "source_service": gateway_request.source_service if gateway_request else None,
            "model": gateway_request.model if gateway_request else None,
            "chat_id": gateway_request.chat_id if gateway_request else None,
            "telegram_message_id": gateway_request.telegram_message_id if gateway_request else None,
            "error_type": type(error).__name__,
        },
    )


async def _resolve_readiness(readiness_check: ReadinessCheck | None) -> dict[str, Any]:
    if readiness_check is None:
        return {"ok": True, "checks": {}}
    result = readiness_check()
    if hasattr(result, "__await__"):
        result = await result
    return result if isinstance(result, dict) else {"ok": False, "checks": {}}


def _is_authorized(*, authorization: str | None, auth_token: str) -> bool:
    return bool(auth_token) and authorization == f"Bearer {auth_token}"


async def _safe_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def _request_id_from_body(body: dict[str, Any]) -> str | None:
    request_id = body.get("request_id")
    return str(request_id) if request_id is not None else None


def _parse_gateway_request(body: dict[str, Any]) -> GatewayChatRequest:
    if hasattr(GatewayChatRequest, "model_validate"):
        return GatewayChatRequest.model_validate(body)
    return GatewayChatRequest(**body)


def _parse_tts_request(body: dict[str, Any]) -> GatewayTTSRequest:
    if hasattr(GatewayTTSRequest, "model_validate"):
        return GatewayTTSRequest.model_validate(body)
    return GatewayTTSRequest(**body)


def _parse_embedding_request(body: dict[str, Any]) -> GatewayEmbeddingRequest:
    if hasattr(GatewayEmbeddingRequest, "model_validate"):
        return GatewayEmbeddingRequest.model_validate(body)
    return GatewayEmbeddingRequest(**body)


def _error_response(
    *,
    request_id: str | None,
    status_code: int,
    reason: str,
    retryable: bool,
    public_message: str | None = None,
    provider_reason: str | None = None,
    provider_status_code: int | None = None,
    retry_after_seconds: int | None = None,
    route_context: dict[str, str | None] | None = None,
) -> JSONResponse:
    content = {
        "request_id": request_id,
        "error": public_message or public_message_for_reason(reason),
        "reason": reason,
        "retryable": retryable,
    }
    if provider_reason:
        content["provider_reason"] = provider_reason
    if provider_status_code is not None:
        content["provider_status_code"] = provider_status_code
    if retry_after_seconds is not None and retry_after_seconds > 0:
        content["retry_after_seconds"] = retry_after_seconds
    if route_context:
        content.update({key: value for key, value in route_context.items() if value is not None})
    return JSONResponse(
        status_code=status_code,
        content=content,
    )


def _gateway_error_route_context(error: GatewayError) -> dict[str, str | None]:
    return {
        "route_label": _optional_string(getattr(error, "route_label", None)),
        "project_label": _optional_string(getattr(error, "project_label", None)),
        "key_label": _optional_string(getattr(error, "key_label", None)),
        "proxy_label": _optional_string(getattr(error, "proxy_label", None)),
        "transport_mode": _optional_string(getattr(error, "transport_mode", None)),
    }


def _gateway_error_retry_after_seconds(error: GatewayError) -> int | None:
    retry_after_seconds = getattr(error, "retry_after_seconds", None)
    if isinstance(retry_after_seconds, int) and retry_after_seconds > 0:
        return retry_after_seconds

    sleep_until = getattr(error, "sleep_until", None)
    if not isinstance(sleep_until, datetime):
        return None
    if sleep_until.tzinfo is None:
        sleep_until = sleep_until.replace(tzinfo=UTC)
    seconds = ceil((sleep_until - datetime.now(tz=UTC)).total_seconds())
    return max(1, seconds) if seconds > 0 else None


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return value
    return {key: item for key, item in vars(value).items() if item is not None}
