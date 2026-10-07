from __future__ import annotations

import logging
from typing import Any

from fastapi.testclient import TestClient

from gemini_gateway.api import create_app
from gemini_gateway.contracts import GatewayChatResponse, GatewayEmbeddingResponse, GatewayTTSResponse
from gemini_gateway.errors import GatewayError
from gemini_gateway.monitoring import MonitorWindow


class _SuccessfulService:
    async def complete(self, request: Any) -> GatewayChatResponse:
        return GatewayChatResponse(
            request_id=request.request_id,
            generation_id="gen-1",
            model=request.model,
            choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            raw_response={"id": "gen-1"},
            provider_specific_fields={"diagnostic": "raw provider trace must not leave gateway"},
            provider_timing={
                "operation_type": "chat",
                "payload_kind": "text",
                "request_bytes": 128,
                "response_bytes": 64,
                "headers": {"authorization": "Bearer leaked"},
                "response_body": "raw provider body",
            },
        )

    async def embed(self, request: Any) -> GatewayEmbeddingResponse:
        return GatewayEmbeddingResponse(
            request_id=request.request_id,
            generation_id="embed-1",
            model=request.model,
            embedding=[0.1] * request.dimensions,
            dimensions=request.dimensions,
            usage={"prompt_tokens": 1, "total_tokens": 1},
            provider_timing={
                "operation_type": "embedding",
                "payload_kind": "media",
                "request_bytes": 256,
                "response_bytes": 128,
                "headers": {"authorization": "Bearer leaked"},
                "response_body": "raw embedding body",
            },
        )

    async def synthesize_speech(self, request: Any) -> GatewayTTSResponse:
        return GatewayTTSResponse(
            request_id=request.request_id,
            generation_id="tts-1",
            model=request.model,
            audio_base64="UklGRg==",
            audio_mime_type="audio/wav",
            usage={"prompt_tokens": 3, "total_tokens": 3},
            provider_timing={
                "operation_type": "tts",
                "payload_kind": "tts",
                "request_bytes": 512,
                "response_bytes": 256,
                "headers": {"authorization": "Bearer leaked"},
                "response_body": "raw tts body",
            },
        )


class _FailingService:
    async def complete(self, request: Any) -> GatewayChatResponse:
        raise GatewayError(
            reason="rate_limited",
            retryable=True,
            provider_status_code=429,
            provider_message_safe="HTTP 429 raw secret",
            request_id=request.request_id,
            provider_timing={
                "operation_type": "chat",
                "headers": {"authorization": "Bearer leaked"},
                "response_body": "raw failure body",
            },
        )

    async def synthesize_speech(self, request: Any) -> GatewayTTSResponse:
        raise GatewayError(
            reason="no_route",
            retryable=True,
            provider_message_safe="raw route details",
            request_id=request.request_id,
            provider_timing={
                "operation_type": "tts",
                "headers": {"authorization": "Bearer leaked"},
                "response_body": "raw tts failure body",
            },
        )


class _CircuitOpenService:
    async def complete(self, request: Any) -> GatewayChatResponse:
        raise GatewayError(
            reason="cooldown_active",
            retryable=True,
            retry_after_seconds=23,
            provider_called=False,
            request_id=request.request_id,
        )


class _RoutedFailingService:
    async def complete(self, request: Any) -> GatewayChatResponse:
        error = GatewayError(
            reason="provider_unavailable",
            retryable=True,
            provider_status_code=503,
            request_id=request.request_id,
        )
        error.route_label = "route-error"
        error.project_label = "project-error"
        error.key_label = "key-error"
        error.proxy_label = "proxy-error"
        error.transport_mode = "proxy"
        raise error


class _RetryAfterFailingService:
    async def complete(self, request: Any) -> GatewayChatResponse:
        raise GatewayError(
            reason="no_route",
            retryable=True,
            request_id=request.request_id,
            retry_after_seconds=900,
        )


class _QuotaExhaustedService:
    async def complete(self, request: Any) -> GatewayChatResponse:
        raise GatewayError(
            reason="quota_exhausted",
            retryable=True,
            request_id=request.request_id,
            retry_after_seconds=3600,
            quota_scope="day",
            quota_reset_at="2026-06-09T00:00:00Z",
            eligible_routes_count=6,
            exhausted_routes_count=6,
            disabled_routes_count=4,
        )


class _BrokenService:
    async def complete(self, _: Any) -> GatewayChatResponse:
        raise RuntimeError("raw stack with AIza-secret")


class _BrokenTTSService(_SuccessfulService):
    async def synthesize_speech(self, request: Any) -> GatewayTTSResponse:
        del request
        raise RuntimeError("raw TTS stack with SECRET_TOKEN")


class _ContentFilteredService:
    def __init__(self, provider_message_safe: str = "content_filter: PROHIBITED_CONTENT") -> None:
        self._provider_message_safe = provider_message_safe

    async def complete(self, request: Any) -> GatewayChatResponse:
        raise GatewayError(
            reason="content_filtered",
            retryable=False,
            provider_status_code=400,
            provider_message_safe=self._provider_message_safe,
            request_id=request.request_id,
        )


class _NetworkTimeoutService:
    def __init__(self, provider_message_safe: str = "HTTP 408 raw timeout details") -> None:
        self._provider_message_safe = provider_message_safe

    async def complete(self, request: Any) -> GatewayChatResponse:
        raise GatewayError(
            reason="network_timeout",
            retryable=True,
            provider_status_code=408,
            provider_message_safe=self._provider_message_safe,
            request_id=request.request_id,
        )

    async def synthesize_speech(self, request: Any) -> GatewayTTSResponse:
        raise GatewayError(
            reason="network_timeout",
            retryable=True,
            provider_status_code=408,
            provider_message_safe=self._provider_message_safe,
            request_id=request.request_id,
        )


def test_api_requires_bearer_token_and_returns_safe_auth_error() -> None:
    app = create_app(auth_token="secret-token", completion_service=_SuccessfulService())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "request_id": "req-1",
            "source_service": "test",
            "model": "gemini-3.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 401
    assert response.json() == {
        "request_id": None,
        "error": "Недостаточно прав для выполнения запроса",
        "reason": "unauthorized",
        "retryable": False,
    }


def test_api_rejects_unauthorized_request_before_json_parsing() -> None:
    app = create_app(auth_token="secret-token", completion_service=_SuccessfulService())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        content=b'{"request_id": "req-bad"',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 401
    assert response.json()["request_id"] is None
    assert response.json()["reason"] == "unauthorized"


def test_api_success_preserves_gateway_response_shape() -> None:
    app = create_app(auth_token="secret-token", completion_service=_SuccessfulService())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-1",
            "source_service": "test",
            "model": "gemini-3.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["generation_id"] == "gen-1"
    assert payload["choices"][0]["message"]["content"] == "ok"
    assert payload["usage"]["total_tokens"] == 2
    assert payload["provider_specific_fields"] == {}
    assert "raw provider trace" not in str(payload)
    assert "raw_response" not in payload
    assert "provider_timing" not in payload
    assert "Bearer leaked" not in response.text
    assert "raw provider body" not in response.text


def test_embeddings_api_success_does_not_expose_provider_timing() -> None:
    app = create_app(auth_token="secret-token", completion_service=_SuccessfulService())
    client = TestClient(app)

    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-embedding",
            "source_service": "media_memory",
            "model": "google/gemini-embedding-2",
            "input": [{"type": "text", "text": "hello"}],
            "dimensions": 768,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["generation_id"] == "embed-1"
    assert payload["dimensions"] == 768
    assert "provider_timing" not in payload
    assert "Bearer leaked" not in response.text
    assert "raw embedding body" not in response.text


def test_tts_api_requires_bearer_token() -> None:
    app = create_app(auth_token="secret-token", completion_service=_SuccessfulService())
    client = TestClient(app)

    response = client.post(
        "/v1/audio/speech",
        json={
            "request_id": "req-tts",
            "source_service": "voice_tts",
            "model": "google/gemini-3.1-flash-tts-preview",
            "text": "привет",
        },
    )

    assert response.status_code == 401
    assert response.json()["request_id"] is None
    assert response.json()["reason"] == "unauthorized"


def test_tts_api_success_preserves_audio_response_shape() -> None:
    app = create_app(auth_token="secret-token", completion_service=_SuccessfulService())
    client = TestClient(app)

    response = client.post(
        "/v1/audio/speech",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-tts",
            "source_service": "voice_tts",
            "model": "google/gemini-3.1-flash-tts-preview",
            "text": "привет",
            "voice_name": "Kore",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["generation_id"] == "tts-1"
    assert payload["audio_base64"] == "UklGRg=="
    assert payload["audio_mime_type"] == "audio/wav"
    assert payload["usage"]["total_tokens"] == 3
    assert "provider_timing" not in payload
    assert "Bearer leaked" not in response.text
    assert "raw tts body" not in response.text


def test_tts_api_gateway_error_handler_returns_stable_safe_json() -> None:
    app = create_app(auth_token="secret-token", completion_service=_FailingService())
    client = TestClient(app)

    response = client.post(
        "/v1/audio/speech",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-tts-no-route",
            "source_service": "voice_tts",
            "model": "google/gemini-3.1-flash-tts-preview",
            "text": "привет",
        },
    )

    assert response.status_code == 429
    assert response.json() == {
        "request_id": "req-tts-no-route",
        "error": "Сейчас нет доступного маршрута для Gemini, попробуй позже",
        "reason": "no_route",
        "retryable": True,
        "provider_called": False,
    }
    assert "raw route details" not in response.text
    assert "provider_timing" not in response.json()
    assert "Bearer leaked" not in response.text
    assert "raw tts failure body" not in response.text


def test_api_gateway_error_handler_returns_stable_safe_json() -> None:
    app = create_app(auth_token="secret-token", completion_service=_FailingService())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-2",
            "source_service": "test",
            "model": "gemini-3.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 429
    assert response.json() == {
        "request_id": "req-2",
        "error": "Слишком много запросов, попробуй чуть позже",
        "reason": "rate_limited",
        "retryable": True,
        "provider_status_code": 429,
    }
    assert "HTTP 429" not in response.text
    assert "raw secret" not in response.text
    assert "provider_timing" not in response.json()
    assert "Bearer leaked" not in response.text
    assert "raw failure body" not in response.text


def test_api_gateway_error_reports_when_provider_was_not_called() -> None:
    app = create_app(auth_token="secret-token", completion_service=_CircuitOpenService())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-circuit-open",
            "source_service": "test",
            "model": "gemini-3.8-flash",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 429
    assert response.json()["provider_called"] is False


def test_api_gateway_error_handler_includes_safe_route_context() -> None:
    app = create_app(auth_token="secret-token", completion_service=_RoutedFailingService())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-route-error",
            "source_service": "test",
            "model": "gemini-3.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "request_id": "req-route-error",
        "error": "Сервис временно недоступен",
        "reason": "provider_unavailable",
        "retryable": True,
        "provider_status_code": 503,
        "route_label": "route-error",
        "project_label": "project-error",
        "key_label": "key-error",
        "proxy_label": "proxy-error",
        "transport_mode": "proxy",
    }


def test_api_gateway_error_handler_includes_retry_after_hint() -> None:
    app = create_app(auth_token="secret-token", completion_service=_RetryAfterFailingService())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-retry-after",
            "source_service": "test",
            "model": "gemini-3.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "900"
    assert response.json()["reason"] == "no_route"
    assert response.json()["retry_after_seconds"] == 900


def test_api_gateway_error_handler_returns_quota_exhausted_diagnostics() -> None:
    app = create_app(auth_token="secret-token", completion_service=_QuotaExhaustedService())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-quota",
            "source_service": "test",
            "model": "gemini-3.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "3600"
    assert response.json() == {
        "request_id": "req-quota",
        "error": "Квота AI-провайдера временно исчерпана, попробуй позже",
        "reason": "quota_exhausted",
        "error_code": "quota_exhausted",
        "retryable": True,
        "provider_called": False,
        "retry_after_seconds": 3600,
        "quota_scope": "day",
        "quota_reset_at": "2026-06-09T00:00:00Z",
        "eligible_routes_count": 6,
        "exhausted_routes_count": 6,
        "disabled_routes_count": 4,
    }


def test_api_gateway_error_handler_includes_safe_provider_reason() -> None:
    app = create_app(auth_token="secret-token", completion_service=_ContentFilteredService())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-content-filter",
            "source_service": "test",
            "model": "gemini-3.1-flash-lite",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "request_id": "req-content-filter",
        "error": "Не могу ответить на этот запрос",
        "reason": "content_filtered",
        "retryable": False,
        "provider_reason": "content_filtered",
        "provider_status_code": 400,
    }


def test_api_gateway_error_handler_preserves_network_timeout_status_without_raw_text() -> None:
    app = create_app(auth_token="secret-token", completion_service=_NetworkTimeoutService())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-network-timeout",
            "source_service": "test",
            "model": "gemini-3.1-flash-lite",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 408
    assert response.json() == {
        "request_id": "req-network-timeout",
        "error": "Сервис отвечает слишком долго, попробуй позже",
        "reason": "network_timeout",
        "retryable": True,
        "provider_status_code": 408,
    }
    assert "HTTP 408" not in response.text
    assert "raw timeout details" not in response.text


def test_api_gateway_error_handler_includes_stable_timeout_provider_reason() -> None:
    app = create_app(auth_token="secret-token", completion_service=_NetworkTimeoutService("read_timeout"))
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-network-timeout-kind",
            "source_service": "test",
            "model": "gemini-3.1-flash-lite",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 408
    assert response.json() == {
        "request_id": "req-network-timeout-kind",
        "error": "Сервис отвечает слишком долго, попробуй позже",
        "reason": "network_timeout",
        "retryable": True,
        "provider_reason": "read_timeout",
        "provider_status_code": 408,
    }


def test_api_gateway_error_handler_does_not_echo_provider_reason_text() -> None:
    app = create_app(
        auth_token="secret-token",
        completion_service=_ContentFilteredService(
            provider_message_safe="blocked by SAFETY policy near raw prompt text HTTP 400 SECRET_TOKEN"
        ),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-content-filter-raw",
            "source_service": "test",
            "model": "gemini-3.1-flash-lite",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "request_id": "req-content-filter-raw",
        "error": "Не могу ответить на этот запрос",
        "reason": "content_filtered",
        "retryable": False,
        "provider_reason": "content_filtered",
        "provider_status_code": 400,
    }
    assert "raw prompt text" not in response.text
    assert "HTTP 400" not in response.text
    assert "SECRET_TOKEN" not in response.text


def test_api_unknown_exception_is_sanitized(caplog: Any) -> None:
    app = create_app(auth_token="secret-token", completion_service=_BrokenService(), environment="test")
    client = TestClient(app)

    with caplog.at_level(logging.ERROR, logger="gemini_gateway.api"):
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret-token"},
            json={
                "request_id": "req-3",
                "source_service": "test",
                "model": "gemini-3.5-flash",
                "chat_id": -1001,
                "telegram_message_id": 9042,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "request_id": "req-3",
        "error": "Не удалось обработать запрос",
        "reason": "request_failed",
        "retryable": True,
    }
    assert "AIza-secret" not in response.text
    assert "RuntimeError" not in response.text
    assert "AIza-secret" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    [record] = [record for record in caplog.records if getattr(record, "event", None) == "gemini_gateway_api_error"]
    assert record.service == "gemini-gateway"
    assert record.environment == "test"
    assert record.status == "error"
    assert record.reason == "unhandled_exception"
    assert record.response_reason == "request_failed"
    assert record.retryable is True
    assert record.failed_stage == "chat_completion_handler"
    assert record.endpoint == "chat_completions"
    assert record.request_id == "req-3"
    assert record.source_service == "test"
    assert record.model == "gemini-3.5-flash"
    assert record.chat_id == -1001
    assert record.telegram_message_id == 9042
    assert record.error_type == "RuntimeError"
    assert record.error_message == "Не удалось обработать запрос"


def test_tts_api_unknown_exception_logs_safe_error(caplog: Any) -> None:
    app = create_app(auth_token="secret-token", completion_service=_BrokenTTSService(), environment="test")
    client = TestClient(app)

    with caplog.at_level(logging.ERROR, logger="gemini_gateway.api"):
        response = client.post(
            "/v1/audio/speech",
            headers={"Authorization": "Bearer secret-token"},
            json={
                "request_id": "req-tts-broken",
                "source_service": "voice_tts",
                "model": "google/gemini-3.1-flash-tts-preview",
                "chat_id": -1002,
                "telegram_message_id": 9055,
                "text": "привет",
            },
        )

    assert response.status_code == 500
    assert response.json()["request_id"] == "req-tts-broken"
    assert response.json()["reason"] == "request_failed"
    assert "SECRET_TOKEN" not in response.text
    assert "SECRET_TOKEN" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    [record] = [record for record in caplog.records if getattr(record, "event", None) == "gemini_gateway_tts_api_error"]
    assert record.service == "gemini-gateway"
    assert record.environment == "test"
    assert record.status == "error"
    assert record.reason == "unhandled_exception"
    assert record.response_reason == "request_failed"
    assert record.retryable is True
    assert record.failed_stage == "tts_handler"
    assert record.endpoint == "audio_speech"
    assert record.request_id == "req-tts-broken"
    assert record.source_service == "voice_tts"
    assert record.model == "google/gemini-3.1-flash-tts-preview"
    assert record.chat_id == -1002
    assert record.telegram_message_id == 9055
    assert record.error_type == "RuntimeError"
    assert record.error_message == "Не удалось обработать запрос"


def test_health_returns_unready_when_readiness_check_fails() -> None:
    async def readiness_check() -> dict[str, Any]:
        return {"ok": False, "checks": {"database": True, "schema": False, "routes": False}}

    app = create_app(
        auth_token="secret-token",
        completion_service=_SuccessfulService(),
        readiness_check=readiness_check,
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unready",
        "checks": {"database": True, "schema": False, "routes": False},
    }


def test_health_live_returns_ok_even_when_readiness_fails() -> None:
    async def readiness_check() -> dict[str, Any]:
        return {"ok": False, "checks": {"database": True, "schema": True, "routes": False}}

    app = create_app(
        auth_token="secret-token",
        completion_service=_SuccessfulService(),
        readiness_check=readiness_check,
    )
    client = TestClient(app)

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_readiness_exception_logs_safe_error(caplog: Any) -> None:
    async def readiness_check() -> dict[str, Any]:
        raise RuntimeError("readiness failed SECRET_TOKEN")

    app = create_app(
        auth_token="secret-token",
        completion_service=_SuccessfulService(),
        readiness_check=readiness_check,
        environment="test",
    )
    client = TestClient(app)

    with caplog.at_level(logging.ERROR, logger="gemini_gateway.api"):
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "unready", "checks": {}}
    assert "SECRET_TOKEN" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    [record] = [record for record in caplog.records if getattr(record, "event", None) == "gemini_gateway_health_error"]
    assert record.service == "gemini-gateway"
    assert record.environment == "test"
    assert record.status == "error"
    assert record.reason == "readiness_check_failed"
    assert record.response_reason == "health_unready"
    assert record.retryable is True
    assert record.failed_stage == "readiness_check"
    assert record.error_message == "Не удалось обработать запрос"


def test_monitor_summary_requires_bearer_token() -> None:
    app = create_app(auth_token="secret-token", completion_service=_SuccessfulService())
    client = TestClient(app)

    response = client.get("/admin/monitor/api/summary")

    assert response.status_code == 401
    assert response.json()["error"]


def test_monitor_dashboard_html_is_public_and_contains_required_monitoring_contract() -> None:
    app = create_app(auth_token="secret-token", completion_service=_SuccessfulService())
    client = TestClient(app)

    response = client.get("/admin/monitor")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Gemini Proxy Monitor" in response.text
    assert "/admin/monitor/api/summary" in response.text
    assert "/admin/monitor/api/timeseries" in response.text
    assert "Введите токен мониторинга" in response.text
    assert 'name="token"' in response.text
    assert 'name="minutes"' in response.text
    assert 'name="bucket_seconds"' in response.text
    assert 'name="model"' in response.text
    assert 'name="proxy_label"' in response.text
    assert 'name="refresh_seconds"' in response.text
    assert 'name="auto_refresh"' in response.text
    assert "localStorage" in response.text
    assert "setInterval" in response.text
    assert "class MonitorApiError" in response.text
    assert "error instanceof MonitorApiError" in response.text
    assert "return fallbackMessage" in response.text
    assert "Authorization" in response.text
    assert "Bearer" in response.text


def test_monitor_dashboard_html_does_not_expose_sensitive_runtime_details() -> None:
    app = create_app(auth_token="secret-token", completion_service=_SuccessfulService())
    client = TestClient(app)

    response = client.get("/admin/monitor")
    html = response.text.lower()

    assert response.status_code == 200
    assert "secret-token" not in response.text
    assert "proxy_url" not in html
    assert "api_key" not in html
    assert "raw_provider_payload" not in html
    assert "raw payload" not in html
    assert "base64" not in html
    assert "stack trace" not in html
    assert "traceback" not in html
    assert "cdn" not in html
    assert "http://" not in html
    assert "https://" not in html


def test_monitor_dashboard_json_apis_still_require_bearer_token() -> None:
    app = create_app(auth_token="secret-token", completion_service=_SuccessfulService())
    client = TestClient(app)

    summary_response = client.get("/admin/monitor/api/summary")
    timeseries_response = client.get("/admin/monitor/api/timeseries")

    assert summary_response.status_code == 401
    assert timeseries_response.status_code == 401


def test_monitor_summary_returns_safe_json_without_proxy_url() -> None:
    async def fetch_summary(window: MonitorWindow) -> dict[str, Any]:
        assert window.minutes == 180
        return {
            "total_requests": 1,
            "proxy_count": 1,
            "proxies": [{"proxy_label": "proxy-a", "total_requests": 1}],
        }

    app = create_app(
        auth_token="secret-token",
        completion_service=_SuccessfulService(),
        monitoring_summary_fetcher=fetch_summary,
    )
    client = TestClient(app)

    response = client.get("/admin/monitor/api/summary", headers={"Authorization": "Bearer secret-token"})

    assert response.status_code == 200
    assert "proxies" in response.json()
    assert "proxy_url" not in response.text


def test_monitor_summary_sanitizes_sensitive_key_variants() -> None:
    async def fetch_summary(_: MonitorWindow) -> dict[str, Any]:
        return {
            "total_requests": 1,
            "proxy_count": 1,
            "proxy_url": "http://secret-proxy",
            "proxies": [
                {
                    "proxy_label": "proxy-a",
                    "route_label": "route-a",
                    "request_bytes": 128,
                    "response_bytes": 256,
                    "avg_response_body_ms": 70,
                    "p95_response_body_ms": 120,
                    "response_body_timeout_count": 1,
                    "response_body": "raw provider body SECRET_TOKEN",
                    "timeout_kind": "read_timeout",
                    "key_fingerprint": "fingerprint-secret",
                    "proxy_host": "10.0.0.1",
                    "proxy_port": 8080,
                    "audio_base64": "UklGRg==",
                    "prompt_text": "secret prompt",
                    "nested": {"headers": {"authorization": "Bearer SECRET_TOKEN"}},
                }
            ],
        }

    app = create_app(
        auth_token="secret-token",
        completion_service=_SuccessfulService(),
        monitoring_summary_fetcher=fetch_summary,
    )
    client = TestClient(app)

    response = client.get("/admin/monitor/api/summary", headers={"Authorization": "Bearer secret-token"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["proxies"][0]["proxy_label"] == "proxy-a"
    assert payload["proxies"][0]["route_label"] == "route-a"
    assert payload["proxies"][0]["request_bytes"] == 128
    assert payload["proxies"][0]["response_bytes"] == 256
    assert payload["proxies"][0]["avg_response_body_ms"] == 70
    assert payload["proxies"][0]["p95_response_body_ms"] == 120
    assert payload["proxies"][0]["response_body_timeout_count"] == 1
    assert payload["proxies"][0]["timeout_kind"] == "read_timeout"
    assert '"response_body":' not in response.text
    assert "raw provider body" not in response.text
    assert "proxy_url" not in response.text
    assert "key_fingerprint" not in response.text
    assert "proxy_host" not in response.text
    assert "proxy_port" not in response.text
    assert "audio_base64" not in response.text
    assert "prompt_text" not in response.text
    assert "headers" not in response.text
    assert "authorization" not in response.text
    assert "SECRET_TOKEN" not in response.text


def test_monitor_summary_redacts_sensitive_values_under_allowed_keys() -> None:
    async def fetch_summary(_: MonitorWindow) -> dict[str, Any]:
        return {
            "total_requests": 1,
            "proxy_count": 1,
            "proxies": [
                {
                    "proxy_label": "http://proxy-user:proxy-pass@10.0.0.1:8080",
                    "route_label": "Bearer SECRET_TOKEN_VALUE",
                    "proxy_status": "api_key=AIzaSySecretSecretSecret",
                    "cooldown_until": "2026-06-20T10:00:00Z",
                    "total_requests": 1,
                }
            ],
        }

    app = create_app(
        auth_token="secret-token",
        completion_service=_SuccessfulService(),
        monitoring_summary_fetcher=fetch_summary,
    )
    client = TestClient(app)

    response = client.get("/admin/monitor/api/summary", headers={"Authorization": "Bearer secret-token"})

    proxy = response.json()["proxies"][0]
    assert response.status_code == 200
    assert proxy["proxy_label"] == "[redacted]"
    assert proxy["route_label"] == "[redacted]"
    assert proxy["proxy_status"] == "[redacted]"
    assert proxy["cooldown_until"] == "2026-06-20T10:00:00Z"
    assert "http://" not in response.text
    assert "proxy-pass" not in response.text
    assert "SECRET_TOKEN_VALUE" not in response.text
    assert "AIzaSy" not in response.text


def test_monitor_summary_preserves_proxy_p95_aggregate_keys_and_strips_raw_body() -> None:
    async def fetch_summary(_: MonitorWindow) -> dict[str, Any]:
        return {
            "total_requests": 2,
            "proxy_count": 1,
            "proxies": [
                {
                    "proxy_label": "proxy-a",
                    "max_route_p95_provider_total_ms": 240,
                    "max_route_p95_request_prepare_ms": 18,
                    "max_route_p95_response_headers_ms": 35,
                    "max_route_p95_response_body_ms": 120,
                    "max_route_p95_response_parse_ms": 15,
                    "response_body": "raw provider body SECRET_TOKEN",
                }
            ],
        }

    app = create_app(
        auth_token="secret-token",
        completion_service=_SuccessfulService(),
        monitoring_summary_fetcher=fetch_summary,
    )
    client = TestClient(app)

    response = client.get("/admin/monitor/api/summary", headers={"Authorization": "Bearer secret-token"})

    proxy = response.json()["proxies"][0]
    assert response.status_code == 200
    assert proxy["max_route_p95_provider_total_ms"] == 240
    assert proxy["max_route_p95_request_prepare_ms"] == 18
    assert proxy["max_route_p95_response_headers_ms"] == 35
    assert proxy["max_route_p95_response_body_ms"] == 120
    assert proxy["max_route_p95_response_parse_ms"] == 15
    assert '"response_body":' not in response.text
    assert "raw provider body" not in response.text
    assert "SECRET_TOKEN" not in response.text


def test_monitor_summary_invalid_minutes_uses_default_window() -> None:
    seen: dict[str, int] = {}

    async def fetch_summary(window: MonitorWindow) -> dict[str, Any]:
        seen["minutes"] = window.minutes
        return {"total_requests": 0, "proxy_count": 0, "proxies": []}

    app = create_app(
        auth_token="secret-token",
        completion_service=_SuccessfulService(),
        monitoring_summary_fetcher=fetch_summary,
    )
    client = TestClient(app)

    response = client.get(
        "/admin/monitor/api/summary?minutes=abc",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 200
    assert seen["minutes"] == 180
    assert response.json()["proxies"] == []


def test_monitor_summary_error_response_is_safe(caplog: Any) -> None:
    async def fetch_summary(_: MonitorWindow) -> dict[str, Any]:
        raise RuntimeError("raw SQL SECRET_TOKEN")

    app = create_app(
        auth_token="secret-token",
        completion_service=_SuccessfulService(),
        monitoring_summary_fetcher=fetch_summary,
        environment="test",
    )
    client = TestClient(app)

    with caplog.at_level(logging.ERROR, logger="gemini_gateway.api"):
        response = client.get("/admin/monitor/api/summary", headers={"Authorization": "Bearer secret-token"})

    assert response.status_code == 500
    assert response.json() == {"error": "Не удалось загрузить мониторинг, попробуйте позже"}
    assert "raw SQL" not in response.text
    assert "SECRET_TOKEN" not in response.text
    assert "raw SQL" not in caplog.text
    assert "SECRET_TOKEN" not in caplog.text
    [record] = [record for record in caplog.records if getattr(record, "event", None) == "gemini_gateway_monitor_error"]
    assert record.service == "gemini-gateway"
    assert record.environment == "test"
    assert record.status == "error"
    assert record.endpoint == "monitor_summary"
    assert record.error_type == "RuntimeError"
    assert record.error_message == "Не удалось загрузить мониторинг, попробуйте позже"


def test_monitor_timeseries_requires_bearer_token() -> None:
    app = create_app(auth_token="secret-token", completion_service=_SuccessfulService())
    client = TestClient(app)

    response = client.get("/admin/monitor/api/timeseries")

    assert response.status_code == 401
    assert response.json()["error"]


def test_monitor_timeseries_returns_safe_series_json() -> None:
    async def fetch_timeseries(window: MonitorWindow) -> dict[str, Any]:
        assert window.bucket_seconds == 60
        return {
            "bucket_seconds": window.bucket_seconds,
            "total_requests": 1,
            "series": [
                {
                    "bucket_start": "2026-06-20T10:00:00Z",
                    "proxy_label": "proxy-a",
                    "route_label": "route-a",
                    "total_requests": 1,
                }
            ],
        }

    app = create_app(
        auth_token="secret-token",
        completion_service=_SuccessfulService(),
        monitoring_timeseries_fetcher=fetch_timeseries,
    )
    client = TestClient(app)

    response = client.get("/admin/monitor/api/timeseries", headers={"Authorization": "Bearer secret-token"})

    assert response.status_code == 200
    assert response.json()["series"][0]["proxy_label"] == "proxy-a"


def test_monitor_timeseries_invalid_bucket_seconds_uses_default_window() -> None:
    seen: dict[str, int] = {}

    async def fetch_timeseries(window: MonitorWindow) -> dict[str, Any]:
        seen["bucket_seconds"] = window.bucket_seconds
        return {"bucket_seconds": window.bucket_seconds, "total_requests": 0, "series": []}

    app = create_app(
        auth_token="secret-token",
        completion_service=_SuccessfulService(),
        monitoring_timeseries_fetcher=fetch_timeseries,
    )
    client = TestClient(app)

    response = client.get(
        "/admin/monitor/api/timeseries?bucket_seconds=abc",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 200
    assert seen["bucket_seconds"] == 60
    assert response.json()["series"] == []


def test_monitor_timeseries_sanitizes_sensitive_key_variants() -> None:
    async def fetch_timeseries(_: MonitorWindow) -> dict[str, Any]:
        return {
            "bucket_seconds": 60,
            "total_requests": 1,
            "series": [
                {
                    "bucket_start": "2026-06-20T10:00:00Z",
                    "proxy_label": "proxy-a",
                    "route_label": "route-a",
                    "request_bytes": 10,
                    "response_bytes": 20,
                    "avg_response_body_ms": 70,
                    "p95_response_body_ms": 120,
                    "response_body_timeout_count": 1,
                    "response_body": "raw provider body SECRET_TOKEN",
                    "proxy_url": "http://secret-proxy",
                    "key_fingerprint": "fingerprint-secret",
                    "proxy_host": "10.0.0.1",
                    "proxy_port": 8080,
                    "audio_base64": "UklGRg==",
                    "prompt_text": "secret prompt",
                    "nested": {"headers": {"authorization": "Bearer SECRET_TOKEN"}},
                }
            ],
        }

    app = create_app(
        auth_token="secret-token",
        completion_service=_SuccessfulService(),
        monitoring_timeseries_fetcher=fetch_timeseries,
    )
    client = TestClient(app)

    response = client.get("/admin/monitor/api/timeseries", headers={"Authorization": "Bearer secret-token"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["series"][0]["proxy_label"] == "proxy-a"
    assert payload["series"][0]["request_bytes"] == 10
    assert payload["series"][0]["response_bytes"] == 20
    assert payload["series"][0]["avg_response_body_ms"] == 70
    assert payload["series"][0]["p95_response_body_ms"] == 120
    assert payload["series"][0]["response_body_timeout_count"] == 1
    assert '"response_body":' not in response.text
    assert "raw provider body" not in response.text
    assert "proxy_url" not in response.text
    assert "key_fingerprint" not in response.text
    assert "proxy_host" not in response.text
    assert "proxy_port" not in response.text
    assert "audio_base64" not in response.text
    assert "prompt_text" not in response.text
    assert "headers" not in response.text
    assert "authorization" not in response.text
    assert "SECRET_TOKEN" not in response.text


def test_monitor_timeseries_error_response_is_safe(caplog: Any) -> None:
    async def fetch_timeseries(_: MonitorWindow) -> dict[str, Any]:
        raise RuntimeError("raw SQL SECRET_TOKEN")

    app = create_app(
        auth_token="secret-token",
        completion_service=_SuccessfulService(),
        monitoring_timeseries_fetcher=fetch_timeseries,
        environment="test",
    )
    client = TestClient(app)

    with caplog.at_level(logging.ERROR, logger="gemini_gateway.api"):
        response = client.get("/admin/monitor/api/timeseries", headers={"Authorization": "Bearer secret-token"})

    assert response.status_code == 500
    assert response.json() == {"error": "Не удалось загрузить мониторинг, попробуйте позже"}
    assert "raw SQL" not in response.text
    assert "SECRET_TOKEN" not in response.text
    assert "raw SQL" not in caplog.text
    assert "SECRET_TOKEN" not in caplog.text
    [record] = [record for record in caplog.records if getattr(record, "event", None) == "gemini_gateway_monitor_error"]
    assert record.endpoint == "monitor_timeseries"
    assert record.error_type == "RuntimeError"
