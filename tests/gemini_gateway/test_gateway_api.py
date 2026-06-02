from __future__ import annotations

import logging
from typing import Any

from fastapi.testclient import TestClient

from gemini_gateway.api import create_app
from gemini_gateway.contracts import GatewayChatResponse, GatewayTTSResponse
from gemini_gateway.errors import GatewayError


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
        )

    async def synthesize_speech(self, request: Any) -> GatewayTTSResponse:
        return GatewayTTSResponse(
            request_id=request.request_id,
            generation_id="tts-1",
            model=request.model,
            audio_base64="UklGRg==",
            audio_mime_type="audio/wav",
            usage={"prompt_tokens": 3, "total_tokens": 3},
        )


class _FailingService:
    async def complete(self, request: Any) -> GatewayChatResponse:
        raise GatewayError(
            reason="rate_limited",
            retryable=True,
            provider_status_code=429,
            provider_message_safe="HTTP 429 raw secret",
            request_id=request.request_id,
        )

    async def synthesize_speech(self, request: Any) -> GatewayTTSResponse:
        raise GatewayError(
            reason="no_route",
            retryable=True,
            provider_message_safe="raw route details",
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
    }
    assert "raw route details" not in response.text


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
    assert response.json()["reason"] == "no_route"
    assert response.json()["retry_after_seconds"] == 900


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
