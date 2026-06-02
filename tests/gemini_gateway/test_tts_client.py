from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gemini_gateway.contracts import GatewayTTSRequest
from gemini_gateway.errors import GatewayError
from gemini_gateway.tts_client import GeminiTTSClient

_PROXY_URL = "http://user:pass@127.0.0.1:8000"


@pytest.mark.asyncio
async def test_tts_client_forwards_native_payload_and_preserves_audio_response() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["api_key"] = request.headers.get("x-goog-api-key")
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "responseId": "tts-1",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/wav",
                                        "data": "UklGRg==",
                                    }
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 9,
                    "candidatesTokenCount": 12,
                    "totalTokenCount": 21,
                },
            },
        )

    client = GeminiTTSClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    response = await client.synthesize(
        request=GatewayTTSRequest(
            request_id="req-tts",
            source_service="voice_tts",
            model="google/gemini-3.1-flash-tts-preview",
            text="привет",
            voice_name="Kore",
        ),
        api_key="AIza-test-key",
        proxy_url=_PROXY_URL,
    )

    assert seen["url"] == "https://example.test/v1beta/models/gemini-3.1-flash-tts-preview:generateContent"
    assert seen["api_key"] == "AIza-test-key"
    assert seen["json"]["contents"][0]["parts"][0]["text"] == "привет"
    assert seen["json"]["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert seen["json"]["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"] == {
        "voiceName": "Kore"
    }
    assert response.audio_base64 == "UklGRg=="
    assert response.audio_mime_type == "audio/wav"
    assert response.generation_id == "tts-1"
    assert response.usage == {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}


@pytest.mark.asyncio
async def test_tts_client_ignores_bool_usage_metadata() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "responseId": "tts-bool-usage",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/wav",
                                        "data": "UklGRg==",
                                    }
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": True,
                    "candidatesTokenCount": False,
                    "totalTokenCount": "21",
                },
            },
        )

    client = GeminiTTSClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    response = await client.synthesize(
        request=GatewayTTSRequest(
            request_id="req-tts-bool-usage",
            source_service="voice_tts",
            model="google/gemini-3.1-flash-tts-preview",
            text="привет",
        ),
        api_key="AIza-test-key",
        proxy_url=_PROXY_URL,
    )

    assert response.usage == {"prompt_tokens": None, "completion_tokens": None, "total_tokens": 21}


@pytest.mark.asyncio
async def test_tts_client_does_not_forward_raw_provider_metadata() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "responseId": "tts-metadata",
                "modelVersion": "raw-internal-version",
                "diagnostic": "raw provider trace must not leave gateway",
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/wav",
                                        "data": "UklGRg==",
                                    }
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 1, "totalTokenCount": 1},
            },
        )

    client = GeminiTTSClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    response = await client.synthesize(
        request=GatewayTTSRequest(
            request_id="req-tts-metadata",
            source_service="voice_tts",
            model="google/gemini-3.1-flash-tts-preview",
            text="привет",
        ),
        api_key="AIza-test-key",
        proxy_url=_PROXY_URL,
    )

    assert response.provider_specific_fields == {}
    assert "raw provider trace" not in str(response.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_tts_client_maps_provider_error_without_raw_secret() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "quota failed for AIza-raw-secret"}})

    client = GeminiTTSClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.synthesize(
            request=GatewayTTSRequest(
                request_id="req-tts-error",
                source_service="voice_tts",
                model="google/gemini-3.1-flash-tts-preview",
                text="привет",
            ),
            api_key="AIza-test-key",
            proxy_url=_PROXY_URL,
        )

    error = exc_info.value
    assert error.reason == "rate_limited"
    assert error.retryable is True
    assert error.provider_status_code == 429
    assert "AIza-raw-secret" not in str(error.provider_message_safe)


@pytest.mark.asyncio
async def test_tts_client_rejects_success_payload_with_embedded_provider_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "responseId": "tts-embedded-error",
                "error": {"code": 500, "message": "HTTP 500 Authorization: Bearer sk-secret123456"},
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/wav",
                                        "data": "UklGRg==",
                                    }
                                }
                            ]
                        },
                    }
                ],
            },
        )

    client = GeminiTTSClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.synthesize(
            request=GatewayTTSRequest(
                request_id="req-tts-embedded-error",
                source_service="voice_tts",
                model="google/gemini-3.1-flash-tts-preview",
                text="привет",
            ),
            api_key="AIza-test-key",
            proxy_url=_PROXY_URL,
        )

    error = exc_info.value
    assert error.reason == "provider_unavailable"
    assert error.retryable is True
    assert error.provider_called is True
    assert "sk-secret123456" not in str(error.provider_message_safe)


@pytest.mark.asyncio
async def test_tts_client_maps_proxy_transport_error_to_proxy_failed() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("proxy connection failed", request=request)

    client = GeminiTTSClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.synthesize(
            request=GatewayTTSRequest(
                request_id="req-tts-proxy-transport",
                source_service="voice_tts",
                model="google/gemini-3.1-flash-tts-preview",
                text="привет",
            ),
            api_key="AIza-test-key",
            proxy_url=_PROXY_URL,
        )

    error = exc_info.value
    assert error.reason == "proxy_failed"
    assert error.retryable is True
    assert error.provider_called is False


@pytest.mark.asyncio
async def test_tts_client_records_stable_timeout_kind_without_raw_transport_text() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.PoolTimeout("pool timeout with api_key=AIza-secret", request=request)

    client = GeminiTTSClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.synthesize(
            request=GatewayTTSRequest(
                request_id="req-tts-pool-timeout",
                source_service="voice_tts",
                model="google/gemini-3.1-flash-tts-preview",
                text="привет",
            ),
            api_key="AIza-secret",
            proxy_url=_PROXY_URL,
        )

    error = exc_info.value
    assert error.reason == "network_timeout"
    assert error.provider_message_safe == "pool_timeout"
    assert "AIza-secret" not in str(error.provider_message_safe)


@pytest.mark.asyncio
async def test_tts_client_marks_malformed_success_as_provider_called() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "responseId": "tts-malformed-success",
                "candidates": [{"content": {"parts": [{"text": "no audio"}]}}],
            },
        )

    client = GeminiTTSClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.synthesize(
            request=GatewayTTSRequest(
                request_id="req-tts-malformed-success",
                source_service="voice_tts",
                model="google/gemini-3.1-flash-tts-preview",
                text="hi",
            ),
            api_key="AIza-test-key",
            proxy_url=_PROXY_URL,
        )

    assert exc_info.value.reason == "invalid_response"
    assert exc_info.value.retryable is False
    assert exc_info.value.provider_called is True


@pytest.mark.asyncio
async def test_tts_client_maps_gemini_400_quota_wording_to_invalid_response() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "invalid response rate config"}})

    client = GeminiTTSClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.synthesize(
            request=GatewayTTSRequest(
                request_id="req-tts-400",
                source_service="voice_tts",
                model="google/gemini-3.1-flash-tts-preview",
                text="привет",
            ),
            api_key="AIza-test-key",
            proxy_url=_PROXY_URL,
        )

    assert exc_info.value.reason == "invalid_response"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_tts_client_rejects_missing_proxy_url_without_provider_call() -> None:
    called = False

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(
            200,
            json={
                "responseId": "tts-must-not-run-direct",
                "candidates": [{"content": {"parts": [{"inlineData": {"data": "UklGRg=="}}]}}],
            },
        )

    client = GeminiTTSClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.synthesize(
            request=GatewayTTSRequest(
                request_id="req-tts-missing-proxy",
                source_service="voice_tts",
                model="google/gemini-3.1-flash-tts-preview",
                text="привет",
            ),
            api_key="AIza-test-key",
            proxy_url=None,
        )

    assert called is False
    assert exc_info.value.reason == "no_route"
    assert exc_info.value.retryable is True
    assert exc_info.value.provider_called is False
