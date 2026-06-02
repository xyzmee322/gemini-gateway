from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gemini_gateway.gemini_client import GeminiOpenAIClient
from gemini_gateway.repository import GatewayChatRequest, GatewayError

_PROXY_URL = "http://user:pass@127.0.0.1:8000"


@pytest.mark.asyncio
async def test_client_forwards_openai_payload_and_preserves_provider_response() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "model": "gemini-3.5-flash",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [{"id": "call-1", "type": "function"}],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "safety_ratings": [{"category": "safe"}],
                "reasoning": {"summary": "kept"},
                "provider_specific_fields": {"thought_signature": "opaque"},
            },
        )

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    response = await client.complete(
        request=GatewayChatRequest(
            request_id="req-1",
            source_service="test",
            model="google/gemini-3.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
                }
            ],
            tool_choice={"type": "function", "function": {"name": "lookup"}},
            response_format={"type": "json_object"},
            reasoning={"effort": "low"},
        ),
        api_key="AIza-test-key",
        proxy_url=_PROXY_URL,
    )

    assert seen["url"] == "https://example.test/openai/chat/completions"
    assert seen["auth"] == "Bearer AIza-test-key"
    assert seen["json"]["model"] == "gemini-3.5-flash"
    assert seen["json"]["tools"][0]["function"]["name"] == "lookup"
    assert seen["json"]["tool_choice"]["function"]["name"] == "lookup"
    assert seen["json"]["response_format"] == {"type": "json_object"}
    assert seen["json"]["reasoning_effort"] == "low"
    assert "reasoning" not in seen["json"]
    assert "proxy_url" not in seen["json"]
    assert response.generation_id == "gen-1"
    assert response.choices[0]["message"]["tool_calls"][0]["id"] == "call-1"
    assert response.usage["total_tokens"] == 15
    assert response.finish_reason == "tool_calls"
    assert response.raw_response["safety_ratings"][0]["category"] == "safe"
    assert response.provider_specific_fields["thought_signature"] == "opaque"
    assert "reasoning" not in response.provider_specific_fields


@pytest.mark.asyncio
async def test_client_sanitizes_success_provider_specific_fields() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-provider-fields",
                "model": "gemini-3.5-flash",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "provider_specific_fields": {
                            "native_finish_reason": "STOP",
                            "diagnostic": "raw prompt fragment must not leave gateway",
                        },
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "provider_specific_fields": {
                                "reasoning_details": [{"type": "reasoning.text", "text": "keep"}],
                                "diagnostic": "raw provider trace must not leave gateway",
                            },
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "send_message", "arguments": "{}"},
                                    "provider_specific_fields": {
                                        "thought_signature": "call-signature",
                                        "google": {
                                            "thought_signature": "google-signature",
                                            "diagnostic": "raw provider trace",
                                        },
                                        "diagnostic": "raw provider trace",
                                    },
                                    "extra_content": {
                                        "google": {
                                            "thought_signature": "extra-signature",
                                            "diagnostic": "raw provider trace",
                                        },
                                        "diagnostic": "raw provider trace",
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {},
                "safety_ratings": [{"raw": "raw prompt fragment"}],
                "provider_specific_fields": {
                    "thought_signature": "top-signature",
                    "google": {
                        "thought_signature": "top-google-signature",
                        "diagnostic": "raw provider trace",
                    },
                    "reasoning_details": [{"type": "reasoning.text", "text": "top keep"}],
                    "diagnostic": "raw provider trace",
                },
            },
        )

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    response = await client.complete(
        request=GatewayChatRequest(
            request_id="req-provider-fields",
            source_service="test",
            model="gemini-3.5-flash",
            messages=[{"role": "user", "content": "hi"}],
        ),
        api_key="AIza-test-key",
        proxy_url=_PROXY_URL,
    )

    assert response.provider_specific_fields == {
        "thought_signature": "top-signature",
        "google": {"thought_signature": "top-google-signature"},
        "reasoning_details": [{"type": "reasoning.text", "text": "top keep"}],
    }
    assert "safety_ratings" not in response.provider_specific_fields

    choice = response.choices[0]
    message = choice["message"]
    tool_call = message["tool_calls"][0]

    assert "provider_specific_fields" not in choice
    assert message["provider_specific_fields"] == {
        "reasoning_details": [{"type": "reasoning.text", "text": "keep"}]
    }
    assert tool_call["provider_specific_fields"] == {
        "thought_signature": "call-signature",
        "google": {"thought_signature": "google-signature"},
    }
    assert tool_call["extra_content"] == {"google": {"thought_signature": "extra-signature"}}
    assert "raw provider trace" not in str(response.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_client_maps_litellm_reasoning_to_openai_reasoning_effort() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "gen-no-reasoning",
                "model": "gemini-3.5-flash",
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {},
            },
        )

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    await client.complete(
        request=GatewayChatRequest(
            request_id="req-no-reasoning",
            source_service="test",
            model="google/gemini-3.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            reasoning={"effort": "high"},
        ),
        api_key="AIza-test-key",
        proxy_url=_PROXY_URL,
    )

    assert seen["json"]["reasoning_effort"] == "high"
    assert "reasoning" not in seen["json"]


@pytest.mark.asyncio
async def test_client_normalizes_tool_history_for_gemini_openai_payload() -> None:
    seen: dict[str, Any] = {}
    thought_signature = "x" * 512
    raw_call_id = f"call-send__thought__{thought_signature}"

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "gen-history",
                "model": "gemini-3.5-flash",
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {},
            },
        )

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    await client.complete(
        request=GatewayChatRequest(
            request_id="req-history",
            source_service="test",
            model="google/gemini-3.5-flash",
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_details": [{"type": "reasoning.text", "text": "keep out"}],
                    "tool_calls": [
                        {
                            "id": raw_call_id,
                            "type": "function",
                            "index": 0,
                            "function": {"name": "send_message", "arguments": None},
                            "provider_specific_fields": {"thought_signature": thought_signature},
                            "extra_content": {"google": {"thought_signature": thought_signature}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": raw_call_id, "content": '{"status":"success"}'},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "send_message", "parameters": {"type": "object", "properties": {}}},
                }
            ],
            tool_choice="auto",
        ),
        api_key="AIza-test-key",
        proxy_url=_PROXY_URL,
    )

    assistant_message = seen["json"]["messages"][1]
    tool_call = assistant_message["tool_calls"][0]
    tool_message = seen["json"]["messages"][2]

    assert tool_call == {
        "id": "call-send",
        "type": "function",
        "function": {"name": "send_message", "arguments": "{}"},
        "extra_content": {"google": {"thought_signature": thought_signature}},
    }
    assert tool_message["tool_call_id"] == "call-send"
    assert tool_message["name"] == "send_message"
    assert "reasoning_details" not in assistant_message
    assert "provider_specific_fields" not in tool_call


@pytest.mark.asyncio
async def test_client_normalizes_multimodal_tool_result_for_gemini_openai_payload() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "gen-profile-photo",
                "model": "gemini-3.5-flash",
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {},
            },
        )

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    await client.complete(
        request=GatewayChatRequest(
            request_id="req-profile-photo",
            source_service="test",
            model="google/gemini-3.5-flash",
            messages=[
                {"role": "user", "content": "чекни аву"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-photo",
                            "type": "function",
                            "function": {"name": "view_profile_photo", "arguments": '{"reply_to_id":123}'},
                            "extra_content": {"google": {"thought_signature": "sig"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-photo",
                    "content": [
                        {"type": "text", "text": '{"status":"success","images_count":2}'},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,def"}},
                    ],
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "view_profile_photo",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            tool_choice="auto",
        ),
        api_key="AIza-test-key",
        proxy_url=_PROXY_URL,
    )

    tool_message = seen["json"]["messages"][2]
    image_followup = seen["json"]["messages"][3]

    assert tool_message == {
        "role": "tool",
        "tool_call_id": "call-photo",
        "name": "view_profile_photo",
        "content": '{"status":"success","images_count":2}',
    }
    assert image_followup["role"] == "user"
    assert [part["type"] for part in image_followup["content"]] == ["text", "image_url", "image_url"]
    assert "data:image" not in tool_message["content"]


@pytest.mark.asyncio
async def test_client_raises_safe_gateway_error_for_provider_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "HTTP 429 quota for key AIza-secret", "status": "RESOURCE_EXHAUSTED"}},
        )

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(
            request=GatewayChatRequest(
                request_id="req-2",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            ),
            api_key="AIza-secret",
            proxy_url=_PROXY_URL,
        )

    error = exc_info.value
    assert error.reason == "rate_limited"
    assert error.retryable is True
    assert error.status_code == 429
    assert error.public_message
    assert "AIza-secret" not in str(error)
    assert "HTTP 429" not in error.public_message


@pytest.mark.asyncio
async def test_client_maps_gemini_400_to_non_cooling_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "invalid function response turn"}})

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(
            request=GatewayChatRequest(
                request_id="req-400",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            ),
            api_key="AIza-secret",
            proxy_url=_PROXY_URL,
        )

    assert exc_info.value.reason == "invalid_response"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_client_maps_gemini_400_api_key_error_to_auth_failed() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "API key not valid. Please pass a valid API key. api_key=AIza-secret"}},
        )

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(
            request=GatewayChatRequest(
                request_id="req-400-api-key",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            ),
            api_key="AIza-secret",
            proxy_url=_PROXY_URL,
        )

    assert exc_info.value.reason == "auth_failed"
    assert exc_info.value.retryable is False
    assert "AIza-secret" not in str(exc_info.value.provider_message_safe)


@pytest.mark.asyncio
async def test_client_maps_gemini_400_quota_wording_to_invalid_response() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "invalid request rate config"}})

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(
            request=GatewayChatRequest(
                request_id="req-400-rate-word",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            ),
            api_key="AIza-secret",
            proxy_url=_PROXY_URL,
        )

    assert exc_info.value.reason == "invalid_response"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_client_maps_gemini_400_safety_to_content_filtered() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "blocked by SAFETY policy"}})

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(
            request=GatewayChatRequest(
                request_id="req-safety",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            ),
            api_key="AIza-secret",
            proxy_url=_PROXY_URL,
        )

    assert exc_info.value.reason == "content_filtered"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_client_maps_gemini_200_content_filter_finish_reason_to_content_filtered() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-filtered",
                "model": "gemini-3.1-flash-lite",
                "choices": [{"index": 0, "finish_reason": "content_filter: PROHIBITED_CONTENT"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
            },
        )

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(
            request=GatewayChatRequest(
                request_id="req-content-filter-finish",
                source_service="test",
                model="gemini-3.1-flash-lite",
                messages=[{"role": "user", "content": "hi"}],
            ),
            api_key="AIza-secret",
            proxy_url=_PROXY_URL,
        )

    assert exc_info.value.reason == "content_filtered"
    assert exc_info.value.retryable is False
    assert exc_info.value.provider_called is True
    assert exc_info.value.provider_message_safe == "content_filter: PROHIBITED_CONTENT"


@pytest.mark.asyncio
async def test_client_rejects_success_payload_with_top_level_provider_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-embedded-error",
                "model": "gemini-3.5-flash",
                "error": {"code": 500, "message": "HTTP 500 Traceback Authorization: Bearer sk-secret123456"},
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {},
            },
        )

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(
            request=GatewayChatRequest(
                request_id="req-embedded-top-level-error",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            ),
            api_key="AIza-secret",
            proxy_url=_PROXY_URL,
        )

    error = exc_info.value
    assert error.reason == "provider_unavailable"
    assert error.retryable is True
    assert "sk-secret123456" not in str(error.provider_message_safe)


@pytest.mark.asyncio
async def test_client_maps_proxy_transport_error_to_proxy_failed() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("proxy connection failed", request=request)

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(
            request=GatewayChatRequest(
                request_id="req-proxy-transport",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            ),
            api_key="AIza-secret",
            proxy_url=_PROXY_URL,
        )

    error = exc_info.value
    assert error.reason == "proxy_failed"
    assert error.retryable is True
    assert error.provider_called is False


@pytest.mark.asyncio
async def test_client_records_stable_timeout_kind_without_raw_transport_text() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("raw timeout with api_key=AIza-secret", request=request)

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(
            request=GatewayChatRequest(
                request_id="req-read-timeout",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            ),
            api_key="AIza-secret",
            proxy_url=_PROXY_URL,
        )

    error = exc_info.value
    assert error.reason == "network_timeout"
    assert error.provider_message_safe == "read_timeout"
    assert "AIza-secret" not in str(error.provider_message_safe)


@pytest.mark.asyncio
async def test_client_rejects_success_payload_with_choice_provider_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-choice-error",
                "model": "gemini-3.5-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "provider_specific_fields": {
                            "error": {"code": 429, "message": "quota failed for api_key=AIza-secret"}
                        },
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {},
            },
        )

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(
            request=GatewayChatRequest(
                request_id="req-choice-error",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            ),
            api_key="AIza-secret",
            proxy_url=_PROXY_URL,
        )

    error = exc_info.value
    assert error.reason == "rate_limited"
    assert error.retryable is True
    assert "AIza-secret" not in str(error.provider_message_safe)


@pytest.mark.asyncio
async def test_client_rejects_malformed_success_without_choices() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "gen-bad", "model": "gemini-3.5-flash"})

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(
            request=GatewayChatRequest(
                request_id="req-malformed",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            ),
            api_key="AIza-secret",
            proxy_url=_PROXY_URL,
        )

    assert exc_info.value.reason == "invalid_response"
    assert exc_info.value.retryable is False
    assert exc_info.value.provider_called is True


@pytest.mark.asyncio
async def test_client_preserves_retry_after_header() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "600"},
            json={"error": {"message": "quota exhausted", "status": "RESOURCE_EXHAUSTED"}},
        )

    client = GeminiOpenAIClient(base_url="https://example.test/openai", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(
            request=GatewayChatRequest(
                request_id="req-retry-after",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            ),
            api_key="AIza-secret",
            proxy_url=_PROXY_URL,
        )

    assert exc_info.value.retry_after_seconds == 600


@pytest.mark.asyncio
async def test_client_applies_request_timeout_to_upstream_call() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = request.extensions["timeout"]
        return httpx.Response(
            200,
            json={
                "id": "gen-timeout",
                "model": "gemini-3.5-flash",
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {},
            },
        )

    client = GeminiOpenAIClient(
        base_url="https://example.test/openai",
        timeout=35,
        transport=httpx.MockTransport(handler),
    )

    await client.complete(
        request=GatewayChatRequest(
            request_id="req-timeout",
            source_service="test",
            model="gemini-3.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            timeout_seconds=7,
        ),
        api_key="AIza-secret",
        proxy_url=_PROXY_URL,
    )

    assert seen["timeout"]["read"] == 7


@pytest.mark.asyncio
async def test_client_rejects_missing_proxy_url_without_provider_call() -> None:
    called = False

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(
            200,
            json={
                "id": "gen-must-not-run-direct",
                "model": "gemini-3.5-flash",
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {},
            },
        )

    client = GeminiOpenAIClient(
        base_url="https://example.test/openai",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(
            request=GatewayChatRequest(
                request_id="req-missing-proxy",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            ),
            api_key="AIza-secret",
            proxy_url=None,
        )

    assert called is False
    assert exc_info.value.reason == "no_route"
    assert exc_info.value.retryable is True
    assert exc_info.value.provider_called is False
