from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
import pytest

from gemini_gateway.contracts import GatewayChatRequest
from gemini_gateway.errors import GatewayError
from gemini_gateway.openlux_chat_client import OpenLuxChatClient
from gemini_gateway.openlux_pricing import OpenLuxPricingSnapshot


def _sse_response(*events: dict[str, Any], headers: dict[str, str] | None = None) -> httpx.Response:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx.Response(
        200,
        headers={"Content-Type": "text/event-stream", **(headers or {})},
        content=body.encode("utf-8"),
    )


@pytest.mark.asyncio
async def test_openlux_client_streams_tools_and_preserves_openai_payload() -> None:
    captured: dict[str, Any] = {}

    class _PricingCatalog:
        async def get_snapshot(self, model: str) -> OpenLuxPricingSnapshot:
            assert model == "gemini-3.8-flash"
            return OpenLuxPricingSnapshot(
                model=model,
                fetched_at=datetime(2026, 8, 27, tzinfo=UTC),
                input_usd_per_token=Decimal("0.0000000551475"),
                output_usd_per_token=Decimal("0.0000002757375"),
                metadata={"group": "Anti-Gemini-1"},
            )

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        return _sse_response(
            {
                "id": "chatcmpl-openlux",
                "model": "gemini-3.8-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {"name": "weather", "arguments": "{\"city\":"},
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": "\"Moscow\"}"}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 21, "completion_tokens": 8, "total_tokens": 29},
            },
            headers={"x-api-request-id": "openlux-request-123"},
        )

    client = OpenLuxChatClient(
        base_url="https://api.openlux.ai/v1",
        transport=httpx.MockTransport(handler),
        pricing_catalog=_PricingCatalog(),
    )
    request = GatewayChatRequest(
        request_id="req-openlux-tool",
        source_service="test",
        model="google/gemini-3.6-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Что на картинке?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
                ],
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Погода",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        tool_choice="required",
        reasoning={"effort": "low"},
        metadata={"trace": "safe"},
    )

    response = await client.complete(
        request=request,
        api_key="sk-openlux-test-secret",
        model="gemini-3.8-flash",
    )

    assert captured["url"] == "https://api.openlux.ai/v1/chat/completions"
    assert captured["authorization"] == "Bearer sk-openlux-test-secret"
    assert captured["payload"]["model"] == "gemini-3.8-flash"
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["stream_options"] == {"include_usage": True}
    assert captured["payload"]["tools"] == request.tools
    assert captured["payload"]["tool_choice"] == "required"
    assert captured["payload"]["reasoning_effort"] == "low"
    assert captured["payload"]["messages"] == request.messages
    assert response.provider_request_id == "openlux-request-123"
    assert response.generation_id == "chatcmpl-openlux"
    assert response.model == "gemini-3.8-flash"
    assert response.finish_reason == "tool_calls"
    assert response.usage == {
        "prompt_tokens": 21,
        "completion_tokens": 8,
        "total_tokens": 29,
        "cost": 3.363998e-06,
        "cost_details": {"upstream_inference_cost": 3.363998e-06},
    }
    assert response.pricing_snapshot_json["metadata"] == {"group": "Anti-Gemini-1"}
    assert response.choices == [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{\"city\":\"Moscow\"}"},
                    }
                ],
            },
        }
    ]


@pytest.mark.asyncio
async def test_openlux_client_adapts_response_schema_to_gemini_subset() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _sse_response(
            {
                "id": "chatcmpl-openlux-structured",
                "model": "gemini-3.8-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": '{"operations":[]}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            }
        )

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "memory_reflection_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "maxItems": 12,
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": ["string", "null"], "maxLength": 320},
                                "weight": {"type": "number", "multipleOf": 0.1},
                            },
                        },
                    },
                    "no_changes_reason": {"type": ["string", "null"], "maxLength": 500},
                },
            },
        },
    }
    client = OpenLuxChatClient(transport=httpx.MockTransport(handler))
    request = GatewayChatRequest(
        request_id="req-openlux-structured",
        source_service="memory_reflection",
        model="google/gemini-3.6-flash",
        messages=[{"role": "user", "content": "Обнови память"}],
        response_format=response_format,
    )

    await client.complete(
        request=request,
        api_key="sk-openlux-test-secret",
        model="gemini-3.8-flash",
    )

    assert captured["payload"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "memory_reflection_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "nullable": True, "maxLength": 320},
                                "weight": {"type": "number"},
                            },
                        },
                    },
                    "no_changes_reason": {"type": "string", "nullable": True, "maxLength": 500},
                },
            },
        },
    }


@pytest.mark.asyncio
async def test_openlux_client_keeps_tool_calls_separate_when_provider_restarts_index() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _sse_response(
            {
                "id": "chatcmpl-openlux-restarted-tool-index",
                "model": "gemini-3.8-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-first",
                                    "type": "function",
                                    "function": {
                                        "name": "send_message",
                                        "arguments": '{"text":"Первый ответ","reply_to_id":107254}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-second",
                                    "type": "function",
                                    "function": {
                                        "name": "send_message",
                                        "arguments": '{"text":"Второй ответ"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    client = OpenLuxChatClient(transport=httpx.MockTransport(handler))
    request = GatewayChatRequest(
        request_id="req-openlux-restarted-tool-index",
        source_service="test",
        model="google/gemini-3.6-flash",
        messages=[{"role": "user", "content": "Ответь двумя сообщениями"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "Отправляет сообщение",
                    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
                },
            }
        ],
        tool_choice="required",
    )

    response = await client.complete(
        request=request,
        api_key="sk-openlux-test-secret",
        model="gemini-3.8-flash",
    )

    assert response.choices[0]["message"]["tool_calls"] == [
        {
            "id": "call-first",
            "type": "function",
            "function": {
                "name": "send_message",
                "arguments": '{"text":"Первый ответ","reply_to_id":107254}',
            },
        },
        {
            "id": "call-second",
            "type": "function",
            "function": {"name": "send_message", "arguments": '{"text":"Второй ответ"}'},
        },
    ]


@pytest.mark.asyncio
async def test_openlux_client_stops_oversized_stream_without_exposing_content() -> None:
    secret_fragment = "private-provider-output-" * 200

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _sse_response(
            {
                "id": "chatcmpl-large",
                "choices": [{"index": 0, "delta": {"content": secret_fragment}}],
            }
        )

    client = OpenLuxChatClient(
        transport=httpx.MockTransport(handler),
        max_stream_bytes=512,
    )
    request = GatewayChatRequest(
        request_id="req-openlux-large",
        source_service="test",
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "hi"}],
    )

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(request=request, api_key="sk-openlux-test-secret", model="gemini-3.8-flash")

    error = exc_info.value
    assert error.reason == "invalid_response"
    assert error.retryable is False
    assert secret_fragment not in str(error)
    assert secret_fragment not in str(error.provider_message_safe)


@pytest.mark.asyncio
async def test_openlux_client_opens_cooldown_after_rate_limit() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        del request
        calls += 1
        return httpx.Response(
            429,
            headers={"Retry-After": "30"},
            json={"error": {"message": "rate limit exceeded"}},
        )

    client = OpenLuxChatClient(transport=httpx.MockTransport(handler))
    request = GatewayChatRequest(
        request_id="req-openlux-rate-limit",
        source_service="test",
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "hi"}],
    )

    with pytest.raises(GatewayError) as first_exc:
        await client.complete(request=request, api_key="sk-openlux-test-secret", model="gemini-3.8-flash")
    with pytest.raises(GatewayError) as second_exc:
        await client.complete(request=request, api_key="sk-openlux-test-secret", model="gemini-3.8-flash")

    assert first_exc.value.reason == "rate_limited"
    assert first_exc.value.retry_after_seconds == 30
    assert first_exc.value.provider_called is True
    assert second_exc.value.reason == "cooldown_active"
    assert second_exc.value.provider_called is False
    assert calls == 1


@pytest.mark.asyncio
async def test_openlux_client_marks_http_400_retryable_without_opening_cooldown() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        del request
        calls += 1
        return httpx.Response(
            400,
            json={"error": {"code": 400, "message": "Invalid JSON payload received"}},
        )

    client = OpenLuxChatClient(transport=httpx.MockTransport(handler))
    request = GatewayChatRequest(
        request_id="req-openlux-invalid-json",
        source_service="test",
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "hi"}],
    )

    with pytest.raises(GatewayError) as first_exc:
        await client.complete(request=request, api_key="sk-openlux-test-secret", model="gemini-3.8-flash")
    with pytest.raises(GatewayError) as second_exc:
        await client.complete(request=request, api_key="sk-openlux-test-secret", model="gemini-3.8-flash")

    assert first_exc.value.reason == "invalid_response"
    assert first_exc.value.retryable is True
    assert first_exc.value.provider_status_code == 400
    assert second_exc.value.provider_called is True
    assert calls == 2


@pytest.mark.asyncio
async def test_openlux_client_classifies_exhausted_balance_as_quota_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            402,
            json={"error": {"message": "insufficient quota"}},
        )

    client = OpenLuxChatClient(transport=httpx.MockTransport(handler))
    request = GatewayChatRequest(
        request_id="req-openlux-no-balance",
        source_service="test",
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "hi"}],
    )

    with pytest.raises(GatewayError) as exc_info:
        await client.complete(request=request, api_key="sk-openlux-test-secret", model="gemini-3.8-flash")

    assert exc_info.value.reason == "quota_exhausted"
    assert exc_info.value.provider_status_code == 402
    assert exc_info.value.provider_called is True
