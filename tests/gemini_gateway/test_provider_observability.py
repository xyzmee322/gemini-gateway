import json

import httpx
import pytest

from gemini_gateway.provider_observability import (
    classify_chat_payload,
    classify_embedding_payload,
    classify_tts_payload,
    provider_timing_columns,
    send_timed_json,
    timing_to_dict,
)


class _FailingBodyStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self):
        raise httpx.ReadTimeout("body timeout")
        yield b""

    async def aclose(self) -> None:
        self.closed = True


def test_classify_chat_payload_detects_tool_image_payload() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "photo"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
                ],
            },
        ]
    }

    assert classify_chat_payload(payload).model_dump() == {
        "operation_type": "chat",
        "payload_kind": "media",
        "media_count": 1,
        "image_count": 1,
    }


def test_classify_embedding_payload_detects_image_parts() -> None:
    payload = {
        "content": {
            "parts": [
                {"text": "cat"},
                {"inlineData": {"mimeType": "image/jpeg", "data": "abc"}},
            ]
        }
    }

    assert classify_embedding_payload(payload).model_dump() == {
        "operation_type": "embedding",
        "payload_kind": "media",
        "media_count": 1,
        "image_count": 1,
    }


def test_classify_tts_payload_is_tts() -> None:
    assert classify_tts_payload({"contents": [{"parts": [{"text": "hi"}]}]}).model_dump() == {
        "operation_type": "tts",
        "payload_kind": "tts",
        "media_count": 0,
        "image_count": 0,
    }


def test_timing_to_dict_sanitizes_raw_dict() -> None:
    raw_timing = {
        "operation_type": "chat",
        "payload_kind": "text",
        "request_bytes": 42,
        "response_bytes": 17,
        "media_count": 0,
        "image_count": 0,
        "provider_total_ms": 12,
        "request_prepare_ms": 1,
        "response_headers_ms": 2,
        "response_body_ms": 3,
        "response_parse_ms": 4,
        "timeout_kind": None,
        "timeout_stage": None,
        "headers": {"authorization": "Bearer secret"},
        "api_key": "secret",
        "proxy_url": "http://proxy.test",
        "prompt": "raw prompt",
        "response": "raw response",
        "data": "base64-media",
        "cookie": "session=secret",
    }

    assert timing_to_dict(raw_timing) == {
        "operation_type": "chat",
        "payload_kind": "text",
        "request_bytes": 42,
        "response_bytes": 17,
        "media_count": 0,
        "image_count": 0,
        "provider_total_ms": 12,
        "request_prepare_ms": 1,
        "response_headers_ms": 2,
        "response_body_ms": 3,
        "response_parse_ms": 4,
        "timeout_kind": None,
        "timeout_stage": None,
    }


def test_timing_to_dict_ignores_malformed_raw_dict_fields() -> None:
    assert timing_to_dict(
        {
            "operation_type": "chat",
            "payload_kind": "secret",
            "request_bytes": 42,
            "response_bytes": -1,
            "provider_total_ms": "slow",
            "timeout_stage": "internal",
            "headers": {"authorization": "Bearer secret"},
        }
    ) == {
        "operation_type": "chat",
        "request_bytes": 42,
    }


def test_provider_timing_columns_returns_stable_safe_columns() -> None:
    assert provider_timing_columns(
        {
            "operation_type": "chat",
            "payload_kind": "media",
            "request_bytes": 2048,
            "response_bytes": -1,
            "media_count": 2,
            "image_count": True,
            "provider_total_ms": 1200,
            "request_prepare_ms": 2,
            "timeout_kind": "internal_timeout_name",
            "timeout_stage": "response_headers",
            "headers": {"authorization": "Bearer secret"},
        }
    ) == {
        "operation_type": "chat",
        "payload_kind": "media",
        "request_bytes": 2048,
        "response_bytes": None,
        "media_count": 2,
        "image_count": None,
        "provider_total_ms": 1200,
        "request_prepare_ms": 2,
        "response_headers_ms": None,
        "response_body_ms": None,
        "response_parse_ms": None,
        "timeout_kind": "timeout",
        "timeout_stage": "response_headers",
    }


@pytest.mark.asyncio
async def test_send_timed_json_measures_bytes_and_response_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"messages": [{"role": "user", "content": "hi"}]}
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        result = await send_timed_json(
            client=client,
            method="POST",
            url="https://example.test/v1",
            headers={"Content-Type": "application/json"},
            payload={"messages": [{"role": "user", "content": "hi"}]},
            payload_summary=classify_chat_payload({"messages": [{"role": "user", "content": "hi"}]}),
            timeout_seconds=5,
        )

    assert result.payload == {"ok": True}
    assert result.timing.request_bytes > 0
    assert result.timing.response_bytes == len(result.body_bytes)
    assert result.timing.response_headers_ms is not None
    assert result.timing.response_body_ms is not None
    assert result.timing.response_parse_ms is not None


@pytest.mark.asyncio
async def test_send_timed_json_preserves_http_error_with_non_json_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"upstream unavailable")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        result = await send_timed_json(
            client=client,
            method="POST",
            url="https://example.test/v1",
            headers={"Content-Type": "application/json"},
            payload={"messages": [{"role": "user", "content": "hi"}]},
            payload_summary=classify_chat_payload({"messages": [{"role": "user", "content": "hi"}]}),
            timeout_seconds=5,
        )

    assert result.response.status_code == 500
    assert result.payload is None
    assert result.body_bytes == b"upstream unavailable"
    assert result.timing.response_headers_ms is not None
    assert result.timing.response_body_ms is not None
    assert result.timing.response_parse_ms is not None


@pytest.mark.asyncio
async def test_send_timed_json_preserves_http_error_with_invalid_utf8_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"\xff")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        result = await send_timed_json(
            client=client,
            method="POST",
            url="https://example.test/v1",
            headers={"Content-Type": "application/json"},
            payload={"messages": [{"role": "user", "content": "hi"}]},
            payload_summary=classify_chat_payload({"messages": [{"role": "user", "content": "hi"}]}),
            timeout_seconds=5,
        )

    assert result.response.status_code == 500
    assert result.payload is None
    assert result.body_bytes == b"\xff"
    assert result.timing.response_headers_ms is not None
    assert result.timing.response_body_ms is not None
    assert result.timing.response_parse_ms is not None


@pytest.mark.asyncio
async def test_send_timed_json_preserves_non_object_json_without_wrapping() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "object"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        result = await send_timed_json(
            client=client,
            method="POST",
            url="https://example.test/v1",
            headers={"Content-Type": "application/json"},
            payload={"messages": [{"role": "user", "content": "hi"}]},
            payload_summary=classify_chat_payload({"messages": [{"role": "user", "content": "hi"}]}),
            timeout_seconds=5,
        )

    assert result.response.status_code == 200
    assert result.payload is None
    assert result.timing.response_parse_ms is not None


@pytest.mark.asyncio
async def test_send_timed_json_attaches_timeout_stage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("raw details", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        with pytest.raises(httpx.ReadTimeout) as exc_info:
            await send_timed_json(
                client=client,
                method="POST",
                url="https://example.test/v1",
                headers={"Content-Type": "application/json"},
                payload={"messages": [{"role": "user", "content": "hi"}]},
                payload_summary=classify_chat_payload({"messages": [{"role": "user", "content": "hi"}]}),
                timeout_seconds=5,
            )

    timing = getattr(exc_info.value, "provider_timing", None)
    assert timing is not None
    assert timing.timeout_kind == "read_timeout"
    assert timing.timeout_stage == "response_headers"


@pytest.mark.asyncio
async def test_send_timed_json_closes_response_when_body_read_times_out() -> None:
    body_stream = _FailingBodyStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body_stream, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        with pytest.raises(httpx.ReadTimeout) as exc_info:
            await send_timed_json(
                client=client,
                method="POST",
                url="https://example.test/v1",
                headers={"Content-Type": "application/json"},
                payload={"messages": [{"role": "user", "content": "hi"}]},
                payload_summary=classify_chat_payload({"messages": [{"role": "user", "content": "hi"}]}),
                timeout_seconds=5,
            )

    timing = getattr(exc_info.value, "provider_timing", None)
    assert body_stream.closed is True
    assert timing is not None
    assert timing.timeout_kind == "read_timeout"
    assert timing.timeout_stage == "response_body"


@pytest.mark.asyncio
async def test_send_timed_json_attaches_timing_to_proxy_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ProxyError("raw proxy details", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        with pytest.raises(httpx.ProxyError) as exc_info:
            await send_timed_json(
                client=client,
                method="POST",
                url="https://example.test/v1",
                headers={"Content-Type": "application/json"},
                payload={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe"},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
                            ],
                        }
                    ]
                },
                payload_summary=classify_chat_payload(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "describe"},
                                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
                                ],
                            }
                        ]
                    }
                ),
                timeout_seconds=5,
            )

    timing = getattr(exc_info.value, "provider_timing", None)
    assert timing is not None
    assert timing.operation_type == "chat"
    assert timing.payload_kind == "media"
    assert timing.request_bytes > 0
    assert timing.media_count == 1
    assert timing.image_count == 1
    assert timing.provider_total_ms >= 0
    assert timing.request_prepare_ms >= 0
    assert timing.timeout_kind is None
    assert timing.timeout_stage is None
    assert "raw proxy details" not in str(timing_to_dict(timing))
