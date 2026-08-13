from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gemini_gateway.contracts import GatewayEmbeddingInputPart, GatewayEmbeddingRequest
from gemini_gateway.errors import GatewayError
from gemini_gateway.openrouter_embedding_client import OpenRouterEmbeddingClient


def _embedding_values(dimensions: int = 1536) -> list[float]:
    return [0.1] * dimensions


@pytest.mark.asyncio
async def test_openrouter_embedding_client_posts_direct_payload_and_parses_response() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("Authorization")
        seen["json"] = json.loads(request.content)
        seen["timeout"] = request.extensions["timeout"]
        return httpx.Response(
            200,
            json={
                "id": "gen-openrouter-1",
                "model": "google/gemini-embedding-2",
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": _embedding_values()}],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
        )

    client = OpenRouterEmbeddingClient(
        base_url="https://openrouter.test/api/v1",
        transport=httpx.MockTransport(handler),
    )

    response = await client.embed(
        request=GatewayEmbeddingRequest(
            request_id="req-openrouter-emb",
            source_service="media_memory",
            model="google/gemini-embedding-2",
            input=[
                {"type": "text", "text": "  кот на диване  "},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZS1qcGc="}},
            ],
            dimensions=1536,
            timeout_seconds=9,
            chat_id=42,
        ),
        api_key="sk-or-secret",
    )

    assert seen["url"] == "https://openrouter.test/api/v1/embeddings"
    assert seen["authorization"] == "Bearer sk-or-secret"
    assert seen["json"] == {
        "model": "google/gemini-embedding-2",
        "input": [
            {
                "content": [
                    {"type": "text", "text": "  кот на диване  "},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZS1qcGc="}},
                ]
            }
        ],
        "dimensions": 1536,
        "encoding_format": "float",
        "user": "42",
    }
    assert seen["timeout"]["read"] == 9
    assert response.request_id == "req-openrouter-emb"
    assert response.generation_id == "gen-openrouter-1"
    assert response.model == "google/gemini-embedding-2"
    assert len(response.embedding) == 1536
    assert response.dimensions == 1536
    assert response.usage == {"prompt_tokens": 3, "total_tokens": 3}
    assert response.raw_response["id"] == "gen-openrouter-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mime_type", "data", "expected_content"),
    [
        (
            "image/png",
            "iVBORw==",
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw=="},
            },
        ),
        (
            "audio/wav",
            "UklGRg==",
            {
                "type": "input_audio",
                "input_audio": {
                    "data": "data:audio/wav;base64,UklGRg==",
                    "format": "wav",
                },
            },
        ),
        (
            "video/mp4",
            "AAAAIGZ0eXA=",
            {
                "type": "video_url",
                "video_url": {"url": "data:video/mp4;base64,AAAAIGZ0eXA="},
            },
        ),
        (
            "application/pdf",
            "JVBERi0=",
            {
                "type": "file",
                "file": {
                    "filename": "embedding-input.pdf",
                    "file_data": "data:application/pdf;base64,JVBERi0=",
                },
            },
        ),
    ],
)
async def test_openrouter_embedding_client_serializes_inline_multimodal_data(
    mime_type: str,
    data: str,
    expected_content: dict[str, Any],
) -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"embedding": _embedding_values()}]})

    client = OpenRouterEmbeddingClient(
        base_url="https://openrouter.test/api/v1",
        transport=httpx.MockTransport(handler),
    )

    await client.embed(
        request=GatewayEmbeddingRequest(
            request_id="req-openrouter-inline-multimodal",
            source_service="media_memory",
            model="google/gemini-embedding-2",
            input=[
                {
                    "type": "inline_data",
                    "inline_data": {"mime_type": mime_type, "data": data},
                },
            ],
        ),
        api_key="sk-or-secret",
    )

    assert seen["json"]["input"] == [{"content": [expected_content]}]


@pytest.mark.asyncio
async def test_openrouter_embedding_client_rejects_audio_data_url_image_url_without_provider_call() -> None:
    called = False

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"data": [{"embedding": _embedding_values()}]})

    client = OpenRouterEmbeddingClient(
        base_url="https://openrouter.test/api/v1",
        transport=httpx.MockTransport(handler),
    )
    request = GatewayEmbeddingRequest.model_construct(
        request_id="req-openrouter-audio-image-url",
        source_service="media_memory",
        model="google/gemini-embedding-2",
        input=[
            GatewayEmbeddingInputPart.model_construct(
                type="image_url",
                image_url={"url": "data:audio/wav;base64,UklGRg=="},
            )
        ],
        dimensions=1536,
        timeout_seconds=30,
        chat_id=None,
    )

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(request=request, api_key="sk-or-secret")

    assert called is False
    assert exc_info.value.reason == "bad_request"
    assert exc_info.value.retryable is False
    assert exc_info.value.provider_called is False
    assert "UklGRg" not in str(exc_info.value.provider_message_safe)


@pytest.mark.asyncio
async def test_openrouter_embedding_client_rejects_dimension_mismatch() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}], "model": "google/gemini-embedding-2"})

    client = OpenRouterEmbeddingClient(base_url="https://openrouter.test/api/v1", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(
            request=GatewayEmbeddingRequest(
                request_id="req-openrouter-dim",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "text", "text": "кот"}],
                dimensions=1536,
            ),
            api_key="sk-or-secret",
        )

    assert exc_info.value.reason == "invalid_response"
    assert exc_info.value.retryable is False
    assert exc_info.value.provider_called is True


@pytest.mark.asyncio
async def test_openrouter_embedding_client_rejects_non_numeric_embedding_values() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [0.1, True]}], "model": "google/gemini-embedding-2"})

    client = OpenRouterEmbeddingClient(base_url="https://openrouter.test/api/v1", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(
            request=GatewayEmbeddingRequest(
                request_id="req-openrouter-bool",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "text", "text": "кот"}],
            ),
            api_key="sk-or-secret",
        )

    assert exc_info.value.reason == "invalid_response"
    assert exc_info.value.retryable is False
    assert exc_info.value.provider_called is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"model": "google/gemini-embedding-2"},
        {"data": [{"embedding": []}], "model": "google/gemini-embedding-2"},
        ["not", "an", "object"],
    ],
)
async def test_openrouter_embedding_client_rejects_malformed_embedding_payloads(payload: Any) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = OpenRouterEmbeddingClient(base_url="https://openrouter.test/api/v1", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(
            request=GatewayEmbeddingRequest(
                request_id="req-openrouter-malformed",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "text", "text": "кот"}],
            ),
            api_key="sk-or-secret",
        )

    assert exc_info.value.reason == "invalid_response"
    assert exc_info.value.retryable is False
    assert exc_info.value.provider_called is True


@pytest.mark.asyncio
@pytest.mark.parametrize("embedding_literal", ["NaN", "Infinity", "-Infinity"])
async def test_openrouter_embedding_client_rejects_non_finite_embedding_values(embedding_literal: str) -> None:
    values = ",".join(["0.1"] * 767)

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                '{"data":[{"embedding":['
                f"{embedding_literal},{values}"
                ']}],"model":"google/gemini-embedding-2"}'
            ),
            headers={"Content-Type": "application/json"},
        )

    client = OpenRouterEmbeddingClient(base_url="https://openrouter.test/api/v1", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(
            request=GatewayEmbeddingRequest(
                request_id="req-openrouter-non-finite",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "text", "text": "кот"}],
                dimensions=768,
            ),
            api_key="sk-or-secret",
        )

    assert exc_info.value.reason == "invalid_response"
    assert exc_info.value.retryable is False
    assert exc_info.value.provider_called is True


@pytest.mark.asyncio
async def test_openrouter_embedding_client_maps_payment_required_to_quota_exhausted() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": {"message": "insufficient credits for sk-or-secret"}})

    client = OpenRouterEmbeddingClient(base_url="https://openrouter.test/api/v1", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(
            request=GatewayEmbeddingRequest(
                request_id="req-openrouter-402",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "text", "text": "кот"}],
            ),
            api_key="sk-or-secret",
        )

    error = exc_info.value
    assert error.reason == "quota_exhausted"
    assert error.retryable is False
    assert error.provider_status_code == 402
    assert "sk-or-secret" not in str(error.provider_message_safe)


@pytest.mark.asyncio
async def test_openrouter_embedding_client_maps_error_wrapped_in_success_status() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": {"code": 429, "message": "rate limited for sk-or-secret"}},
        )

    client = OpenRouterEmbeddingClient(base_url="https://openrouter.test/api/v1", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(
            request=GatewayEmbeddingRequest(
                request_id="req-openrouter-wrapped-429",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "text", "text": "кот"}],
            ),
            api_key="sk-or-secret",
        )

    error = exc_info.value
    assert error.reason == "rate_limited"
    assert error.retryable is True
    assert error.provider_status_code == 429
    assert error.provider_called is True
    assert "sk-or-secret" not in str(error.provider_message_safe)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "message", "expected_reason", "expected_retryable"),
    [
        (401, "invalid credentials", "auth_failed", False),
        (403, "input was flagged by moderation", "content_filtered", False),
        (403, "prohibited content", "content_filtered", False),
        (403, "blocked by policy", "content_filtered", False),
        (403, "forbidden", "auth_failed", False),
        (408, "request timed out", "network_timeout", True),
        (409, "conflict", "invalid_response", True),
        (429, "rate limited", "rate_limited", True),
        (500, "upstream unavailable", "provider_unavailable", True),
        (524, "edge network timeout", "network_timeout", True),
        (529, "provider overloaded", "provider_unavailable", True),
    ],
)
async def test_openrouter_embedding_client_maps_provider_http_statuses(
    status_code: int,
    message: str,
    expected_reason: str,
    expected_retryable: bool,
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": {"message": message}}, headers={"Retry-After": "7"})

    client = OpenRouterEmbeddingClient(base_url="https://openrouter.test/api/v1", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(
            request=GatewayEmbeddingRequest(
                request_id=f"req-openrouter-{status_code}",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "text", "text": "кот"}],
            ),
            api_key="sk-or-secret",
        )

    assert exc_info.value.reason == expected_reason
    assert exc_info.value.retryable is expected_retryable
    assert exc_info.value.provider_status_code == status_code
    assert exc_info.value.retry_after_seconds == 7


@pytest.mark.asyncio
async def test_openrouter_embedding_client_records_stable_timeout_kind() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timeout with sk-or-secret", request=request)

    client = OpenRouterEmbeddingClient(base_url="https://openrouter.test/api/v1", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(
            request=GatewayEmbeddingRequest(
                request_id="req-openrouter-timeout",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "text", "text": "кот"}],
            ),
            api_key="sk-or-secret",
        )

    assert exc_info.value.reason == "network_timeout"
    assert exc_info.value.provider_message_safe == "read_timeout"
