from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gemini_gateway.contracts import GatewayEmbeddingRequest
from gemini_gateway.embedding_client import GeminiEmbeddingClient
from gemini_gateway.errors import GatewayError

_PROXY_URL = "http://user:pass@127.0.0.1:8000"


def _embedding_values(dimensions: int = 1536) -> list[float]:
    return [0.1] * dimensions


@pytest.mark.asyncio
async def test_embedding_client_requires_proxy_url_without_provider_call() -> None:
    called = False

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"embedding": {"values": [0.1]}})

    client = GeminiEmbeddingClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(
            request=GatewayEmbeddingRequest(
                request_id="req-emb-no-proxy",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "text", "text": "кот"}],
            ),
            api_key="AIza-test-key",
            proxy_url=None,
        )

    assert called is False
    assert exc_info.value.reason == "no_route"
    assert exc_info.value.retryable is True
    assert exc_info.value.provider_called is False


@pytest.mark.asyncio
async def test_embedding_client_posts_native_payload_through_proxy_and_parses_response() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["api_key"] = request.headers.get("x-goog-api-key")
        seen["json"] = json.loads(request.content)
        seen["timeout"] = request.extensions["timeout"]
        return httpx.Response(
            200,
            json={
                "embedding": {"values": _embedding_values()},
                "usageMetadata": {"totalTokenCount": 3},
            },
        )

    client = GeminiEmbeddingClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    response = await client.embed(
        request=GatewayEmbeddingRequest(
            request_id="req-emb",
            source_service="media_memory",
            model="google/gemini-embedding-2",
            input=[
                {"type": "text", "text": "кот на диване"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,ZmFrZS1qcGc="},
                },
            ],
            dimensions=1536,
            timeout_seconds=9,
        ),
        api_key="AIza-test-key",
        proxy_url=_PROXY_URL,
    )

    assert "models/gemini-embedding-2:embedContent" in seen["url"]
    assert seen["api_key"] == "AIza-test-key"
    assert seen["json"]["content"]["parts"] == [
        {"text": "кот на диване"},
        {"inlineData": {"mimeType": "image/jpeg", "data": "ZmFrZS1qcGc="}},
    ]
    assert seen["json"]["outputDimensionality"] == 1536
    assert "proxy_url" not in seen["json"]
    assert seen["timeout"]["read"] == 9
    assert len(response.embedding) == 1536
    assert response.dimensions == 1536
    assert response.usage["total_tokens"] == 3
    assert response.raw_response["usageMetadata"]["totalTokenCount"] == 3


@pytest.mark.asyncio
async def test_embedding_client_ignores_bool_usage_metadata() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "embedding": {"values": _embedding_values()},
                "usageMetadata": {"totalTokenCount": True},
            },
        )

    client = GeminiEmbeddingClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    response = await client.embed(
        request=GatewayEmbeddingRequest(
            request_id="req-emb-bool-usage",
            source_service="media_memory",
            model="google/gemini-embedding-2",
            input=[{"type": "text", "text": "кот"}],
            dimensions=1536,
        ),
        api_key="AIza-test-key",
        proxy_url=_PROXY_URL,
    )

    assert response.usage == {"total_tokens": None}


@pytest.mark.asyncio
async def test_embedding_client_rejects_vector_dimension_mismatch() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": {"values": [0.1, 0.2, 0.3]}})

    client = GeminiEmbeddingClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(
            request=GatewayEmbeddingRequest(
                request_id="req-emb-dim-mismatch",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "text", "text": "кот"}],
                dimensions=1536,
            ),
            api_key="AIza-test-key",
            proxy_url=_PROXY_URL,
        )

    assert exc_info.value.reason == "invalid_response"
    assert exc_info.value.retryable is False
    assert exc_info.value.provider_called is True


@pytest.mark.asyncio
async def test_embedding_client_rejects_unsupported_image_url_without_provider_call() -> None:
    called = False

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"embedding": {"values": _embedding_values()}})

    client = GeminiEmbeddingClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(
            request=GatewayEmbeddingRequest(
                request_id="req-emb-unsupported-url",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}],
                dimensions=1536,
            ),
            api_key="AIza-test-key",
            proxy_url=_PROXY_URL,
        )

    assert called is False
    assert exc_info.value.reason == "bad_request"
    assert exc_info.value.retryable is False
    assert exc_info.value.provider_called is False


@pytest.mark.asyncio
async def test_embedding_client_rejects_malformed_success_payload() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": {"values": []}})

    client = GeminiEmbeddingClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(
            request=GatewayEmbeddingRequest(
                request_id="req-emb-bad",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "text", "text": "кот"}],
            ),
            api_key="AIza-test-key",
            proxy_url=_PROXY_URL,
        )

    assert exc_info.value.reason == "invalid_response"
    assert exc_info.value.retryable is False
    assert exc_info.value.provider_called is True


@pytest.mark.asyncio
async def test_embedding_client_records_stable_timeout_kind_without_raw_transport_text() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect timeout with api_key=AIza-secret", request=request)

    client = GeminiEmbeddingClient(base_url="https://example.test/v1beta", transport=httpx.MockTransport(handler))

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(
            request=GatewayEmbeddingRequest(
                request_id="req-emb-connect-timeout",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "text", "text": "кот"}],
            ),
            api_key="AIza-secret",
            proxy_url=_PROXY_URL,
        )

    error = exc_info.value
    assert error.reason == "network_timeout"
    assert error.provider_message_safe == "connect_timeout"
    assert "AIza-secret" not in str(error.provider_message_safe)
