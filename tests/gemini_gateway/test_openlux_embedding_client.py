from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gemini_gateway.contracts import GatewayEmbeddingRequest
from gemini_gateway.errors import GatewayError
from gemini_gateway.openlux_embedding_client import OpenLuxEmbeddingClient


def _embedding_values(dimensions: int) -> list[float]:
    return [0.125] * dimensions


@pytest.mark.asyncio
async def test_openlux_embedding_client_uses_compatible_endpoint_for_text() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("Authorization")
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"x-api-request-id": "openlux-text-request"},
            json={
                "model": "gemini-embedding-2-preview",
                "data": [{"embedding": _embedding_values(3072)}],
                "usage": {"prompt_tokens": 7, "total_tokens": 7},
            },
        )

    client = OpenLuxEmbeddingClient(
        base_url="https://openlux.test/v1",
        transport=httpx.MockTransport(handler),
    )
    request = GatewayEmbeddingRequest(
        request_id="req-openlux-text",
        source_service="media_memory",
        model="google/gemini-embedding-2",
        input=[{"type": "text", "text": "кот на диване"}],
        dimensions=3072,
        timeout_seconds=9,
    )

    response = await client.embed(
        request=request,
        api_key="sk-openlux-secret",
        model="gemini-embedding-2-preview",
    )

    assert seen == {
        "url": "https://openlux.test/v1/embeddings",
        "authorization": "Bearer sk-openlux-secret",
        "json": {
            "model": "gemini-embedding-2-preview",
            "input": "кот на диване",
            "dimensions": 3072,
            "encoding_format": "float",
        },
    }
    assert response.request_id == request.request_id
    assert response.model == request.model
    assert response.dimensions == 3072
    assert response.usage == {"prompt_tokens": 7, "total_tokens": 7}
    assert response.provider_request_id == "openlux-text-request"


@pytest.mark.asyncio
async def test_openlux_embedding_client_uses_native_camel_case_payload_for_media() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("Authorization")
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"x-api-request-id": "openlux-image-request"},
            json={
                "embedding": {"values": _embedding_values(1536)},
                "usageMetadata": {
                    "promptTokenCount": 258,
                    "promptTokenDetails": [{"modality": "IMAGE", "tokenCount": 258}],
                },
            },
        )

    client = OpenLuxEmbeddingClient(
        base_url="https://openlux.test/v1",
        transport=httpx.MockTransport(handler),
    )
    request = GatewayEmbeddingRequest(
        request_id="req-openlux-image",
        source_service="media_memory",
        model="google/gemini-embedding-2",
        input=[
            {
                "type": "inline_data",
                "inline_data": {"mime_type": "image/jpeg", "data": "ZmFrZS1qcGc="},
            }
        ],
        dimensions=1536,
        timeout_seconds=11,
    )

    response = await client.embed(
        request=request,
        api_key="sk-openlux-secret",
        model="gemini-embedding-2-preview",
    )

    assert seen == {
        "url": "https://openlux.test/v1beta/models/gemini-embedding-2-preview:embedContent",
        "authorization": "Bearer sk-openlux-secret",
        "json": {
            "content": {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": "ZmFrZS1qcGc=",
                        }
                    }
                ]
            },
            "outputDimensionality": 1536,
        },
    }
    assert response.request_id == request.request_id
    assert response.model == request.model
    assert response.dimensions == 1536
    assert response.usage == {"prompt_tokens": 258, "total_tokens": 258}
    assert response.provider_request_id == "openlux-image-request"


@pytest.mark.asyncio
async def test_openlux_embedding_client_maps_exhausted_balance_to_retryable_quota_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": {"message": "insufficient balance"}})

    client = OpenLuxEmbeddingClient(
        base_url="https://openlux.test/v1",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(GatewayError) as exc_info:
        await client.embed(
            request=GatewayEmbeddingRequest(
                request_id="req-openlux-no-balance",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "text", "text": "кот"}],
                dimensions=768,
            ),
            api_key="sk-openlux-secret",
            model="gemini-embedding-2-preview",
        )

    assert exc_info.value.reason == "quota_exhausted"
    assert exc_info.value.retryable is True
    assert exc_info.value.provider_called is True


@pytest.mark.asyncio
@pytest.mark.parametrize("dimensions", [768, 1536])
async def test_reduced_text_dimensions_use_native_without_compatible_attempt(dimensions: int) -> None:
    calls = []
    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        payload = json.loads(request.content)
        assert request.url.path == "/v1beta/models/gemini-embedding-2-preview:embedContent"
        assert payload == {"content": {"parts": [{"text": "Поиск старых сообщений"}]}, "outputDimensionality": dimensions}
        return httpx.Response(200, json={"embedding": {"values": _embedding_values(dimensions)}, "usageMetadata": {"promptTokenCount": 8}})
    client = OpenLuxEmbeddingClient(base_url="https://openlux.test/v1", transport=httpx.MockTransport(handler))
    response = await client.embed(request=GatewayEmbeddingRequest(
        request_id="native-dimension-test", source_service="message_search", model="google/gemini-embedding-2",
        input=[{"type": "text", "text": "Поиск старых сообщений"}], dimensions=dimensions,
    ), api_key="test-secret", model="gemini-embedding-2-preview")
    assert len(calls) == 1
    assert len(response.embedding) == dimensions
    assert response.usage['prompt_tokens'] == 8
