from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from gemini_gateway.api import create_app
from gemini_gateway.contracts import GatewayEmbeddingResponse


class _SuccessfulEmbeddingService:
    def __init__(self) -> None:
        self.seen_request: Any = None

    async def embed(self, request: Any) -> GatewayEmbeddingResponse:
        self.seen_request = request
        return GatewayEmbeddingResponse(
            request_id=request.request_id,
            model=request.model,
            embedding=[0.1, 0.2, 0.3],
            dimensions=request.dimensions,
            usage={"total_tokens": 3},
            route={
                "project_label": "friend-emb",
                "route_label": "route-emb",
                "key_label": "key-emb",
                "proxy_label": "proxy-emb",
                "transport_mode": "proxy",
            },
        )


def test_embeddings_api_requires_bearer_token() -> None:
    app = create_app(auth_token="secret-token", completion_service=_SuccessfulEmbeddingService())
    client = TestClient(app)

    response = client.post(
        "/v1/embeddings",
        json={
            "request_id": "req-emb-auth",
            "source_service": "media_memory",
            "model": "google/gemini-embedding-2",
            "input": [{"type": "text", "text": "кот"}],
        },
    )

    assert response.status_code == 401
    assert response.json() == {
        "request_id": None,
        "error": "Недостаточно прав для выполнения запроса",
        "reason": "unauthorized",
        "retryable": False,
    }


def test_embeddings_api_returns_embedding_response() -> None:
    service = _SuccessfulEmbeddingService()
    app = create_app(auth_token="secret-token", completion_service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-emb",
            "source_service": "media_memory",
            "model": "google/gemini-embedding-2",
            "input": [{"type": "text", "text": "кот"}],
            "dimensions": 1536,
            "chat_id": 42,
            "telegram_message_id": 100,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert service.seen_request.request_id == "req-emb"
    assert service.seen_request.dimensions == 1536
    assert payload["request_id"] == "req-emb"
    assert payload["model"] == "google/gemini-embedding-2"
    assert payload["embedding"] == [0.1, 0.2, 0.3]
    assert payload["dimensions"] == 1536
    assert payload["usage"]["total_tokens"] == 3
    assert payload["route"]["transport_mode"] == "proxy"
    assert "raw_response" not in payload


def test_gateway_main_wires_embedding_client_into_production_service() -> None:
    source = Path("gemini_gateway/main.py").read_text(encoding="utf-8")
    service_start = source.index("service = CompletionService(")
    service_end = source.index("    app = create_app(")

    assert "from gemini_gateway.embedding_client import GeminiEmbeddingClient" in source
    assert "embedding_client = GeminiEmbeddingClient(" in source[:service_start]
    assert "base_url=_native_gemini_base_url(settings.upstream_base_url)" in source[:service_start]
    assert "client_pool=http_client_pool" in source[:service_start]
    assert "embedding_client=embedding_client" in source[service_start:service_end]
