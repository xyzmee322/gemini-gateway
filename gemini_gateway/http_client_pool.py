from __future__ import annotations

from typing import Any

import httpx


class GatewayHttpClientPool:
    """Переиспользует HTTP-клиенты gateway по transport/proxy ключу."""

    def __init__(
        self,
        *,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout = timeout
        self._transport = transport
        self._clients: dict[str, httpx.AsyncClient] = {}

    def get(self, *, proxy_url: str | None) -> httpx.AsyncClient:
        proxy_url = _require_proxy_url(proxy_url)
        key = "__transport__" if self._transport is not None else proxy_url
        client = self._clients.get(key)
        if client is not None:
            return client

        client_kwargs: dict[str, Any] = {
            "timeout": self._timeout,
            "trust_env": False,
        }
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        elif proxy_url is not None:
            client_kwargs["proxy"] = proxy_url

        client = httpx.AsyncClient(**client_kwargs)
        self._clients[key] = client
        return client

    async def aclose(self) -> None:
        clients = list(self._clients.values())
        self._clients.clear()
        for client in clients:
            await client.aclose()


def _require_proxy_url(proxy_url: str | None) -> str:
    """Не дает HTTP pool создать direct-клиент без proxy."""

    normalized = (proxy_url or "").strip()
    if not normalized:
        raise ValueError("proxy_url is required for Gemini gateway HTTP client")
    return normalized
