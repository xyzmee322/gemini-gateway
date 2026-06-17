# OpenRouter Embedding Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a direct OpenRouter fallback for `google/gemini-embedding-2` embeddings that is used only when the Gemini route pool cannot provide any usable key/proxy route for that model.

**Architecture:** Keep the existing Gemini route pool proxy-only. Add a separate OpenRouter embeddings client and wire it into `CompletionService.embed()` as a fallback after route acquisition fails with `no_route`, `cooldown_active`, or `quota_exhausted`. Do not fallback after a leased Gemini route fails, because one provider/proxy/key failure does not prove that all Gemini keys are unavailable.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, httpx async client, pytest, pytest-asyncio, Docker Compose.

---

## File Structure

- Modify `gemini_gateway/config.py`: add OpenRouter settings, validation, and placeholder protection.
- Modify `gemini_gateway/contracts.py`: allow response route metadata to represent direct fallback transport while keeping seeded Gemini bindings proxy-only.
- Create `gemini_gateway/openrouter_embedding_client.py`: direct no-proxy OpenRouter embeddings HTTP client and response parser.
- Modify `gemini_gateway/service.py`: fallback orchestration, safe route metadata, and wide-event logging.
- Modify `gemini_gateway/main.py`: construct and inject OpenRouter fallback dependencies.
- Modify `gemini_gateway/errors.py`: make quota public text provider-neutral.
- Modify `docker-compose.yml`: expose OpenRouter fallback env vars to the gateway container.
- Create `docker-compose.dev.yml`: local override carrying the same fallback env vars for development compose runs.
- Modify `README.md`: document fallback behavior, env vars, and trigger conditions.
- Create `tests/gemini_gateway/test_openrouter_embedding_client.py`: client contract tests.
- Modify `tests/gemini_gateway/test_settings.py`: settings and validation tests.
- Modify `tests/gemini_gateway/test_contracts.py`: direct route metadata response test.
- Modify `tests/gemini_gateway/test_routing.py`: service fallback orchestration tests.
- Modify `tests/gemini_gateway/test_gateway_api_embeddings.py`: API response shape test for direct fallback metadata.
- Modify `tests/gemini_gateway/test_compose_config.py`: compose env wiring tests.

## Task 1: Settings And Compose Wiring

**Files:**
- Modify: `gemini_gateway/config.py`
- Modify: `tests/gemini_gateway/test_settings.py`
- Modify: `docker-compose.yml`
- Create: `docker-compose.dev.yml`
- Modify: `tests/gemini_gateway/test_compose_config.py`

- [ ] **Step 1: Write failing settings tests**

Add these tests to `tests/gemini_gateway/test_settings.py`:

```python
def test_gateway_settings_load_openrouter_embedding_fallback_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_API_KEY", "sk-or-openrouter-secret-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/")

    settings = GeminiGatewaySettings()

    assert settings.openrouter_embeddings_fallback_enabled is True
    assert settings.openrouter_api_key is not None
    assert settings.openrouter_api_key.get_secret_value() == "sk-or-openrouter-secret-value"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.openrouter_embeddings_fallback_model == "google/gemini-embedding-2"


def test_gateway_settings_reject_enabled_openrouter_fallback_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED", "true")

    with pytest.raises(ValidationError):
        GeminiGatewaySettings()


def test_gateway_settings_allows_blank_openrouter_key_when_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_API_KEY", "")

    settings = GeminiGatewaySettings()

    assert settings.openrouter_embeddings_fallback_enabled is False
    assert settings.openrouter_api_key is None


def test_gateway_settings_reject_openrouter_placeholder_outside_dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_GATEWAY_ENVIRONMENT", "production")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_POSTGRES_SYNC_DSN", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("GEMINI_GATEWAY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("GEMINI_GATEWAY_HMAC_KEY", "h" * 32)
    monkeypatch.setenv("GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN", "secret-token-value")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GEMINI_GATEWAY_OPENROUTER_API_KEY", "replace-with-openrouter-key")

    with pytest.raises(ValidationError):
        GeminiGatewaySettings()
```

- [ ] **Step 2: Run settings tests and verify failure**

Run:

```bash
uv run pytest tests/gemini_gateway/test_settings.py -q
```

Expected: FAIL because `GeminiGatewaySettings` does not yet expose OpenRouter fallback fields.

- [ ] **Step 3: Implement settings**

In `gemini_gateway/config.py`, change:

```python
from typing import Literal
```

to:

```python
from typing import Any, Literal
```

In `gemini_gateway/config.py`, add these fields to `GeminiGatewaySettings`:

```python
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_embeddings_fallback_enabled: bool = False
    openrouter_embeddings_fallback_model: str = "google/gemini-embedding-2"
```

Add validators:

```python
    @field_validator("openrouter_api_key", mode="before")
    @classmethod
    def normalize_optional_openrouter_key(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("openrouter_api_key")
    @classmethod
    def validate_optional_openrouter_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        raw_value = value.get_secret_value().strip()
        if len(raw_value) < 16:
            raise ValueError("openrouter_api_key must be at least 16 characters")
        return SecretStr(raw_value)

    @field_validator("openrouter_base_url")
    @classmethod
    def normalize_openrouter_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.startswith(("https://", "http://")):
            raise ValueError("openrouter_base_url must be an absolute URL")
        return normalized

    @field_validator("openrouter_embeddings_fallback_model")
    @classmethod
    def validate_openrouter_fallback_model(cls, value: str) -> str:
        normalized = value.strip()
        if normalized != "google/gemini-embedding-2":
            raise ValueError("openrouter fallback is restricted to google/gemini-embedding-2")
        return normalized
```

Update `reject_placeholders_outside_dev()`:

```python
    @model_validator(mode="after")
    def reject_placeholders_outside_dev(self) -> "GeminiGatewaySettings":
        if self.openrouter_embeddings_fallback_enabled and self.openrouter_api_key is None:
            raise ValueError("openrouter fallback requires openrouter_api_key")

        secrets = [self.hmac_key, self.internal_auth_token, self.encryption_key]
        if self.openrouter_api_key is not None:
            secrets.append(self.openrouter_api_key)

        if self.environment == "development":
            return self

        for secret in secrets:
            if _PLACEHOLDER_SECRET_PATTERN.search(secret.get_secret_value()):
                raise ValueError("gateway secrets must not use placeholders outside development")
        return self
```

- [ ] **Step 4: Add compose env tests**

Add helpers and tests to `tests/gemini_gateway/test_compose_config.py`:

```python
def _service_environment(compose_config: dict[str, Any], service_name: str) -> dict[str, Any]:
    environment = compose_config["services"][service_name].get("environment", {})
    assert isinstance(environment, dict)
    return environment


def test_gateway_compose_wires_openrouter_fallback_env() -> None:
    compose_config = _load_compose_config()
    environment = _service_environment(compose_config, "gateway")

    assert environment["GEMINI_GATEWAY_OPENROUTER_API_KEY"] == "${GEMINI_GATEWAY_OPENROUTER_API_KEY:-}"
    assert (
        environment["GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED"]
        == "${GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED:-false}"
    )
    assert (
        environment["GEMINI_GATEWAY_OPENROUTER_BASE_URL"]
        == "${GEMINI_GATEWAY_OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
    )


def test_dev_compose_wires_openrouter_fallback_env() -> None:
    with (_PROJECT_ROOT / "docker-compose.dev.yml").open(encoding="utf-8") as compose_file:
        compose_config = yaml.safe_load(compose_file)

    environment = _service_environment(compose_config, "gateway")

    assert environment["GEMINI_GATEWAY_OPENROUTER_API_KEY"] == "${GEMINI_GATEWAY_OPENROUTER_API_KEY:-}"
    assert (
        environment["GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED"]
        == "${GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED:-false}"
    )
    assert (
        environment["GEMINI_GATEWAY_OPENROUTER_BASE_URL"]
        == "${GEMINI_GATEWAY_OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
    )
```

- [ ] **Step 5: Update compose files**

In `docker-compose.yml`, add these keys under `services.gateway.environment`:

```yaml
      GEMINI_GATEWAY_OPENROUTER_API_KEY: ${GEMINI_GATEWAY_OPENROUTER_API_KEY:-}
      GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED: ${GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED:-false}
      GEMINI_GATEWAY_OPENROUTER_BASE_URL: ${GEMINI_GATEWAY_OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}
```

Create `docker-compose.dev.yml`:

```yaml
services:
  gateway:
    environment:
      GEMINI_GATEWAY_OPENROUTER_API_KEY: ${GEMINI_GATEWAY_OPENROUTER_API_KEY:-}
      GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED: ${GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED:-false}
      GEMINI_GATEWAY_OPENROUTER_BASE_URL: ${GEMINI_GATEWAY_OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}
```

- [ ] **Step 6: Run task tests**

Run:

```bash
uv run pytest tests/gemini_gateway/test_settings.py tests/gemini_gateway/test_compose_config.py -q
```

Expected: PASS.

- [ ] **Step 7: Hold commit until explicit approval**

Run:

```bash
git --no-pager diff -- gemini_gateway/config.py tests/gemini_gateway/test_settings.py docker-compose.yml docker-compose.dev.yml tests/gemini_gateway/test_compose_config.py
```

Expected: diff contains only settings and compose wiring. Do not run `git commit` until the user explicitly asks for a commit.

## Task 2: Direct Route Metadata Contract

**Files:**
- Modify: `gemini_gateway/contracts.py`
- Modify: `tests/gemini_gateway/test_contracts.py`
- Modify: `tests/gemini_gateway/test_gateway_api_embeddings.py`

- [ ] **Step 1: Write failing contract test**

Add to `tests/gemini_gateway/test_contracts.py`:

```python
from gemini_gateway.contracts import GatewayEmbeddingResponse


def test_embedding_response_allows_direct_fallback_route_metadata() -> None:
    response = GatewayEmbeddingResponse(
        request_id="req-direct-route",
        model="google/gemini-embedding-2",
        embedding=[0.1, 0.2, 0.3],
        dimensions=3,
        route={
            "project_label": "openrouter-fallback",
            "route_label": "openrouter-embedding-fallback",
            "key_label": "openrouter-api-key",
            "proxy_label": None,
            "transport_mode": "direct",
        },
    )

    assert response.route["transport_mode"] == "direct"
    assert response.route["proxy_label"] is None
```

Update `_SuccessfulEmbeddingService` in `tests/gemini_gateway/test_gateway_api_embeddings.py` by adding a direct fallback service:

```python
class _DirectFallbackEmbeddingService:
    async def embed(self, request: Any) -> GatewayEmbeddingResponse:
        return GatewayEmbeddingResponse(
            request_id=request.request_id,
            model=request.model,
            embedding=[0.1, 0.2, 0.3],
            dimensions=request.dimensions,
            usage={"total_tokens": 3},
            route={
                "project_label": "openrouter-fallback",
                "route_label": "openrouter-embedding-fallback",
                "key_label": "openrouter-api-key",
                "proxy_label": None,
                "transport_mode": "direct",
            },
        )
```

Add API test:

```python
def test_embeddings_api_returns_direct_fallback_route_metadata() -> None:
    app = create_app(auth_token="secret-token", completion_service=_DirectFallbackEmbeddingService())
    client = TestClient(app)

    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "request_id": "req-emb-direct",
            "source_service": "media_memory",
            "model": "google/gemini-embedding-2",
            "input": [{"type": "text", "text": "кот"}],
            "dimensions": 1536,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["route"]["transport_mode"] == "direct"
    assert payload["route"]["project_label"] == "openrouter-fallback"
    assert payload["route"]["route_label"] == "openrouter-embedding-fallback"
    assert payload["route"]["key_label"] == "openrouter-api-key"
    assert "proxy_label" not in payload["route"]
```

- [ ] **Step 2: Run contract tests and verify failure**

Run:

```bash
uv run pytest tests/gemini_gateway/test_contracts.py tests/gemini_gateway/test_gateway_api_embeddings.py -q
```

Expected: FAIL because `TransportMode` only allows `proxy` and `GatewayRouteMetadata.proxy_label` is required.

- [ ] **Step 3: Implement metadata contract change**

In `gemini_gateway/contracts.py`, change:

```python
TransportMode = Literal["proxy"]
```

to:

```python
TransportMode = Literal["proxy", "direct"]
```

Change `GatewayRouteMetadata`:

```python
class GatewayRouteMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    route_label: str
    project_label: str
    key_label: str
    proxy_label: str | None = None
    transport_mode: TransportMode = "proxy"
```

Keep `RouteCandidate`, `RouteLease`, `SeedBinding`, and DB constraints proxy-only. The direct transport mode is allowed only in response metadata and error metadata, not in seeded Gemini routes.

- [ ] **Step 4: Run task tests**

Run:

```bash
uv run pytest tests/gemini_gateway/test_contracts.py tests/gemini_gateway/test_gateway_api_embeddings.py tests/gemini_gateway/test_routing.py -q
```

Expected: PASS, including the existing test that rejects direct route candidates for Gemini routing.

- [ ] **Step 5: Hold commit until explicit approval**

Run:

```bash
git --no-pager diff -- gemini_gateway/contracts.py tests/gemini_gateway/test_contracts.py tests/gemini_gateway/test_gateway_api_embeddings.py
```

Expected: diff only broadens response metadata and adds tests. Do not run `git commit` until the user explicitly asks for a commit.

## Task 3: OpenRouter Embedding Client

**Files:**
- Create: `gemini_gateway/openrouter_embedding_client.py`
- Create: `tests/gemini_gateway/test_openrouter_embedding_client.py`
- Modify: `gemini_gateway/errors.py`
- Modify: `tests/gemini_gateway/test_gateway_api.py`

- [ ] **Step 1: Write failing client tests**

Create `tests/gemini_gateway/test_openrouter_embedding_client.py`:

```python
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gemini_gateway.contracts import GatewayEmbeddingRequest
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
                {"type": "text", "text": "кот на диване"},
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
                    {"type": "text", "text": "кот на диване"},
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
async def test_openrouter_embedding_client_rejects_dimension_mismatch() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
                "model": "google/gemini-embedding-2",
            },
        )

    client = OpenRouterEmbeddingClient(
        base_url="https://openrouter.test/api/v1",
        transport=httpx.MockTransport(handler),
    )

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
async def test_openrouter_embedding_client_maps_payment_required_to_quota_exhausted() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": {"message": "insufficient credits for sk-or-secret"}})

    client = OpenRouterEmbeddingClient(
        base_url="https://openrouter.test/api/v1",
        transport=httpx.MockTransport(handler),
    )

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
async def test_openrouter_embedding_client_records_stable_timeout_kind() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timeout with sk-or-secret", request=request)

    client = OpenRouterEmbeddingClient(
        base_url="https://openrouter.test/api/v1",
        transport=httpx.MockTransport(handler),
    )

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
```

- [ ] **Step 2: Run client tests and verify failure**

Run:

```bash
uv run pytest tests/gemini_gateway/test_openrouter_embedding_client.py -q
```

Expected: FAIL because `gemini_gateway.openrouter_embedding_client` does not exist.

- [ ] **Step 3: Implement OpenRouter client**

Create `gemini_gateway/openrouter_embedding_client.py`:

```python
from __future__ import annotations

from typing import Any

import httpx

from gemini_gateway.contracts import (
    GatewayEmbeddingInputPart,
    GatewayEmbeddingRequest,
    GatewayEmbeddingResponse,
    GatewayErrorReason,
)
from gemini_gateway.errors import GatewayError
from gemini_gateway.gemini_client import _timeout_error_kind
from gemini_gateway.provider_http_errors import extract_provider_message, parse_retry_after
from gemini_gateway.value_extractors import first_int_value, first_string_value

_RETRYABLE_OPENROUTER_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class OpenRouterEmbeddingClient:
    """HTTP-клиент OpenRouter embeddings без proxy."""

    def __init__(
        self,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport

    async def embed(
        self,
        *,
        request: GatewayEmbeddingRequest,
        api_key: str,
    ) -> GatewayEmbeddingResponse:
        payload = _openrouter_embedding_payload(request)
        try:
            client_kwargs: dict[str, Any] = {"timeout": self._timeout, "trust_env": False}
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=request.timeout_seconds,
                )
        except httpx.TimeoutException as exc:
            raise GatewayError(
                reason="network_timeout",
                retryable=True,
                provider_message_safe=_timeout_error_kind(exc),
                request_id=request.request_id,
            ) from exc
        except httpx.HTTPError as exc:
            raise GatewayError(
                reason="provider_unavailable",
                retryable=True,
                provider_message_safe=str(exc),
                request_id=request.request_id,
            ) from exc

        if response.status_code >= 400:
            raise _gateway_error_from_openrouter_response(response=response, request_id=request.request_id)

        try:
            raw_response = response.json()
        except ValueError as exc:
            raise GatewayError(
                reason="invalid_response",
                retryable=False,
                provider_message_safe=str(exc),
                request_id=request.request_id,
                provider_called=True,
            ) from exc
        if not isinstance(raw_response, dict):
            raise GatewayError(
                reason="invalid_response",
                retryable=False,
                provider_message_safe="OpenRouter embedding response payload must be a JSON object",
                request_id=request.request_id,
                provider_called=True,
            )
        return _to_gateway_embedding_response(request=request, raw_response=raw_response)


def _openrouter_embedding_payload(request: GatewayEmbeddingRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model,
        "input": [{"content": [_openrouter_part(part) for part in request.input]}],
        "dimensions": request.dimensions,
        "encoding_format": "float",
    }
    if request.chat_id is not None:
        payload["user"] = str(request.chat_id)
    return payload


def _openrouter_part(part: GatewayEmbeddingInputPart) -> dict[str, Any]:
    if part.type == "text":
        return {"type": "text", "text": str(part.text or "").strip()}
    image_url = part.image_url or {}
    return {"type": "image_url", "image_url": {"url": str(image_url.get("url") or "").strip()}}


def _gateway_error_from_openrouter_response(*, response: httpx.Response, request_id: str) -> GatewayError:
    provider_message = extract_provider_message(response)
    return GatewayError(
        reason=_openrouter_error_reason(status_code=response.status_code, provider_message=provider_message),
        retryable=response.status_code in _RETRYABLE_OPENROUTER_STATUS_CODES,
        provider_status_code=response.status_code,
        provider_message_safe=provider_message,
        retry_after_seconds=parse_retry_after(response.headers.get("Retry-After")),
        request_id=request_id,
    )


def _openrouter_error_reason(*, status_code: int, provider_message: str | None) -> GatewayErrorReason:
    message = (provider_message or "").lower()
    if status_code == 402:
        return "quota_exhausted"
    if status_code == 429:
        return "rate_limited"
    if status_code == 403 and any(marker in message for marker in ("safety", "moderation", "flagged")):
        return "content_filtered"
    if status_code in {401, 403}:
        return "auth_failed"
    if status_code in {408, 504}:
        return "network_timeout"
    if status_code >= 500:
        return "provider_unavailable"
    return "invalid_response"


def _to_gateway_embedding_response(
    *,
    request: GatewayEmbeddingRequest,
    raw_response: dict[str, Any],
) -> GatewayEmbeddingResponse:
    values = _embedding_values(raw_response)
    if not values:
        raise GatewayError(
            reason="invalid_response",
            retryable=False,
            provider_message_safe="OpenRouter embedding response does not contain embedding values",
            request_id=request.request_id,
            provider_called=True,
        )
    if len(values) != request.dimensions:
        raise GatewayError(
            reason="invalid_response",
            retryable=False,
            provider_message_safe="OpenRouter embedding response dimensions do not match requested dimensions",
            request_id=request.request_id,
            provider_called=True,
        )
    usage = raw_response.get("usage") if isinstance(raw_response.get("usage"), dict) else {}
    return GatewayEmbeddingResponse(
        request_id=request.request_id,
        generation_id=first_string_value(raw_response, "id"),
        model=first_string_value(raw_response, "model") or request.model,
        embedding=values,
        dimensions=len(values),
        usage={
            "prompt_tokens": first_int_value(usage, "prompt_tokens"),
            "total_tokens": first_int_value(usage, "total_tokens"),
        },
        raw_response=raw_response,
        provider_specific_fields={},
    )


def _embedding_values(raw_response: dict[str, Any]) -> list[float] | None:
    data = raw_response.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None
    values = data[0].get("embedding")
    if not isinstance(values, list) or not values:
        return None
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        normalized.append(float(value))
    return normalized
```

- [ ] **Step 4: Make quota error public text provider-neutral**

In `gemini_gateway/errors.py`, change:

```python
    "quota_exhausted": "Квота Gemini временно исчерпана, попробуй позже",
```

to:

```python
    "quota_exhausted": "Квота AI-провайдера временно исчерпана, попробуй позже",
```

Update expected text in any tests that assert the old string. Search with:

```bash
rg -n "Квота Gemini временно исчерпана|quota_exhausted" tests gemini_gateway
```

Expected known update: `tests/gemini_gateway/test_gateway_api.py` if it asserts the old public message.

- [ ] **Step 5: Run task tests**

Run:

```bash
uv run pytest tests/gemini_gateway/test_openrouter_embedding_client.py tests/gemini_gateway/test_gateway_api.py -q
```

Expected: PASS.

- [ ] **Step 6: Hold commit until explicit approval**

Run:

```bash
git --no-pager diff -- gemini_gateway/openrouter_embedding_client.py tests/gemini_gateway/test_openrouter_embedding_client.py gemini_gateway/errors.py tests/gemini_gateway/test_gateway_api.py
```

Expected: diff contains one new client, client tests, and provider-neutral quota message. Do not run `git commit` until the user explicitly asks for a commit.

## Task 4: Service Fallback Orchestration

**Files:**
- Modify: `gemini_gateway/service.py`
- Modify: `tests/gemini_gateway/test_routing.py`

- [ ] **Step 1: Write failing service fallback tests**

Add these tests to `tests/gemini_gateway/test_routing.py`:

```python
@pytest.mark.asyncio
async def test_embedding_service_falls_back_to_openrouter_when_all_gemini_routes_unavailable() -> None:
    class _UnavailableRepository:
        def __init__(self) -> None:
            self.failures: list[dict[str, Any]] = []

        async def acquire_route(self, request: GatewayEmbeddingRequest) -> Any:
            raise GatewayError(
                reason="quota_exhausted",
                retryable=True,
                request_id=request.request_id,
                eligible_routes_count=3,
                exhausted_routes_count=3,
            )

        async def record_failure(
            self,
            lease: Any,
            error: GatewayError,
            latency_ms: int,
            provider_called: bool,
        ) -> None:
            self.failures.append(
                {
                    "lease": lease,
                    "reason": error.reason,
                    "provider_called": provider_called,
                    "latency_ms": latency_ms,
                }
            )

    class _OpenRouterFallbackClient:
        def __init__(self) -> None:
            self.seen_api_key: str | None = None

        async def embed(self, request: GatewayEmbeddingRequest, api_key: str) -> GatewayEmbeddingResponse:
            self.seen_api_key = api_key
            return GatewayEmbeddingResponse(
                request_id=request.request_id,
                generation_id="gen-openrouter-fallback",
                model=request.model,
                embedding=[0.1] * request.dimensions,
                dimensions=request.dimensions,
                usage={"prompt_tokens": 5, "total_tokens": 5},
            )

    repository = _UnavailableRepository()
    fallback_client = _OpenRouterFallbackClient()
    service = CompletionService(
        repository=repository,
        gemini_client=None,
        embedding_client=object(),
        openrouter_embedding_client=fallback_client,
        openrouter_api_key="sk-or-fallback",
        openrouter_embeddings_fallback_enabled=True,
        service_name="gemini-gateway",
        environment="test",
    )

    response = await service.embed(
        GatewayEmbeddingRequest(
            request_id="req-fallback-quota",
            source_service="media_memory",
            model="google/gemini-embedding-2",
            input=[{"type": "text", "text": "кот"}],
            dimensions=1536,
        )
    )

    assert fallback_client.seen_api_key == "sk-or-fallback"
    assert response.generation_id == "gen-openrouter-fallback"
    assert response.route == {
        "project_label": "openrouter-fallback",
        "route_label": "openrouter-embedding-fallback",
        "key_label": "openrouter-api-key",
        "proxy_label": None,
        "transport_mode": "direct",
    }
    assert repository.failures[0]["lease"] is None
    assert repository.failures[0]["provider_called"] is False


@pytest.mark.asyncio
async def test_embedding_service_does_not_fallback_for_other_models() -> None:
    class _UnavailableRepository:
        async def acquire_route(self, request: GatewayEmbeddingRequest) -> Any:
            raise GatewayError(reason="quota_exhausted", retryable=True, request_id=request.request_id)

        async def record_failure(
            self,
            lease: Any,
            error: GatewayError,
            latency_ms: int,
            provider_called: bool,
        ) -> None:
            return None

    class _OpenRouterFallbackClient:
        async def embed(self, request: GatewayEmbeddingRequest, api_key: str) -> GatewayEmbeddingResponse:
            raise AssertionError("fallback must not be called for non-embedding-2 model")

    service = CompletionService(
        repository=_UnavailableRepository(),
        gemini_client=None,
        embedding_client=object(),
        openrouter_embedding_client=_OpenRouterFallbackClient(),
        openrouter_api_key="sk-or-fallback",
        openrouter_embeddings_fallback_enabled=True,
    )

    with pytest.raises(GatewayError) as exc_info:
        await service.embed(
            GatewayEmbeddingRequest(
                request_id="req-fallback-other-model",
                source_service="media_memory",
                model="google/gemini-embedding-001",
                input=[{"type": "text", "text": "кот"}],
                dimensions=1536,
            )
        )

    assert exc_info.value.reason == "quota_exhausted"


@pytest.mark.asyncio
async def test_embedding_service_does_not_fallback_after_leased_gemini_route_failure() -> None:
    class _FailingEmbeddingClient:
        async def embed(
            self,
            request: GatewayEmbeddingRequest,
            api_key: str,
            proxy_url: str,
        ) -> GatewayEmbeddingResponse:
            raise GatewayError(reason="quota_exhausted", retryable=True, request_id=request.request_id)

    class _OpenRouterFallbackClient:
        async def embed(self, request: GatewayEmbeddingRequest, api_key: str) -> GatewayEmbeddingResponse:
            raise AssertionError("fallback must not be called after a single leased route failure")

    repository = InMemoryRouteRepository(
        [
            _candidate(
                "leased-failure",
                api_key="AIza-gemini",
                proxy_url="http://user:pass@127.0.0.1:8000",
            ).model_copy(update={"model": "google/gemini-embedding-2"})
        ]
    )
    service = CompletionService(
        repository=repository,
        gemini_client=None,
        embedding_client=_FailingEmbeddingClient(),
        openrouter_embedding_client=_OpenRouterFallbackClient(),
        openrouter_api_key="sk-or-fallback",
        openrouter_embeddings_fallback_enabled=True,
    )

    with pytest.raises(GatewayError) as exc_info:
        await service.embed(
            GatewayEmbeddingRequest(
                request_id="req-leased-failure",
                source_service="media_memory",
                model="google/gemini-embedding-2",
                input=[{"type": "text", "text": "кот"}],
                dimensions=1536,
            )
        )

    assert exc_info.value.reason == "quota_exhausted"
    assert exc_info.value.route_label == "route-leased-failure"
```

Add imports near the top of `tests/gemini_gateway/test_routing.py`:

```python
from gemini_gateway.contracts import GatewayEmbeddingRequest, GatewayEmbeddingResponse
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/gemini_gateway/test_routing.py -q
```

Expected: FAIL because `CompletionService` does not accept OpenRouter fallback dependencies and never falls back.

- [ ] **Step 3: Implement service fallback**

In `gemini_gateway/service.py`, add constants near the top:

```python
_OPENROUTER_EMBEDDING_FALLBACK_REASONS = frozenset({"no_route", "cooldown_active", "quota_exhausted"})
_OPENROUTER_EMBEDDING_FALLBACK_ROUTE = {
    "project_label": "openrouter-fallback",
    "route_label": "openrouter-embedding-fallback",
    "key_label": "openrouter-api-key",
    "proxy_label": None,
    "transport_mode": "direct",
}
```

Extend `CompletionService.__init__()`:

```python
        openrouter_embedding_client: Any | None = None,
        openrouter_api_key: str | None = None,
        openrouter_embeddings_fallback_enabled: bool = False,
        openrouter_embeddings_fallback_model: str = "google/gemini-embedding-2",
```

Store them:

```python
        self._openrouter_embedding_client = openrouter_embedding_client
        self._openrouter_api_key = openrouter_api_key
        self._openrouter_embeddings_fallback_enabled = openrouter_embeddings_fallback_enabled
        self._openrouter_embeddings_fallback_model = openrouter_embeddings_fallback_model
```

Replace `embed()` with:

```python
    async def embed(self, request: GatewayEmbeddingRequest | dict[str, Any]) -> GatewayEmbeddingResponse:
        gateway_request = _ensure_embedding_request(request)
        if self._embedding_client is None:
            raise GatewayError(
                reason="provider_unavailable",
                retryable=True,
                request_id=gateway_request.request_id,
            )
        try:
            return await self._execute_with_route(
                request=gateway_request,
                provider_call=lambda lease: self._embedding_client.embed(
                    request=gateway_request,
                    api_key=lease.api_key,
                    proxy_url=lease.proxy_url,
                ),
            )
        except GatewayError as error:
            if not self._can_use_openrouter_embedding_fallback(request=gateway_request, error=error):
                raise
            return await self._execute_openrouter_embedding_fallback(request=gateway_request)
```

Add methods to `CompletionService`:

```python
    def _can_use_openrouter_embedding_fallback(
        self,
        *,
        request: GatewayEmbeddingRequest,
        error: GatewayError,
    ) -> bool:
        if not self._openrouter_embeddings_fallback_enabled:
            return False
        if self._openrouter_embedding_client is None or not self._openrouter_api_key:
            return False
        if request.model != self._openrouter_embeddings_fallback_model:
            return False
        if error.reason not in _OPENROUTER_EMBEDDING_FALLBACK_REASONS:
            return False
        return getattr(error, "route_label", None) is None

    async def _execute_openrouter_embedding_fallback(
        self,
        *,
        request: GatewayEmbeddingRequest,
    ) -> GatewayEmbeddingResponse:
        started_at = perf_counter()
        try:
            response = await self._openrouter_embedding_client.embed(
                request=request,
                api_key=self._openrouter_api_key,
            )
            response = _attach_static_route_metadata(response, _OPENROUTER_EMBEDDING_FALLBACK_ROUTE)
            latency_ms = _elapsed_ms(started_at)
            self._log_openrouter_fallback_success(request, response, latency_ms)
            return response
        except GatewayError as error:
            latency_ms = _elapsed_ms(started_at)
            _set_request_id(error, request.request_id)
            _attach_static_error_route_metadata(error, _OPENROUTER_EMBEDDING_FALLBACK_ROUTE)
            self._log_openrouter_fallback_failure(request, error, latency_ms)
            raise
```

Add logging methods:

```python
    def _log_openrouter_fallback_success(
        self,
        request: GatewayEmbeddingRequest,
        response: GatewayEmbeddingResponse,
        latency_ms: int,
    ) -> None:
        event = self._base_event(
            request=request,
            lease=None,
            latency_ms=latency_ms,
            status="success",
            route_context=_OPENROUTER_EMBEDDING_FALLBACK_ROUTE,
        )
        usage = response.usage or {}
        event.update(
            {
                "prompt_tokens": _safe_int(usage.get("prompt_tokens")),
                "completion_tokens": _safe_int(usage.get("completion_tokens")),
                "total_tokens": _safe_int(usage.get("total_tokens")),
                "generation_id": response.generation_id,
                "finish_reason": response.finish_reason,
                "error_type": None,
                "error_message": None,
                "error_code": None,
                "retryable": None,
                "cooldown_scope": None,
                "cooldown_level": None,
                "sleep_until": None,
                "quota_scope": None,
                "quota_reset_at": None,
                "eligible_routes_count": None,
                "exhausted_routes_count": None,
                "disabled_routes_count": None,
                "fallback_provider": "openrouter",
            }
        )
        _LOGGER.info("gemini_gateway_request", extra=event)

    def _log_openrouter_fallback_failure(
        self,
        request: GatewayEmbeddingRequest,
        error: GatewayError,
        latency_ms: int,
    ) -> None:
        event = self._base_event(
            request=request,
            lease=None,
            latency_ms=latency_ms,
            status="error",
            route_context=_OPENROUTER_EMBEDDING_FALLBACK_ROUTE,
        )
        event.update(
            {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "finish_reason": None,
                "reason": error.reason,
                "failed_stage": "openrouter_embedding_fallback",
                "error_type": error.reason,
                "error_code": getattr(error, "error_code", None),
                "error_message": error.public_message,
                "provider_reason": public_provider_reason(error.provider_message_safe),
                "provider_status_code": error.provider_status_code,
                "retryable": error.retryable,
                "retry_after_seconds": error.retry_after_seconds,
                "cooldown_scope": getattr(error, "cooldown_scope", None),
                "cooldown_level": getattr(error, "cooldown_level", None),
                "sleep_until": _serialize_datetime(getattr(error, "sleep_until", None)),
                "quota_scope": getattr(error, "quota_scope", None),
                "quota_reset_at": _serialize_datetime(getattr(error, "quota_reset_at", None)),
                "eligible_routes_count": getattr(error, "eligible_routes_count", None),
                "exhausted_routes_count": getattr(error, "exhausted_routes_count", None),
                "disabled_routes_count": getattr(error, "disabled_routes_count", None),
                "fallback_provider": "openrouter",
            }
        )
        _LOGGER.warning("gemini_gateway_request", extra=event)
```

Update `_base_event()` signature and body:

```python
    def _base_event(
        self,
        *,
        request: GatewayRouteRequest,
        lease: RouteLease | None,
        latency_ms: int,
        status: str,
        route_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        route = route_context or {}
        return build_wide_event(
            event="gemini_gateway_request",
            service=self._service_name,
            environment=self._environment,
            request_id=request.request_id,
            source_service=request.source_service,
            chat_id=request.chat_id,
            telegram_message_id=request.telegram_message_id,
            model=request.model,
            route_label=lease.route_label if lease else route.get("route_label"),
            project_label=lease.project_label if lease else route.get("project_label"),
            key_label=lease.key_label if lease else route.get("key_label"),
            proxy_label=lease.proxy_label if lease else route.get("proxy_label"),
            transport_mode=lease.transport_mode if lease else route.get("transport_mode"),
            status=status,
            duration_ms=latency_ms,
            retry_count=getattr(request, "retry_count", 0),
        )
```

Add helper functions near `_attach_route_metadata()`:

```python
def _attach_static_route_metadata(
    response: ResponseT,
    route_context: dict[str, Any],
) -> ResponseT:
    updates = {
        "route": dict(route_context),
        "route_label": route_context["route_label"],
        "project_label": route_context["project_label"],
        "key_label": route_context["key_label"],
        "proxy_label": route_context.get("proxy_label"),
        "transport_mode": route_context["transport_mode"],
    }
    if hasattr(response, "model_copy"):
        return response.model_copy(update=updates)
    for key, value in updates.items():
        setattr(response, key, value)
    return response


def _attach_static_error_route_metadata(error: GatewayError, route_context: dict[str, Any]) -> None:
    error.route_label = route_context["route_label"]
    error.project_label = route_context["project_label"]
    error.key_label = route_context["key_label"]
    error.proxy_label = route_context.get("proxy_label")
    error.transport_mode = route_context["transport_mode"]
```

- [ ] **Step 4: Run task tests**

Run:

```bash
uv run pytest tests/gemini_gateway/test_routing.py -q
```

Expected: PASS.

- [ ] **Step 5: Hold commit until explicit approval**

Run:

```bash
git --no-pager diff -- gemini_gateway/service.py tests/gemini_gateway/test_routing.py
```

Expected: diff shows fallback only in embeddings path. Do not run `git commit` until the user explicitly asks for a commit.

## Task 5: Production Wiring

**Files:**
- Modify: `gemini_gateway/main.py`
- Modify: `tests/gemini_gateway/test_gateway_api_embeddings.py`

- [ ] **Step 1: Write failing main wiring test**

Extend `test_gateway_main_wires_embedding_client_into_production_service()` in `tests/gemini_gateway/test_gateway_api_embeddings.py` with these assertions:

```python
    assert "from gemini_gateway.openrouter_embedding_client import OpenRouterEmbeddingClient" in source
    assert "openrouter_embedding_client = OpenRouterEmbeddingClient(" in source[:service_start]
    assert "base_url=settings.openrouter_base_url" in source[:service_start]
    assert "openrouter_embedding_client=openrouter_embedding_client" in source[service_start:service_end]
    assert "openrouter_api_key=" in source[service_start:service_end]
    assert "openrouter_embeddings_fallback_enabled=settings.openrouter_embeddings_fallback_enabled" in source[
        service_start:service_end
    ]
    assert "openrouter_embeddings_fallback_model=settings.openrouter_embeddings_fallback_model" in source[
        service_start:service_end
    ]
```

- [ ] **Step 2: Run main wiring test and verify failure**

Run:

```bash
uv run pytest tests/gemini_gateway/test_gateway_api_embeddings.py::test_gateway_main_wires_embedding_client_into_production_service -q
```

Expected: FAIL because `main.py` does not wire OpenRouter.

- [ ] **Step 3: Wire OpenRouter client in `main.py`**

Add import:

```python
from gemini_gateway.openrouter_embedding_client import OpenRouterEmbeddingClient
```

After `embedding_client = GeminiEmbeddingClient(...)`, add:

```python
    openrouter_embedding_client = OpenRouterEmbeddingClient(
        base_url=settings.openrouter_base_url,
        timeout=settings.default_request_timeout_seconds,
    )
    openrouter_api_key = (
        settings.openrouter_api_key.get_secret_value()
        if settings.openrouter_api_key is not None
        else None
    )
```

Pass to `CompletionService`:

```python
        openrouter_embedding_client=openrouter_embedding_client,
        openrouter_api_key=openrouter_api_key,
        openrouter_embeddings_fallback_enabled=settings.openrouter_embeddings_fallback_enabled,
        openrouter_embeddings_fallback_model=settings.openrouter_embeddings_fallback_model,
```

- [ ] **Step 4: Run task tests**

Run:

```bash
uv run pytest tests/gemini_gateway/test_gateway_api_embeddings.py tests/gemini_gateway/test_settings.py -q
```

Expected: PASS.

- [ ] **Step 5: Hold commit until explicit approval**

Run:

```bash
git --no-pager diff -- gemini_gateway/main.py tests/gemini_gateway/test_gateway_api_embeddings.py
```

Expected: diff only wires configured fallback dependencies. Do not run `git commit` until the user explicitly asks for a commit.

## Task 6: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Add this section after the local seed instructions in `README.md`:

````markdown
## OpenRouter fallback для embeddings

Gateway может использовать OpenRouter без proxy только как fallback для `POST /v1/embeddings` и только для модели `google/gemini-embedding-2`.

Fallback выключен по умолчанию, чтобы случайно не включить платный путь. Для включения:

```powershell
$env:GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED="true"
$env:GEMINI_GATEWAY_OPENROUTER_API_KEY="sk-or-..."
```

Условия срабатывания:

- сначала gateway пытается выдать обычный Gemini route `api_key + proxy`;
- OpenRouter вызывается только если route pool не может выдать ни одного Gemini route для `google/gemini-embedding-2`;
- разрешённые причины: `no_route`, `cooldown_active`, `quota_exhausted`;
- fallback не срабатывает после ошибки одного уже выбранного Gemini route, потому что это не доказывает недоступность всех ключей;
- chat completions и TTS никогда не используют OpenRouter fallback.

В ответе route metadata будет `transport_mode: direct`, `project_label: openrouter-fallback`, `route_label: openrouter-embedding-fallback`.
````

- [ ] **Step 2: Check docs render context**

Run:

```bash
rg -n "OpenRouter fallback|GEMINI_GATEWAY_OPENROUTER|openrouter-embedding-fallback" README.md
```

Expected: output includes the new section and all three env/metadata terms.

- [ ] **Step 3: Hold commit until explicit approval**

Run:

```bash
git --no-pager diff -- README.md
```

Expected: README documents exact fallback scope and env vars. Do not run `git commit` until the user explicitly asks for a commit.

## Task 7: Full Verification

**Files:**
- No file modifications.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
uv run pytest tests/gemini_gateway/test_openrouter_embedding_client.py tests/gemini_gateway/test_routing.py tests/gemini_gateway/test_gateway_api_embeddings.py tests/gemini_gateway/test_settings.py tests/gemini_gateway/test_compose_config.py tests/gemini_gateway/test_contracts.py -q
```

Expected: PASS.

- [ ] **Step 2: Run all gateway tests**

Run:

```bash
uv run pytest tests/gemini_gateway -q
```

Expected: PASS.

- [ ] **Step 3: Inspect final diff**

Run:

```bash
git --no-pager diff --stat
git --no-pager diff -- gemini_gateway/config.py gemini_gateway/contracts.py gemini_gateway/openrouter_embedding_client.py gemini_gateway/service.py gemini_gateway/main.py gemini_gateway/errors.py docker-compose.yml docker-compose.dev.yml README.md tests/gemini_gateway/test_openrouter_embedding_client.py tests/gemini_gateway/test_settings.py tests/gemini_gateway/test_contracts.py tests/gemini_gateway/test_routing.py tests/gemini_gateway/test_gateway_api_embeddings.py tests/gemini_gateway/test_compose_config.py
```

Expected: changes are limited to fallback settings, direct response metadata, OpenRouter embeddings client, service fallback orchestration, compose wiring, docs, and tests.

- [ ] **Step 4: Optional container config validation**

Run:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml config >/tmp/gemini-gateway-compose-config.yaml
rg -n "GEMINI_GATEWAY_OPENROUTER" /tmp/gemini-gateway-compose-config.yaml
```

Expected: output includes `GEMINI_GATEWAY_OPENROUTER_API_KEY`, `GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED`, and `GEMINI_GATEWAY_OPENROUTER_BASE_URL`.

- [ ] **Step 5: Commit only after explicit user command**

If the user explicitly asks to commit, run:

```bash
git add gemini_gateway/config.py gemini_gateway/contracts.py gemini_gateway/openrouter_embedding_client.py gemini_gateway/service.py gemini_gateway/main.py gemini_gateway/errors.py docker-compose.yml docker-compose.dev.yml README.md tests/gemini_gateway/test_openrouter_embedding_client.py tests/gemini_gateway/test_settings.py tests/gemini_gateway/test_contracts.py tests/gemini_gateway/test_routing.py tests/gemini_gateway/test_gateway_api_embeddings.py tests/gemini_gateway/test_compose_config.py
git commit -m "feat: add openrouter embedding fallback"
```

Expected: commit succeeds only after tests pass and the user has explicitly requested a commit.

## Self-Review

- Spec coverage: The plan covers OpenRouter key support without proxy, restriction to `google/gemini-embedding-2`, fallback only after all Gemini routes are unavailable, safe user-facing errors, settings, compose wiring, tests, and docs.
- Placeholder scan: The plan contains concrete files, commands, code snippets, and expected outcomes.
- Type consistency: `OpenRouterEmbeddingClient.embed()` accepts `GatewayEmbeddingRequest` plus `api_key`; `CompletionService` stores that client and only calls it from `embed()`. Response metadata uses `transport_mode: "direct"` while seeded Gemini route models remain proxy-only.
