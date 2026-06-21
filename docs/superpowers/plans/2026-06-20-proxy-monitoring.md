# Proxy Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add readonly web monitoring for Gemini proxy latency/failures and record enough timing data on existing gateway traffic to distinguish upload/write, response-header wait, response-body read, parse, and timeout stages.

**Architecture:** Keep the gateway proxy-only policy intact and add observability around the existing provider calls. Persist safe aggregate timing fields on `gemini_gateway.route_attempts`, then expose protected admin JSON endpoints plus a lightweight static dashboard served by FastAPI. No proxy auto-disable behavior is added.

**Tech Stack:** Python 3.12, FastAPI, httpx, SQLAlchemy text queries, Alembic, Pydantic v2, vanilla HTML/CSS/JS, pytest/pytest-asyncio.

---

## File Structure

- Create `gemini_gateway/provider_observability.py`: shared provider HTTP timing, request/response byte accounting, payload classification, and safe error attachment helpers.
- Modify `gemini_gateway/gemini_client.py`: send OpenAI-compatible chat requests through the shared timed JSON helper.
- Modify `gemini_gateway/embedding_client.py`: send native embedding requests through the shared timed JSON helper and classify image/text payloads.
- Modify `gemini_gateway/tts_client.py`: send native TTS requests through the shared timed JSON helper and classify TTS payloads.
- Modify `gemini_gateway/provider_http_errors.py`: keep provider HTTP error extraction safe for invalid/non-UTF provider bodies so timing is not lost.
- Modify `gemini_gateway/contracts.py`: add optional safe observability fields to gateway responses if needed for service/repository handoff.
- Modify `gemini_gateway/errors.py`: carry optional provider observability fields on `GatewayError`.
- Modify `gemini_gateway/service.py`: merge provider observability into wide events and pass it to repository success/failure recording.
- Modify `gemini_gateway/repository.py`: persist observability fields on `route_attempts` for success and failure.
- Modify `gemini_gateway/db/models.py`: map new `route_attempts` columns.
- Create `migrations/versions/0002_route_attempt_observability.py`: additive migration for the new nullable columns and indexes.
- Create `gemini_gateway/monitoring.py`: read-only SQL query functions for proxy overview, time series, route samples, and current route state.
- Modify `gemini_gateway/api.py`: add protected `/admin/monitor`, `/admin/monitor/api/summary`, and `/admin/monitor/api/timeseries` routes.
- Modify `gemini_gateway/main.py`: pass the Postgres session factory into the API factory for monitoring endpoints.
- Create `tests/gemini_gateway/test_provider_observability.py`: focused unit tests for timing metadata and classification.
- Modify `tests/gemini_gateway/test_gemini_client.py`, `tests/gemini_gateway/test_embedding_client.py`, `tests/gemini_gateway/test_tts_client.py`: verify clients attach observability on success and timeout.
- Modify `tests/gemini_gateway/test_repository_postgres.py`: verify repository persists observability fields.
- Create `tests/gemini_gateway/test_monitoring.py`: unit tests for monitoring query summarizers using sample rows.
- Modify `tests/gemini_gateway/test_gateway_api.py`: verify admin monitor data endpoints require bearer auth and return safe data.
- Modify `README.md`: document monitor URL, token behavior, graph meanings, and how to interpret timeout stages.

No commits are made during this plan because project instructions forbid commits without a direct user command.

---

### Task 1: Provider Observability Primitives

**Files:**
- Create: `gemini_gateway/provider_observability.py`
- Test: `tests/gemini_gateway/test_provider_observability.py`

- [ ] **Step 1: Write failing tests for payload classification**

Add tests that define the desired public API:

```python
from gemini_gateway.provider_observability import classify_chat_payload, classify_embedding_payload, classify_tts_payload


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
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest tests/gemini_gateway/test_provider_observability.py -q
```

Expected: FAIL because `gemini_gateway.provider_observability` does not exist.

- [ ] **Step 3: Implement classification models and helpers**

Create `gemini_gateway/provider_observability.py` with these public objects:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field


ProviderOperationType = Literal["chat", "embedding", "tts"]
ProviderPayloadKind = Literal["text", "media", "tts"]
ProviderTimeoutStage = Literal["request", "response_headers", "response_body", "unknown"]


class ProviderPayloadSummary(BaseModel):
    """Безопасная сводка payload без текста, base64 и секретов."""

    operation_type: ProviderOperationType
    payload_kind: ProviderPayloadKind
    media_count: int = Field(ge=0)
    image_count: int = Field(ge=0)


class ProviderTimingSummary(BaseModel):
    """Безопасные замеры HTTP-вызова провайдера."""

    operation_type: ProviderOperationType
    payload_kind: ProviderPayloadKind
    request_bytes: int = Field(ge=0)
    response_bytes: int | None = Field(default=None, ge=0)
    media_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    provider_total_ms: int = Field(ge=0)
    request_prepare_ms: int = Field(ge=0)
    response_headers_ms: int | None = Field(default=None, ge=0)
    response_body_ms: int | None = Field(default=None, ge=0)
    response_parse_ms: int | None = Field(default=None, ge=0)
    timeout_kind: str | None = None
    timeout_stage: ProviderTimeoutStage | None = None


@dataclass(frozen=True)
class TimedJsonResponse:
    response: httpx.Response
    payload: dict[str, Any]
    body_bytes: bytes
    timing: ProviderTimingSummary
```

The implementation must include:

```python
def classify_chat_payload(payload: dict[str, Any]) -> ProviderPayloadSummary: ...
def classify_embedding_payload(payload: dict[str, Any]) -> ProviderPayloadSummary: ...
def classify_tts_payload(payload: dict[str, Any]) -> ProviderPayloadSummary: ...
def timing_to_dict(value: ProviderTimingSummary | dict[str, Any] | None) -> dict[str, Any]: ...
def attach_timing_to_error(error: Exception, timing: ProviderTimingSummary | None) -> None: ...
```

Rules:
- Do not store prompt text, response text, image base64, API keys, proxy URLs, headers, or cookies.
- `payload_kind="media"` when any image/media part exists.
- `request_bytes` is the byte length of compact UTF-8 JSON sent upstream.
- All durations are non-negative integer milliseconds.

- [ ] **Step 4: Add timed JSON sender tests**

Add a success test using `httpx.MockTransport`:

```python
import json
import httpx
import pytest

from gemini_gateway.provider_observability import classify_chat_payload, send_timed_json


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
```

Add a read-timeout test:

```python
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
```

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
python -m pytest tests/gemini_gateway/test_provider_observability.py -q
```

Expected: PASS.

---

### Task 2: Instrument Provider Clients

**Files:**
- Modify: `gemini_gateway/gemini_client.py`
- Modify: `gemini_gateway/embedding_client.py`
- Modify: `gemini_gateway/tts_client.py`
- Modify: `gemini_gateway/provider_http_errors.py`
- Modify: `gemini_gateway/errors.py`
- Modify: `gemini_gateway/contracts.py`
- Test: `tests/gemini_gateway/test_gemini_client.py`
- Test: `tests/gemini_gateway/test_embedding_client.py`
- Test: `tests/gemini_gateway/test_tts_client.py`
- Test: `tests/gemini_gateway/test_provider_http_errors.py`

- [ ] **Step 1: Write failing client tests**

Add tests asserting successful responses expose `provider_timing`:

```python
assert response.provider_timing["operation_type"] == "chat"
assert response.provider_timing["payload_kind"] == "media"
assert response.provider_timing["request_bytes"] > 0
assert response.provider_timing["response_bytes"] > 0
assert response.provider_timing["response_headers_ms"] is not None
```

Add timeout tests asserting `GatewayError` carries timing:

```python
with pytest.raises(GatewayError) as exc_info:
    await client.complete(...)

assert exc_info.value.provider_timing["timeout_kind"] == "read_timeout"
assert exc_info.value.provider_timing["timeout_stage"] == "response_headers"
assert exc_info.value.provider_timing["request_bytes"] > 0
```

Use existing client test style with `httpx.MockTransport`. Keep assertions free of raw proxy URL and API key values.

- [ ] **Step 2: Run focused client tests and verify RED**

Run:

```bash
python -m pytest tests/gemini_gateway/test_gemini_client.py tests/gemini_gateway/test_embedding_client.py tests/gemini_gateway/test_tts_client.py -q
```

Expected: FAIL because response/error objects do not yet expose `provider_timing`.

- [ ] **Step 3: Add safe timing fields to response/error objects**

In `GatewayChatResponse`, `GatewayEmbeddingResponse`, and `GatewayTTSResponse`, add:

```python
provider_timing: dict[str, Any] = Field(default_factory=dict)
```

In `GatewayError.__init__`, accept:

```python
provider_timing: dict[str, Any] | None = None
```

and assign:

```python
self.provider_timing = provider_timing or {}
```

Do not include `provider_timing` in end-user error responses. It is for logs, DB, and monitor only.

- [ ] **Step 4: Use the timed sender in all Gemini clients**

Replace direct `client.post(..., json=payload, timeout=...)` calls with:

```python
result = await send_timed_json(
    client=self._client_pool.get(proxy_url=proxy_url),
    method="POST",
    url=f"{self._base_url}/chat/completions",
    headers=headers,
    payload=payload,
    payload_summary=classify_chat_payload(payload),
    timeout_seconds=request.timeout_seconds,
)
response = result.response
raw_response = result.payload
```

For non-pool client branches, use the same helper inside `async with httpx.AsyncClient(...)`.

For `httpx.ProxyError`, `httpx.TimeoutException`, and `httpx.HTTPError`, read `getattr(exc, "provider_timing", None)`, convert it with `timing_to_dict`, and pass it to `GatewayError(provider_timing=...)`.

For provider HTTP error responses and invalid JSON errors, attach timing from the successful HTTP exchange to the raised `GatewayError`.

- [ ] **Step 5: Run focused client tests and verify GREEN**

Run:

```bash
python -m pytest tests/gemini_gateway/test_gemini_client.py tests/gemini_gateway/test_embedding_client.py tests/gemini_gateway/test_tts_client.py -q
```

Expected: PASS.

---

### Task 3: Persist Route Attempt Observability

**Files:**
- Modify: `gemini_gateway/db/models.py`
- Create: `migrations/versions/0002_route_attempt_observability.py`
- Modify: `gemini_gateway/repository.py`
- Modify: `gemini_gateway/service.py`
- Test: `tests/gemini_gateway/test_repository_postgres.py`
- Test: `tests/gemini_gateway/test_routing.py`

- [ ] **Step 1: Write failing persistence tests**

Add a Postgres repository test that records a success response with `provider_timing` and reads the saved row:

```python
response = GatewayChatResponse(
    request_id=request.request_id,
    model=request.model,
    choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    provider_timing={
        "operation_type": "chat",
        "payload_kind": "media",
        "request_bytes": 2048,
        "response_bytes": 512,
        "media_count": 2,
        "image_count": 2,
        "provider_total_ms": 1200,
        "request_prepare_ms": 2,
        "response_headers_ms": 900,
        "response_body_ms": 100,
        "response_parse_ms": 3,
        "timeout_kind": None,
        "timeout_stage": None,
    },
)
```

Assert the row contains every value above.

Add a failure test that calls `record_failure` with a `GatewayError(provider_timing={... "timeout_kind": "read_timeout", "timeout_stage": "response_headers"})` and asserts those columns are persisted.

- [ ] **Step 2: Run persistence tests and verify RED**

Run:

```bash
python -m pytest tests/gemini_gateway/test_repository_postgres.py -q
```

Expected: FAIL until migration/model/repository are updated. If local Postgres is unavailable, run the non-DB routing tests and report DB tests as skipped by project policy.

- [ ] **Step 3: Add migration and ORM columns**

Create additive migration with nullable columns:

```python
op.add_column("route_attempts", sa.Column("operation_type", sa.String(length=32), nullable=True), schema=SCHEMA)
op.add_column("route_attempts", sa.Column("payload_kind", sa.String(length=32), nullable=True), schema=SCHEMA)
op.add_column("route_attempts", sa.Column("request_bytes", sa.Integer(), nullable=True), schema=SCHEMA)
op.add_column("route_attempts", sa.Column("response_bytes", sa.Integer(), nullable=True), schema=SCHEMA)
op.add_column("route_attempts", sa.Column("media_count", sa.Integer(), nullable=True), schema=SCHEMA)
op.add_column("route_attempts", sa.Column("image_count", sa.Integer(), nullable=True), schema=SCHEMA)
op.add_column("route_attempts", sa.Column("provider_total_ms", sa.Integer(), nullable=True), schema=SCHEMA)
op.add_column("route_attempts", sa.Column("request_prepare_ms", sa.Integer(), nullable=True), schema=SCHEMA)
op.add_column("route_attempts", sa.Column("response_headers_ms", sa.Integer(), nullable=True), schema=SCHEMA)
op.add_column("route_attempts", sa.Column("response_body_ms", sa.Integer(), nullable=True), schema=SCHEMA)
op.add_column("route_attempts", sa.Column("response_parse_ms", sa.Integer(), nullable=True), schema=SCHEMA)
op.add_column("route_attempts", sa.Column("timeout_kind", sa.String(length=64), nullable=True), schema=SCHEMA)
op.add_column("route_attempts", sa.Column("timeout_stage", sa.String(length=64), nullable=True), schema=SCHEMA)
op.create_index("ix_route_attempts_proxy_created", "route_attempts", ["proxy_id", "created_at"], schema=SCHEMA)
op.create_index("ix_route_attempts_operation_created", "route_attempts", ["operation_type", "created_at"], schema=SCHEMA)
```

Add matching nullable mapped columns to `RouteAttempt`.

- [ ] **Step 4: Persist timing on success and failure**

In `repository.py`, extract safe integer/string values from `response.provider_timing` and `error.provider_timing` through one helper:

```python
def _provider_timing_columns(value: Any) -> dict[str, Any]:
    timing = timing_to_dict(value)
    return {
        "operation_type": _optional_timing_string(timing.get("operation_type"), allowed={"chat", "embedding", "tts"}),
        "payload_kind": _optional_timing_string(timing.get("payload_kind"), allowed={"text", "media", "tts"}),
        "request_bytes": _optional_non_negative_int(timing.get("request_bytes")),
        "response_bytes": _optional_non_negative_int(timing.get("response_bytes")),
        "media_count": _optional_non_negative_int(timing.get("media_count")),
        "image_count": _optional_non_negative_int(timing.get("image_count")),
        "provider_total_ms": _optional_non_negative_int(timing.get("provider_total_ms")),
        "request_prepare_ms": _optional_non_negative_int(timing.get("request_prepare_ms")),
        "response_headers_ms": _optional_non_negative_int(timing.get("response_headers_ms")),
        "response_body_ms": _optional_non_negative_int(timing.get("response_body_ms")),
        "response_parse_ms": _optional_non_negative_int(timing.get("response_parse_ms")),
        "timeout_kind": _optional_timing_string(timing.get("timeout_kind")),
        "timeout_stage": _optional_timing_string(timing.get("timeout_stage")),
    }
```

Update the existing `UPDATE gemini_gateway.route_attempts SET ...` statements in `record_success` and `record_failure`.

- [ ] **Step 5: Include observability in wide events**

In `service.py`, merge provider timing into `_success_log_fields` and `_failure_log_fields` with the same safe field names. Do not include raw payloads, prompt text, response text, proxy URLs, API keys, or headers.

- [ ] **Step 6: Run tests and verify GREEN**

Run:

```bash
python -m pytest tests/gemini_gateway/test_repository_postgres.py tests/gemini_gateway/test_routing.py -q
```

Expected: PASS or DB tests skipped only when the configured local Postgres is unavailable.

---

### Task 4: Monitoring Queries and Protected Admin API

**Files:**
- Create: `gemini_gateway/monitoring.py`
- Modify: `gemini_gateway/api.py`
- Modify: `gemini_gateway/main.py`
- Test: `tests/gemini_gateway/test_monitoring.py`
- Test: `tests/gemini_gateway/test_gateway_api.py`

- [ ] **Step 1: Write failing monitoring tests**

For pure summarizer behavior, create sample rows and assert:

```python
summary = summarize_proxy_overview_rows(rows)
assert summary["total_requests"] == 3
assert summary["proxy_count"] == 2
assert summary["proxies"][0]["failure_rate"] == 0.5
assert summary["proxies"][0]["read_timeout_count"] == 1
assert summary["proxies"][0]["p95_latency_ms"] == 92010
```

For API behavior, add tests:

```python
response = client.get("/admin/monitor/api/summary")
assert response.status_code == 401
assert response.json()["error"]

response = client.get("/admin/monitor/api/summary", headers={"Authorization": "Bearer secret-token"})
assert response.status_code == 200
assert "proxies" in response.json()
assert "proxy_url" not in response.text
```

- [ ] **Step 2: Run API/monitoring tests and verify RED**

Run:

```bash
python -m pytest tests/gemini_gateway/test_monitoring.py tests/gemini_gateway/test_gateway_api.py -q
```

Expected: FAIL because monitoring module and routes do not exist.

- [ ] **Step 3: Implement monitoring query service**

Create `gemini_gateway/monitoring.py` with:

```python
@dataclass(frozen=True)
class MonitorWindow:
    minutes: int = 180
    bucket_seconds: int = 60
    model: str | None = None
    proxy_label: str | None = None


async def fetch_proxy_summary(session_factory: async_sessionmaker[AsyncSession], window: MonitorWindow) -> dict[str, Any]: ...
async def fetch_proxy_timeseries(session_factory: async_sessionmaker[AsyncSession], window: MonitorWindow) -> dict[str, Any]: ...
def summarize_proxy_overview_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]: ...
def summarize_proxy_timeseries_rows(rows: Iterable[Mapping[str, Any]], *, bucket_seconds: int) -> dict[str, Any]: ...
```

SQL must:
- Read only from `gemini_gateway.route_attempts`, `proxy_endpoints`, `key_proxy_bindings`, `api_keys`, `google_projects`, and `cooldowns`.
- Filter by `created_at >= now() - window`.
- Group by `proxy_id`, `proxy_label`, `route_label`.
- Return only labels, counts, latencies, payload sizes, timeout kinds/stages, status, and cooldown state.
- Never return encrypted credentials, proxy host/port, API key fingerprint, raw provider payload, prompt text, response text, or base64.

Use Postgres percentile functions when reading from DB:

```sql
percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE latency_ms IS NOT NULL) AS p95_latency_ms
```

- [ ] **Step 4: Add protected admin JSON routes**

In `api.py`, accept optional `monitoring_session_factory` in `create_app`.

Add:

```python
@app.get("/admin/monitor/api/summary")
async def monitor_summary(authorization: str | None = Header(default=None), minutes: int = 180, model: str | None = None, proxy_label: str | None = None) -> JSONResponse:
    ...

@app.get("/admin/monitor/api/timeseries")
async def monitor_timeseries(authorization: str | None = Header(default=None), minutes: int = 180, bucket_seconds: int = 60, model: str | None = None, proxy_label: str | None = None) -> JSONResponse:
    ...
```

Both endpoints must use `_is_authorized`. On internal errors, log safe `gemini_gateway_monitor_error` and return:

```python
{"error": "Не удалось загрузить мониторинг, попробуйте позже"}
```

No stack traces or raw DB errors reach the response.

In `main.py`, pass `session_factory` into `create_app`.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
python -m pytest tests/gemini_gateway/test_monitoring.py tests/gemini_gateway/test_gateway_api.py -q
```

Expected: PASS.

---

### Task 5: Web Dashboard

**Files:**
- Modify: `gemini_gateway/api.py`
- Test: `tests/gemini_gateway/test_gateway_api.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing dashboard tests**

Add API tests:

```python
response = client.get("/admin/monitor")
assert response.status_code == 200
assert "Gemini Proxy Monitor" in response.text
assert "/admin/monitor/api/summary" in response.text
assert "Введите токен мониторинга" in response.text
```

Assert the page does not contain raw secrets or proxy URLs.

- [ ] **Step 2: Run dashboard tests and verify RED**

Run:

```bash
python -m pytest tests/gemini_gateway/test_gateway_api.py -q
```

Expected: FAIL because `/admin/monitor` is not implemented.

- [ ] **Step 3: Serve lightweight dashboard HTML**

Add a route:

```python
@app.get("/admin/monitor")
async def monitor_page() -> HTMLResponse:
    return HTMLResponse(_monitor_dashboard_html())
```

Dashboard requirements:
- Vanilla HTML/CSS/JS only.
- Token input stores bearer token in `localStorage`.
- Fetches summary and timeseries with `Authorization: Bearer <token>`.
- Shows proxy cards: current status, request count, failure rate, p95 latency, read timeout count, media count, cooldown status.
- Shows timeline charts using inline SVG or canvas, no external CDN.
- Filters: minutes, model, proxy label, bucket seconds.
- Uses Russian user-facing error fallback: `Не удалось загрузить мониторинг, попробуйте позже`.
- Does not display raw HTTP status codes, stack traces, proxy credentials, API key details, raw provider messages, base64, prompt text, or response text.

- [ ] **Step 4: Document usage and interpretation**

In `README.md`, add an operations section:

```markdown
## Proxy monitor

Веб-монитор доступен на `/admin/monitor`. Вставьте `GEMINI_GATEWAY_TOKEN`; страница хранит его в `localStorage` браузера и отправляет только в `Authorization` для JSON API.

Основные признаки:
- `response_headers_ms` высокий: запрос ушел через proxy, но долго не было ответа от Gemini/proxy.
- `response_body_ms` высокий: ответ начал идти, но тело читалось медленно.
- `timeout_kind=write_timeout`: проблема на отправке тела запроса.
- `timeout_kind=read_timeout` + `timeout_stage=response_headers`: долго ждали первый ответ.
- `payload_kind=media` помогает отделять медиа-запросы от обычного текста.
```

- [ ] **Step 5: Run dashboard/API tests and verify GREEN**

Run:

```bash
python -m pytest tests/gemini_gateway/test_gateway_api.py tests/gemini_gateway/test_monitoring.py -q
```

Expected: PASS.

---

### Task 6: Full Verification and Operational Sanity

**Files:**
- Verify all files touched by Tasks 1-5.

- [ ] **Step 1: Run focused non-DB tests**

Run:

```bash
python -m pytest tests/gemini_gateway/test_provider_observability.py tests/gemini_gateway/test_gemini_client.py tests/gemini_gateway/test_embedding_client.py tests/gemini_gateway/test_tts_client.py tests/gemini_gateway/test_gateway_api.py tests/gemini_gateway/test_monitoring.py tests/gemini_gateway/test_routing.py -q
```

Expected: PASS.

- [ ] **Step 2: Run DB tests if local Postgres is available**

Run:

```bash
python -m pytest tests/gemini_gateway/test_repository_postgres.py -q
```

Expected: PASS or project-policy skip/fail explaining local Postgres unavailability.

- [ ] **Step 3: Run full test suite**

Run:

```bash
python -m pytest tests/gemini_gateway -q
```

Expected: PASS, except DB tests may be skipped by existing project policy when DB is unavailable.

- [ ] **Step 4: Check migration ordering**

Run:

```bash
python -m pytest tests/gemini_gateway/test_standalone_boundaries.py tests/gemini_gateway/test_compose_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Manual dashboard smoke check**

If a dev server/container is running, open `/admin/monitor`, enter the gateway token, and verify:
- Summary endpoint returns labels and metrics.
- Timeseries draws non-empty axes even with no data.
- Unauthorized endpoint calls show a human-readable Russian fallback.
- No raw proxy URLs, API keys, stack traces, or SQL errors appear in browser-visible text.

If no dev server is running, report that browser smoke was not run and include the exact tests that covered the page.
