from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from gemini_gateway.contracts import (
    GatewayChatRequest,
    GatewayChatResponse,
    GatewayEmbeddingRequest,
    GatewayEmbeddingResponse,
    GatewayTTSRequest,
    GatewayTTSResponse,
    RouteCandidate,
    RouteLease,
)
from gemini_gateway.errors import GatewayError
from gemini_gateway.gemini_client import GeminiOpenAIClient
from gemini_gateway.repository import (
    ATTEMPTED_ROUTES_EXHAUSTED_ERROR_CODE,
    InMemoryRouteRepository,
    PostgresGatewayRepository,
    RouteScorer,
    _RouteAcquisitionDiagnostics,
)
from gemini_gateway.repository import _safe_provider_response_json
from gemini_gateway.service import CompletionService

_OPENROUTER_EMBEDDING_MODEL = "google/gemini-embedding-2"
_OPENROUTER_API_KEY = "sk-or-test-openrouter-secret"


def _candidate(
    binding_id: str,
    *,
    api_key: str = "AIza-key",
    proxy_url: str = "http://user:pass@127.0.0.1:8000",
    minute_tokens_reserved: int = 0,
    day_requests_used: int = 0,
    day_tokens_reserved: int = 0,
    cooldown_until: datetime | None = None,
    half_open: bool = False,
) -> RouteCandidate:
    return RouteCandidate(
        binding_id=binding_id,
        project_id=f"project-{binding_id}",
        api_key_id=f"key-id-{binding_id}",
        proxy_id=f"proxy-id-{binding_id}",
        api_key=api_key,
        proxy_url=proxy_url,
        model="gemini-3.5-flash",
        route_label=f"route-{binding_id}",
        project_label=f"friend-{binding_id}",
        key_label=f"key-{binding_id}",
        proxy_label=f"proxy-{binding_id}",
        requests_per_minute=10,
        tokens_per_minute=1_000,
        requests_per_day=100,
        minute_requests_used=0,
        minute_tokens_reserved=minute_tokens_reserved,
        day_requests_used=day_requests_used,
        cooldown_until=cooldown_until,
        day_tokens_reserved=day_tokens_reserved,
        half_open=half_open,
    )


def _embedding_request(
    *,
    request_id: str = "req-openrouter-fallback",
    model: str = _OPENROUTER_EMBEDDING_MODEL,
) -> GatewayEmbeddingRequest:
    return GatewayEmbeddingRequest(
        request_id=request_id,
        source_service="media_memory",
        model=model,
        input=[{"type": "text", "text": "кот на диване"}],
        dimensions=1536,
        chat_id=42,
    )


class _RouteFailureRepository:
    def __init__(self, reason: str) -> None:
        self._reason = reason
        self.failures: list[dict[str, Any]] = []
        self.successes: list[dict[str, Any]] = []

    async def acquire_route(self, request: GatewayEmbeddingRequest) -> None:
        raise GatewayError(reason=self._reason, retryable=True, request_id=request.request_id)

    async def record_success(self, lease: Any, response: Any, latency_ms: int) -> None:
        self.successes.append({"lease": lease, "response": response, "latency_ms": latency_ms})

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
                "error": error,
                "latency_ms": latency_ms,
                "provider_called": provider_called,
            }
        )


class _RecordingGeminiEmbeddingClient:
    def __init__(self) -> None:
        self.called = False

    async def embed(
        self,
        request: GatewayEmbeddingRequest,
        api_key: str,
        proxy_url: str,
    ) -> GatewayEmbeddingResponse:
        self.called = True
        return GatewayEmbeddingResponse(
            request_id=request.request_id,
            model=request.model,
            embedding=[0.1] * request.dimensions,
            dimensions=request.dimensions,
            usage={"total_tokens": 5},
        )


class _RecordingOpenRouterEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def embed(
        self,
        *,
        request: GatewayEmbeddingRequest,
        api_key: str,
    ) -> GatewayEmbeddingResponse:
        self.calls.append({"request": request, "api_key": api_key})
        return GatewayEmbeddingResponse(
            request_id=request.request_id,
            generation_id="gen-openrouter-fallback",
            model=request.model,
            embedding=[0.2] * request.dimensions,
            dimensions=request.dimensions,
            usage={"prompt_tokens": 3, "total_tokens": 3},
        )


class _AcquireRecordingRepository:
    def __init__(self) -> None:
        self.acquire_called = False
        self.failures: list[dict[str, Any]] = []

    async def acquire_route(self, request: GatewayEmbeddingRequest) -> RouteLease:
        self.acquire_called = True
        return RouteLease(
            attempt_id="attempt-preflight",
            binding_id="binding-preflight",
            project_id="project-preflight",
            api_key_id="key-preflight",
            proxy_id="proxy-preflight",
            api_key="AIza-preflight",
            proxy_url="http://127.0.0.1:9000",
            model=request.model,
            route_label="route-preflight",
            project_label="project-preflight",
            key_label="key-preflight",
            proxy_label="proxy-preflight",
            estimated_tokens=10,
            leased_at=datetime.now(tz=UTC),
        )

    async def record_failure(
        self,
        lease: RouteLease | None,
        error: GatewayError,
        latency_ms: int,
        provider_called: bool,
    ) -> None:
        self.failures.append(
            {
                "lease": lease,
                "error": error,
                "latency_ms": latency_ms,
                "provider_called": provider_called,
            }
        )


def test_route_scorer_skips_cooldowns_and_insufficient_token_budget() -> None:
    now = datetime(2026, 5, 23, tzinfo=UTC)

    chosen = RouteScorer.choose(
        [
            _candidate("busy", minute_tokens_reserved=900),
            _candidate("cooldown", cooldown_until=now + timedelta(minutes=5)),
            _candidate("ready", minute_tokens_reserved=100),
        ],
        estimated_tokens=200,
        now=now,
    )

    assert chosen is not None
    assert chosen.binding_id == "ready"
    assert RouteScorer.choose([_candidate("too-small", minute_tokens_reserved=900)], 200, now) is None
    assert RouteScorer.choose([_candidate("half-open", half_open=True)], 4097, now) is None


def test_route_scorer_rejects_direct_candidates_for_proxy_only_policy() -> None:
    now = datetime(2026, 5, 30, tzinfo=UTC)
    direct_candidate = _candidate("direct").model_copy(
        update={
            "transport_mode": "direct",
            "proxy_id": None,
            "proxy_url": None,
            "proxy_label": None,
        }
    )

    chosen = RouteScorer.choose([direct_candidate], estimated_tokens=100, now=now)

    assert chosen is None


def test_route_scorer_allows_proxy_candidates_before_secret_decryption() -> None:
    now = datetime(2026, 5, 30, tzinfo=UTC)
    proxy_candidate = _candidate("proxy").model_copy(update={"proxy_url": None})

    chosen = RouteScorer.choose([proxy_candidate], estimated_tokens=100, now=now)

    assert chosen is not None
    assert chosen.binding_id == "proxy"


def test_gateway_repository_omits_raw_provider_response_from_attempt_payload() -> None:
    audio_base64 = "UklGRg==" * 4096
    chat_response = GatewayChatResponse(
        request_id="req-chat",
        model="gemini-3.5-flash",
        generation_id="gen-chat",
        choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": "raw assistant text"}}],
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
            "headers": {"authorization": "Bearer secret"},
            "response_body": "raw provider body",
            "proxy_url": "http://user:pass@127.0.0.1:8000",
        },
        raw_response={"body": "raw response body"},
    )
    tts_response = GatewayTTSResponse(
        request_id="req-tts",
        model="google/gemini-tts",
        audio_base64=audio_base64,
        audio_mime_type="audio/wav",
        provider_timing={"operation_type": "tts", "payload_kind": "tts", "request_bytes": 33},
    )
    embedding_response = GatewayEmbeddingResponse(
        request_id="req-embedding",
        model="google/gemini-embedding-2",
        embedding=[0.1, 0.2, 0.3],
        dimensions=3,
        provider_timing={"operation_type": "embedding", "payload_kind": "text", "request_bytes": 44},
    )

    payload = _safe_provider_response_json(chat_response)
    tts_payload = _safe_provider_response_json(tts_response)
    embedding_payload = _safe_provider_response_json(embedding_response)
    serialized = json.dumps(payload, ensure_ascii=False)
    all_payloads = json.dumps([payload, tts_payload, embedding_payload], ensure_ascii=False)

    assert payload == {
        "request_id": "req-chat",
        "model": "gemini-3.5-flash",
        "generation_id": "gen-chat",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "provider_specific_fields": {},
        "route": {},
        "provider_timing": {
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
    }
    assert tts_payload["provider_timing"] == {
        "operation_type": "tts",
        "payload_kind": "tts",
        "request_bytes": 33,
        "response_bytes": None,
        "media_count": None,
        "image_count": None,
        "provider_total_ms": None,
        "request_prepare_ms": None,
        "response_headers_ms": None,
        "response_body_ms": None,
        "response_parse_ms": None,
        "timeout_kind": None,
        "timeout_stage": None,
    }
    assert embedding_payload["provider_timing"] == {
        "operation_type": "embedding",
        "payload_kind": "text",
        "request_bytes": 44,
        "response_bytes": None,
        "media_count": None,
        "image_count": None,
        "provider_total_ms": None,
        "request_prepare_ms": None,
        "response_headers_ms": None,
        "response_body_ms": None,
        "response_parse_ms": None,
        "timeout_kind": None,
        "timeout_stage": None,
    }
    assert "choices" not in payload
    assert "raw assistant text" not in serialized
    assert "raw response body" not in serialized
    assert "audio_base64" not in tts_payload
    assert audio_base64 not in all_payloads
    assert "embedding" not in embedding_payload
    assert "Bearer secret" not in all_payloads
    assert "raw provider body" not in all_payloads
    assert "user:pass" not in all_payloads
    assert "raw_response" not in payload
    assert "candidates" not in payload


@pytest.mark.asyncio
async def test_repository_acquire_route_reserves_budget_and_success_reconciles_usage() -> None:
    repository = InMemoryRouteRepository([_candidate("a")])
    request = GatewayChatRequest(
        request_id="req-1",
        source_service="test",
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "hello"}],
        estimated_input_tokens=100,
    )

    lease = await repository.acquire_route(request)
    [reserved_route] = await repository.list_route_candidates("gemini-3.5-flash", datetime.now(tz=UTC))

    assert lease.proxy_url == "http://user:pass@127.0.0.1:8000"
    assert reserved_route.minute_requests_used == 1
    assert reserved_route.minute_tokens_reserved == 100

    await repository.record_success(
        lease,
        GatewayChatResponse(
            request_id="req-1",
            generation_id="gen-1",
            model="gemini-3.5-flash",
            choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            raw_response={"id": "gen-1"},
        ),
        latency_ms=25,
    )
    [reconciled_route] = await repository.list_route_candidates("gemini-3.5-flash", datetime.now(tz=UTC))

    assert reconciled_route.minute_tokens_reserved == 30
    assert repository.successes[-1]["latency_ms"] == 25


@pytest.mark.asyncio
async def test_repository_does_not_reuse_route_inside_same_soybob_request() -> None:
    repository = InMemoryRouteRepository([_candidate("a"), _candidate("b"), _candidate("c")])
    leases = []

    for attempt_index in range(3):
        leases.append(
            await repository.acquire_route(
                GatewayChatRequest(
                    request_id=f"req-pool-{attempt_index}",
                    soybob_request_id="req-pool-group",
                    source_service="test",
                    model="gemini-3.5-flash",
                    messages=[{"role": "user", "content": "hello"}],
                    estimated_input_tokens=100,
                    retry_count=attempt_index,
                )
            )
        )

    with pytest.raises(GatewayError) as exc_info:
        await repository.acquire_route(
            GatewayChatRequest(
                request_id="req-pool-overflow",
                soybob_request_id="req-pool-group",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hello"}],
                estimated_input_tokens=100,
                retry_count=3,
            )
        )

    assert [lease.binding_id for lease in leases] == ["a", "b", "c"]
    assert exc_info.value.reason == "no_route"
    assert exc_info.value.error_code == ATTEMPTED_ROUTES_EXHAUSTED_ERROR_CODE


@pytest.mark.asyncio
async def test_repository_preserves_remaining_route_diagnostics_after_attempted_exclusion() -> None:
    repository = InMemoryRouteRepository([_candidate("attempted"), _candidate("daily-quota", day_requests_used=100)])
    await repository.acquire_route(
        GatewayChatRequest(
            request_id="req-attempted-first",
            soybob_request_id="req-mixed-diagnostics",
            source_service="test",
            model="gemini-3.5-flash",
            messages=[{"role": "user", "content": "hello"}],
            estimated_input_tokens=100,
        )
    )

    with pytest.raises(GatewayError) as exc_info:
        await repository.acquire_route(
            GatewayChatRequest(
                request_id="req-attempted-second",
                soybob_request_id="req-mixed-diagnostics",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hello"}],
                estimated_input_tokens=100,
                retry_count=1,
            )
        )

    error = exc_info.value
    assert error.reason == "quota_exhausted"
    assert error.error_code == ATTEMPTED_ROUTES_EXHAUSTED_ERROR_CODE
    assert error.quota_scope == "day"
    assert error.eligible_routes_count == 1
    assert error.exhausted_routes_count == 1


class _DiagnosticsPostgresRepository(PostgresGatewayRepository):
    def __init__(self) -> None:
        self.diagnostics_calls: list[dict[str, Any]] = []
        self.skipped: list[dict[str, Any]] = []

    async def list_route_candidates(self, model: str, now: datetime) -> list[RouteCandidate]:
        del model, now
        return [_candidate("attempted")]

    async def _attempted_binding_ids(self, *, request: GatewayChatRequest) -> set[str]:
        del request
        return {"attempted"}

    async def _route_unavailability_diagnostics(
        self,
        *,
        model: str,
        estimated_tokens: int,
        now: datetime,
        excluded_binding_ids: set[str] | None = None,
    ) -> _RouteAcquisitionDiagnostics:
        self.diagnostics_calls.append(
            {
                "model": model,
                "estimated_tokens": estimated_tokens,
                "excluded_binding_ids": excluded_binding_ids,
                "now": now,
            }
        )
        return _RouteAcquisitionDiagnostics(
            reason="quota_exhausted",
            error_code="quota_exhausted",
            retry_after_seconds=300,
            quota_scope="day",
            quota_reset_at="2026-06-22T00:00:00+00:00",
            eligible_routes_count=1,
            exhausted_routes_count=1,
            disabled_routes_count=0,
        )

    async def _record_skipped_route_unavailable(
        self,
        *,
        request: GatewayChatRequest,
        estimated_tokens: int,
        reason: str,
    ) -> None:
        self.skipped.append({"request_id": request.request_id, "estimated_tokens": estimated_tokens, "reason": reason})


@pytest.mark.asyncio
async def test_postgres_repository_preserves_diagnostics_after_attempted_exclusion() -> None:
    repository = _DiagnosticsPostgresRepository()

    with pytest.raises(GatewayError) as exc_info:
        await repository.acquire_route(
            GatewayChatRequest(
                request_id="req-postgres-attempted",
                soybob_request_id="req-postgres-attempted-group",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hello"}],
                estimated_input_tokens=100,
                retry_count=1,
            )
        )

    error = exc_info.value
    assert error.reason == "quota_exhausted"
    assert error.error_code == ATTEMPTED_ROUTES_EXHAUSTED_ERROR_CODE
    assert error.quota_scope == "day"
    assert error.eligible_routes_count == 1
    assert error.exhausted_routes_count == 1
    assert repository.diagnostics_calls[0]["excluded_binding_ids"] == {"attempted"}
    assert repository.skipped == [
        {
            "request_id": "req-postgres-attempted",
            "estimated_tokens": 100,
            "reason": "quota_exhausted",
        }
    ]


@pytest.mark.asyncio
async def test_in_memory_repository_rejects_proxy_route_without_proxy_url() -> None:
    route = _candidate("missing-proxy-url").model_copy(update={"proxy_url": None})
    repository = InMemoryRouteRepository([route])
    request = GatewayChatRequest(
        request_id="req-missing-proxy-url",
        source_service="test",
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "hello"}],
        estimated_input_tokens=100,
    )

    with pytest.raises(GatewayError) as exc_info:
        await repository.acquire_route(request)

    assert exc_info.value.reason == "no_route"


@pytest.mark.asyncio
async def test_in_memory_repository_reports_quota_exhausted_when_active_routes_hit_daily_limit() -> None:
    repository = InMemoryRouteRepository([_candidate("daily-quota", day_requests_used=100)])
    request = GatewayChatRequest(
        request_id="req-quota-exhausted",
        source_service="test",
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "hello"}],
        estimated_input_tokens=100,
    )

    with pytest.raises(GatewayError) as exc_info:
        await repository.acquire_route(request)

    error = exc_info.value
    assert error.reason == "quota_exhausted"
    assert error.retryable is True
    assert error.error_code == "quota_exhausted"
    assert error.quota_scope == "day"
    assert error.retry_after_seconds is not None
    assert error.quota_reset_at is not None
    assert error.eligible_routes_count == 1
    assert error.exhausted_routes_count == 1
    assert error.disabled_routes_count == 0


@pytest.mark.asyncio
async def test_in_memory_repository_reports_cooldown_active_when_routes_are_sleeping() -> None:
    cooldown_until = datetime.now(tz=UTC) + timedelta(minutes=5)
    repository = InMemoryRouteRepository([_candidate("cooling", cooldown_until=cooldown_until)])
    request = GatewayChatRequest(
        request_id="req-cooldown-active",
        source_service="test",
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "hello"}],
        estimated_input_tokens=100,
    )

    with pytest.raises(GatewayError) as exc_info:
        await repository.acquire_route(request)

    error = exc_info.value
    assert error.reason == "cooldown_active"
    assert error.error_code == "cooldown_active"
    assert error.retryable is True
    assert error.retry_after_seconds is not None
    assert error.sleep_until is not None
    assert error.eligible_routes_count == 1
    assert error.exhausted_routes_count == 0


@pytest.mark.asyncio
async def test_repository_does_not_reconcile_boolean_total_tokens() -> None:
    repository = InMemoryRouteRepository([_candidate("bool-usage")])
    request = GatewayChatRequest(
        request_id="req-bool-usage",
        source_service="test",
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "hello"}],
        estimated_input_tokens=100,
    )

    lease = await repository.acquire_route(request)

    await repository.record_success(
        lease,
        GatewayChatResponse(
            request_id="req-bool-usage",
            generation_id="gen-bool-usage",
            model="gemini-3.5-flash",
            choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            usage={"total_tokens": True},
        ),
        latency_ms=25,
    )

    [route] = await repository.list_route_candidates("gemini-3.5-flash", datetime.now(tz=UTC))
    assert route.minute_tokens_reserved == 100
    assert route.day_tokens_reserved == 100


@pytest.mark.asyncio
async def test_completion_service_uses_selected_route_proxy_and_logs_safe_wide_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    captured: dict[str, Any] = {}

    class _Client:
        async def complete(self, request: GatewayChatRequest, api_key: str, proxy_url: str) -> GatewayChatResponse:
            captured["api_key"] = api_key
            captured["proxy_url"] = proxy_url
            return GatewayChatResponse(
                request_id=request.request_id,
                generation_id="gen-1",
                model=request.model,
                choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                provider_timing={
                    "operation_type": "chat",
                    "payload_kind": "text",
                    "request_bytes": 128,
                    "response_bytes": 64,
                    "media_count": 0,
                    "image_count": 0,
                    "provider_total_ms": 25,
                    "request_prepare_ms": 1,
                    "response_headers_ms": 20,
                    "response_body_ms": 2,
                    "response_parse_ms": 1,
                    "timeout_kind": None,
                    "timeout_stage": None,
                    "headers": {"authorization": api_key},
                    "response_body": "raw provider body",
                },
                raw_response={"id": "gen-1"},
            )

    repository = InMemoryRouteRepository(
        [_candidate("a", api_key="AIza-super-secret", proxy_url="http://user:pass@127.0.0.1:8000")]
    )
    service = CompletionService(
        repository=repository,
        gemini_client=_Client(),
        service_name="gemini-gateway",
        environment="test",
    )
    caplog.set_level(logging.INFO, logger="gemini_gateway.service")

    response = await service.complete(
        GatewayChatRequest(
            request_id="req-2",
            source_service="test",
            model="gemini-3.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            chat_id=123,
            telegram_message_id=456,
        )
    )

    assert captured == {"api_key": "AIza-super-secret", "proxy_url": "http://user:pass@127.0.0.1:8000"}
    assert response.route_label == "route-a"
    assert response.route == {
        "project_label": "friend-a",
        "route_label": "route-a",
        "key_label": "key-a",
        "proxy_label": "proxy-a",
        "transport_mode": "proxy",
    }
    records = [record for record in caplog.records if getattr(record, "event", None) == "gemini_gateway_request"]
    assert len(records) == 1
    record = records[0]
    assert record.service == "gemini-gateway"
    assert record.status == "success"
    assert record.route_label == "route-a"
    assert record.transport_mode == "proxy"
    assert record.generation_id == "gen-1"
    assert record.prompt_tokens == 3
    assert record.finish_reason == "stop"
    assert record.operation_type == "chat"
    assert record.payload_kind == "text"
    assert record.request_bytes == 128
    assert record.response_bytes == 64
    assert record.media_count == 0
    assert record.image_count == 0
    assert record.provider_total_ms == 25
    assert record.request_prepare_ms == 1
    assert record.response_headers_ms == 20
    assert record.response_body_ms == 2
    assert record.response_parse_ms == 1
    assert record.timeout_kind is None
    assert record.timeout_stage is None
    record_payload = json.dumps(record.__dict__, ensure_ascii=False, default=str)
    assert "AIza-super-secret" not in record_payload
    assert "user:pass" not in record_payload
    assert "authorization" not in record_payload
    assert "raw provider body" not in record_payload


@pytest.mark.asyncio
async def test_completion_service_retries_retryable_chat_failure_on_next_route() -> None:
    class _Client:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def complete(
            self,
            request: GatewayChatRequest,
            api_key: str,
            proxy_url: str,
        ) -> GatewayChatResponse:
            self.calls.append(
                {
                    "retry_count": request.retry_count,
                    "api_key": api_key,
                    "proxy_url": proxy_url,
                }
            )
            if len(self.calls) == 1:
                raise GatewayError(
                    reason="network_timeout",
                    retryable=True,
                    request_id=request.request_id,
                    provider_called=True,
                )
            return GatewayChatResponse(
                request_id=request.request_id,
                generation_id="gen-chat-retry",
                model=request.model,
                choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            )

    first_route = _candidate("chat-first", api_key="AIza-first", proxy_url="http://127.0.0.1:8001")
    second_route = _candidate("chat-second", api_key="AIza-second", proxy_url="http://127.0.0.1:8002")
    repository = InMemoryRouteRepository([first_route, second_route])
    client = _Client()
    service = CompletionService(repository=repository, gemini_client=client, environment="test")

    response = await service.complete(
        GatewayChatRequest(
            request_id="req-chat-route-retry",
            source_service="test",
            model="gemini-3.5-flash",
            messages=[{"role": "user", "content": "hi"}],
        )
    )

    assert [call["retry_count"] for call in client.calls] == [0, 1]
    assert client.calls[0]["api_key"] == "AIza-first"
    assert client.calls[1]["api_key"] == "AIza-second"
    assert response.route_label == "route-chat-second"
    assert repository.failures[-1]["reason"] == "network_timeout"
    assert repository.successes[-1]["binding_id"] == "chat-second"


@pytest.mark.asyncio
async def test_completion_service_does_not_retry_non_retryable_chat_error() -> None:
    class _Client:
        def __init__(self) -> None:
            self.calls: list[int] = []

        async def complete(
            self,
            request: GatewayChatRequest,
            api_key: str,
            proxy_url: str,
        ) -> GatewayChatResponse:
            del api_key, proxy_url
            self.calls.append(request.retry_count)
            raise GatewayError(
                reason="content_filtered",
                retryable=False,
                request_id=request.request_id,
                provider_called=True,
            )

    repository = InMemoryRouteRepository([_candidate("chat-filtered"), _candidate("chat-unused")])
    client = _Client()
    service = CompletionService(repository=repository, gemini_client=client, environment="test")

    with pytest.raises(GatewayError) as exc_info:
        await service.complete(
            GatewayChatRequest(
                request_id="req-chat-no-retry",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            )
        )

    assert exc_info.value.reason == "content_filtered"
    assert client.calls == [0]
    assert repository.failures[-1]["binding_id"] == "chat-filtered"


@pytest.mark.asyncio
async def test_completion_service_stops_retryable_chat_failures_at_max_attempts() -> None:
    class _Client:
        def __init__(self) -> None:
            self.calls: list[int] = []

        async def complete(
            self,
            request: GatewayChatRequest,
            api_key: str,
            proxy_url: str,
        ) -> GatewayChatResponse:
            del api_key, proxy_url
            self.calls.append(request.retry_count)
            raise GatewayError(
                reason="network_timeout",
                retryable=True,
                request_id=request.request_id,
                provider_called=True,
            )

    repository = InMemoryRouteRepository(
        [_candidate("chat-one"), _candidate("chat-two"), _candidate("chat-three")]
    )
    client = _Client()
    service = CompletionService(
        repository=repository,
        gemini_client=client,
        environment="test",
        max_route_attempts=2,
    )

    with pytest.raises(GatewayError) as exc_info:
        await service.complete(
            GatewayChatRequest(
                request_id="req-chat-max-route-attempts",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            )
        )

    assert exc_info.value.reason == "network_timeout"
    assert exc_info.value.route_label == "route-chat-two"
    assert client.calls == [0, 1]
    assert [failure["binding_id"] for failure in repository.failures] == ["chat-one", "chat-two"]


@pytest.mark.asyncio
async def test_completion_service_routes_tts_through_same_quota_repository() -> None:
    captured: dict[str, Any] = {}

    class _TTSClient:
        async def synthesize(self, request: GatewayTTSRequest, api_key: str, proxy_url: str) -> GatewayTTSResponse:
            captured["api_key"] = api_key
            captured["proxy_url"] = proxy_url
            captured["model"] = request.model
            return GatewayTTSResponse(
                request_id=request.request_id,
                generation_id="tts-1",
                model=request.model,
                audio_base64="UklGRg==",
                audio_mime_type="audio/wav",
                usage={"prompt_tokens": 11, "total_tokens": 11},
            )

    tts_model = "google/gemini-3.1-flash-tts-preview"
    route = _candidate("tts", api_key="AIza-tts-key", proxy_url="http://user:pass@127.0.0.1:9000")
    route.model = tts_model
    repository = InMemoryRouteRepository([route])
    service = CompletionService(repository=repository, gemini_client=object(), tts_client=_TTSClient())

    response = await service.synthesize_speech(
        GatewayTTSRequest(
            request_id="req-tts",
            source_service="voice_tts",
            model=tts_model,
            text="коротко",
            estimated_input_tokens=40,
        )
    )

    [reserved_route] = await repository.list_route_candidates(tts_model, datetime.now(tz=UTC))
    assert captured == {
        "api_key": "AIza-tts-key",
        "proxy_url": "http://user:pass@127.0.0.1:9000",
        "model": tts_model,
    }
    assert response.route["key_label"] == "key-tts"
    assert reserved_route.minute_requests_used == 1
    assert reserved_route.minute_tokens_reserved == 11


@pytest.mark.asyncio
async def test_completion_service_retries_retryable_tts_failure_on_next_route() -> None:
    class _TTSClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def synthesize(
            self,
            request: GatewayTTSRequest,
            api_key: str,
            proxy_url: str,
        ) -> GatewayTTSResponse:
            self.calls.append(
                {
                    "retry_count": request.retry_count,
                    "api_key": api_key,
                    "proxy_url": proxy_url,
                }
            )
            if len(self.calls) == 1:
                raise GatewayError(
                    reason="provider_unavailable",
                    retryable=True,
                    request_id=request.request_id,
                    provider_called=True,
                )
            return GatewayTTSResponse(
                request_id=request.request_id,
                generation_id="tts-retry",
                model=request.model,
                audio_base64="UklGRg==",
                audio_mime_type="audio/wav",
                usage={"prompt_tokens": 7, "total_tokens": 7},
            )

    tts_model = "google/gemini-3.1-flash-tts-preview"
    first_route = _candidate("tts-first", api_key="AIza-tts-first")
    second_route = _candidate("tts-second", api_key="AIza-tts-second")
    first_route.model = tts_model
    second_route.model = tts_model
    repository = InMemoryRouteRepository([first_route, second_route])
    client = _TTSClient()
    service = CompletionService(repository=repository, gemini_client=object(), tts_client=client)

    response = await service.synthesize_speech(
        GatewayTTSRequest(
            request_id="req-tts-route-retry",
            source_service="voice_tts",
            model=tts_model,
            text="коротко",
        )
    )

    assert [call["retry_count"] for call in client.calls] == [0, 1]
    assert client.calls[0]["api_key"] == "AIza-tts-first"
    assert client.calls[1]["api_key"] == "AIza-tts-second"
    assert response.route_label == "route-tts-second"
    assert repository.failures[-1]["reason"] == "provider_unavailable"
    assert repository.successes[-1]["binding_id"] == "tts-second"


@pytest.mark.asyncio
async def test_completion_service_records_failure_applies_cooldown_and_logs_safe_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Client:
        async def complete(self, request: GatewayChatRequest, api_key: str, proxy_url: str) -> GatewayChatResponse:
            raise GatewayError(
                reason="rate_limited",
                retryable=True,
                provider_status_code=429,
                provider_message_safe=f"HTTP 429 for {api_key} via {proxy_url}",
                request_id=request.request_id,
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
                    "timeout_kind": "read_timeout",
                    "timeout_stage": "response_headers",
                    "cookies": "session=secret",
                    "raw_body": "raw failure body",
                },
            )

    repository = InMemoryRouteRepository(
        [_candidate("a", api_key="AIza-super-secret", proxy_url="http://user:pass@127.0.0.1:8000")]
    )
    service = CompletionService(
        repository=repository,
        gemini_client=_Client(),
        environment="test",
        max_route_attempts=1,
    )
    caplog.set_level(logging.WARNING, logger="gemini_gateway.service")

    with pytest.raises(GatewayError) as exc_info:
        await service.complete(
            GatewayChatRequest(
                request_id="req-3",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            )
        )

    [route] = await repository.list_route_candidates("gemini-3.5-flash", datetime.now(tz=UTC))
    assert exc_info.value.reason == "rate_limited"
    assert route.cooldown_until is not None
    assert repository.failures[-1]["reason"] == "rate_limited"
    assert exc_info.value.route_label == "route-a"
    assert exc_info.value.project_label == "friend-a"
    assert exc_info.value.key_label == "key-a"
    assert exc_info.value.proxy_label == "proxy-a"
    assert exc_info.value.transport_mode == "proxy"
    records = [record for record in caplog.records if getattr(record, "event", None) == "gemini_gateway_request"]
    assert len(records) == 1
    record = records[0]
    assert record.service == "gemini-gateway"
    assert record.status == "error"
    assert record.reason == "rate_limited"
    assert record.failed_stage == "provider_call"
    assert record.provider_status_code == 429
    assert record.retryable is True
    assert record.error_type == "rate_limited"
    assert record.error_message == "Слишком много запросов, попробуй чуть позже"
    assert record.cooldown_scope == "project_model"
    assert record.cooldown_level == 1
    assert record.sleep_until is not None
    assert record.operation_type == "chat"
    assert record.payload_kind == "media"
    assert record.request_bytes == 2048
    assert record.response_bytes == 512
    assert record.media_count == 2
    assert record.image_count == 2
    assert record.provider_total_ms == 1200
    assert record.request_prepare_ms == 2
    assert record.response_headers_ms == 900
    assert record.response_body_ms == 100
    assert record.response_parse_ms == 3
    assert record.timeout_kind == "read_timeout"
    assert record.timeout_stage == "response_headers"
    payload = json.dumps(record.__dict__, ensure_ascii=False, default=str)
    assert "AIza-super-secret" not in payload
    assert "user:pass" not in payload
    assert "HTTP 429" not in payload
    assert "session=secret" not in payload
    assert "raw failure body" not in payload


@pytest.mark.asyncio
async def test_completion_service_logs_route_acquisition_stage_when_no_route(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = CompletionService(repository=InMemoryRouteRepository([]), gemini_client=object(), environment="test")
    caplog.set_level(logging.WARNING, logger="gemini_gateway.service")

    with pytest.raises(GatewayError) as exc_info:
        await service.complete(
            GatewayChatRequest(
                request_id="req-no-route",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            )
        )

    assert exc_info.value.reason == "no_route"
    records = [record for record in caplog.records if getattr(record, "event", None) == "gemini_gateway_request"]
    assert len(records) == 1
    assert records[0].service == "gemini-gateway"
    assert records[0].status == "error"
    assert records[0].reason == "no_route"
    assert records[0].failed_stage == "route_acquisition"
    assert records[0].retryable is True


@pytest.mark.asyncio
async def test_completion_service_wraps_unexpected_provider_exception_with_route_context() -> None:
    class _Client:
        async def complete(self, request: GatewayChatRequest, api_key: str, proxy_url: str) -> GatewayChatResponse:
            del request
            raise RuntimeError(f"transport blew up for {api_key} via {proxy_url}")

    repository = InMemoryRouteRepository(
        [_candidate("a", api_key="AIza-super-secret", proxy_url="http://user:pass@127.0.0.1:8000")]
    )
    service = CompletionService(
        repository=repository,
        gemini_client=_Client(),
        environment="test",
        max_route_attempts=1,
    )

    with pytest.raises(GatewayError) as exc_info:
        await service.complete(
            GatewayChatRequest(
                request_id="req-unexpected-route-context",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            )
        )

    error = exc_info.value
    assert error.reason == "request_failed"
    assert error.route_label == "route-a"
    assert error.project_label == "friend-a"
    assert error.key_label == "key-a"
    assert error.proxy_label == "proxy-a"
    assert error.transport_mode == "proxy"
    response_payload = error.to_response().model_dump(exclude_none=True)
    assert response_payload["route_label"] == "route-a"
    serialized = json.dumps(response_payload, ensure_ascii=False, default=str)
    assert "AIza-super-secret" not in serialized
    assert "user:pass" not in serialized


@pytest.mark.asyncio
async def test_completion_service_keeps_provider_stage_for_post_response_gateway_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-filtered",
                "model": "gemini-3.5-flash",
                "choices": [{"index": 0, "finish_reason": "content_filter: PROHIBITED_CONTENT"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
            },
        )

    repository = InMemoryRouteRepository([_candidate("post-response")])
    service = CompletionService(
        repository=repository,
        gemini_client=GeminiOpenAIClient(
            base_url="https://example.test/openai",
            transport=httpx.MockTransport(handler),
        ),
        environment="test",
    )
    caplog.set_level(logging.WARNING, logger="gemini_gateway.service")

    with pytest.raises(GatewayError) as exc_info:
        await service.complete(
            GatewayChatRequest(
                request_id="req-post-response-error",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
                estimated_input_tokens=250,
            )
        )

    [route] = await repository.list_route_candidates("gemini-3.5-flash", datetime.now(tz=UTC))
    assert exc_info.value.reason == "content_filtered"
    assert exc_info.value.provider_called is True
    assert repository.failures[-1]["provider_called"] is True
    assert route.minute_tokens_reserved == 250
    assert route.day_tokens_reserved == 250
    records = [record for record in caplog.records if getattr(record, "event", None) == "gemini_gateway_request"]
    assert len(records) == 1
    assert records[0].failed_stage == "provider_call"


@pytest.mark.asyncio
async def test_completion_service_releases_reservation_when_proxy_fails_before_provider_call() -> None:
    class _Client:
        async def complete(self, request: GatewayChatRequest, api_key: str, proxy_url: str) -> GatewayChatResponse:
            raise GatewayError(
                reason="proxy_failed",
                retryable=True,
                request_id=request.request_id,
            )

    repository = InMemoryRouteRepository([_candidate("a")])
    service = CompletionService(repository=repository, gemini_client=_Client(), environment="test")

    with pytest.raises(GatewayError):
        await service.complete(
            GatewayChatRequest(
                request_id="req-proxy-failed",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
                estimated_input_tokens=250,
            )
        )

    [route] = await repository.list_route_candidates("gemini-3.5-flash", datetime.now(tz=UTC))
    assert route.minute_tokens_reserved == 0
    assert route.day_tokens_reserved == 0


@pytest.mark.asyncio
async def test_completion_service_does_not_cool_route_for_pre_provider_unavailable() -> None:
    class _Client:
        async def complete(self, request: GatewayChatRequest, api_key: str, proxy_url: str) -> GatewayChatResponse:
            raise GatewayError(
                reason="provider_unavailable",
                retryable=True,
                request_id=request.request_id,
                provider_called=False,
            )

    repository = InMemoryRouteRepository([_candidate("a")])
    service = CompletionService(repository=repository, gemini_client=_Client(), environment="test")

    with pytest.raises(GatewayError):
        await service.complete(
            GatewayChatRequest(
                request_id="req-provider-unavailable-before-call",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
                estimated_input_tokens=250,
            )
        )

    [route] = await repository.list_route_candidates("gemini-3.5-flash", datetime.now(tz=UTC))
    assert repository.failures[-1]["provider_called"] is False
    assert route.minute_tokens_reserved == 0
    assert route.day_tokens_reserved == 0
    assert route.cooldown_until is None


@pytest.mark.asyncio
async def test_completion_service_cools_down_route_when_proxy_fails() -> None:
    class _Client:
        async def complete(self, request: GatewayChatRequest, api_key: str, proxy_url: str) -> GatewayChatResponse:
            raise GatewayError(
                reason="proxy_failed",
                retryable=True,
                request_id=request.request_id,
            )

    repository = InMemoryRouteRepository([_candidate("a")])
    service = CompletionService(repository=repository, gemini_client=_Client(), environment="test")

    with pytest.raises(GatewayError):
        await service.complete(
            GatewayChatRequest(
                request_id="req-proxy-cooldown",
                source_service="test",
                model="gemini-3.5-flash",
                messages=[{"role": "user", "content": "hi"}],
            )
        )

    [route] = await repository.list_route_candidates("gemini-3.5-flash", datetime.now(tz=UTC))
    assert route.cooldown_until is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("route_failure_reason", ["quota_exhausted", "cooldown_active", "no_route"])
async def test_completion_service_uses_openrouter_embedding_fallback_for_route_acquisition_failures(
    route_failure_reason: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    repository = _RouteFailureRepository(route_failure_reason)
    gemini_client = _RecordingGeminiEmbeddingClient()
    openrouter_client = _RecordingOpenRouterEmbeddingClient()
    service = CompletionService(
        repository=repository,
        gemini_client=object(),
        embedding_client=gemini_client,
        openrouter_embedding_client=openrouter_client,
        openrouter_api_key=_OPENROUTER_API_KEY,
        openrouter_embeddings_fallback_enabled=True,
        environment="test",
    )
    request = _embedding_request(request_id=f"req-fallback-{route_failure_reason}")
    caplog.set_level(logging.INFO, logger="gemini_gateway.service")

    response = await service.embed(request)

    assert gemini_client.called is False
    assert openrouter_client.calls == [{"request": request, "api_key": _OPENROUTER_API_KEY}]
    assert len(repository.failures) == 1
    assert repository.failures[0]["lease"] is None
    assert repository.failures[0]["provider_called"] is False
    assert repository.failures[0]["error"].reason == route_failure_reason
    assert response.route.proxy_label is None
    assert response.route["proxy_label"] is None
    assert response.route.model_dump(mode="json", exclude_none=True) == {
        "project_label": "openrouter-fallback",
        "route_label": "openrouter-embedding-fallback",
        "key_label": "openrouter-api-key",
        "transport_mode": "direct",
    }
    assert response.model_dump(mode="json", exclude_none=True)["route"] == {
        "project_label": "openrouter-fallback",
        "route_label": "openrouter-embedding-fallback",
        "key_label": "openrouter-api-key",
        "transport_mode": "direct",
    }
    assert response.project_label == "openrouter-fallback"
    assert response.route_label == "openrouter-embedding-fallback"
    assert response.key_label == "openrouter-api-key"
    assert response.proxy_label is None
    assert response.transport_mode == "direct"

    route_acquisition_records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "gemini_gateway_request"
        and getattr(record, "status", None) == "error"
        and getattr(record, "fallback_provider", None) is None
    ]
    assert len(route_acquisition_records) == 1
    route_acquisition_record = route_acquisition_records[0]
    assert route_acquisition_record.reason == route_failure_reason
    assert route_acquisition_record.failed_stage == "route_acquisition"
    assert route_acquisition_record.route_label is None
    assert route_acquisition_record.transport_mode is None

    fallback_records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "gemini_gateway_request"
        and getattr(record, "fallback_provider", None) == "openrouter"
    ]
    assert len(fallback_records) == 1
    fallback_record = fallback_records[0]
    assert fallback_record.status == "success"
    assert fallback_record.project_label == "openrouter-fallback"
    assert fallback_record.route_label == "openrouter-embedding-fallback"
    assert fallback_record.key_label == "openrouter-api-key"
    assert fallback_record.proxy_label is None
    assert fallback_record.transport_mode == "direct"


@pytest.mark.asyncio
async def test_completion_service_does_not_acquire_route_or_fallback_without_gemini_embedding_client() -> None:
    repository = _AcquireRecordingRepository()
    openrouter_client = _RecordingOpenRouterEmbeddingClient()
    service = CompletionService(
        repository=repository,
        gemini_client=object(),
        embedding_client=None,
        openrouter_embedding_client=openrouter_client,
        openrouter_api_key=_OPENROUTER_API_KEY,
        openrouter_embeddings_fallback_enabled=True,
        environment="test",
    )

    with pytest.raises(GatewayError) as exc_info:
        await service.embed(_embedding_request(request_id="req-no-gemini-embedding-client"))

    assert exc_info.value.reason == "provider_unavailable"
    assert repository.acquire_called is False
    assert repository.failures == []
    assert openrouter_client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fallback_enabled", "openrouter_api_key", "openrouter_client"),
    [
        (False, _OPENROUTER_API_KEY, _RecordingOpenRouterEmbeddingClient()),
        (True, "   ", _RecordingOpenRouterEmbeddingClient()),
        (True, _OPENROUTER_API_KEY, None),
    ],
)
async def test_completion_service_does_not_use_openrouter_embedding_fallback_when_not_configured(
    fallback_enabled: bool,
    openrouter_api_key: str,
    openrouter_client: _RecordingOpenRouterEmbeddingClient | None,
) -> None:
    repository = _RouteFailureRepository("no_route")
    service = CompletionService(
        repository=repository,
        gemini_client=object(),
        embedding_client=_RecordingGeminiEmbeddingClient(),
        openrouter_embedding_client=openrouter_client,
        openrouter_api_key=openrouter_api_key,
        openrouter_embeddings_fallback_enabled=fallback_enabled,
        environment="test",
    )

    with pytest.raises(GatewayError) as exc_info:
        await service.embed(_embedding_request(request_id="req-openrouter-not-configured"))

    assert exc_info.value.reason == "no_route"
    if openrouter_client is not None:
        assert openrouter_client.calls == []


@pytest.mark.asyncio
async def test_completion_service_does_not_use_openrouter_embedding_fallback_for_other_models() -> None:
    repository = _RouteFailureRepository("no_route")
    openrouter_client = _RecordingOpenRouterEmbeddingClient()
    service = CompletionService(
        repository=repository,
        gemini_client=object(),
        embedding_client=_RecordingGeminiEmbeddingClient(),
        openrouter_embedding_client=openrouter_client,
        openrouter_api_key=_OPENROUTER_API_KEY,
        openrouter_embeddings_fallback_enabled=True,
        environment="test",
    )

    with pytest.raises(GatewayError) as exc_info:
        await service.embed(_embedding_request(request_id="req-other-embedding-model", model="text-embedding-004"))

    assert exc_info.value.reason == "no_route"
    assert openrouter_client.calls == []


@pytest.mark.asyncio
async def test_completion_service_uses_openrouter_embedding_fallback_after_gemini_routes_exhausted() -> None:
    class _FailingGeminiEmbeddingClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def embed(
            self,
            request: GatewayEmbeddingRequest,
            api_key: str,
            proxy_url: str,
        ) -> GatewayEmbeddingResponse:
            self.calls.append(
                {
                    "retry_count": request.retry_count,
                    "api_key": api_key,
                    "proxy_url": proxy_url,
                }
            )
            raise GatewayError(
                reason="network_timeout",
                retryable=True,
                request_id=request.request_id,
                provider_called=True,
            )

    first_route = _candidate(
        "emb-first",
        api_key="AIza-emb-first",
        proxy_url="http://127.0.0.1:9101",
    )
    second_route = _candidate(
        "emb-second",
        api_key="AIza-emb-second",
        proxy_url="http://127.0.0.1:9102",
    )
    first_route.model = _OPENROUTER_EMBEDDING_MODEL
    second_route.model = _OPENROUTER_EMBEDDING_MODEL
    repository = InMemoryRouteRepository([first_route, second_route])
    gemini_client = _FailingGeminiEmbeddingClient()
    openrouter_client = _RecordingOpenRouterEmbeddingClient()
    service = CompletionService(
        repository=repository,
        gemini_client=object(),
        embedding_client=gemini_client,
        openrouter_embedding_client=openrouter_client,
        openrouter_api_key=_OPENROUTER_API_KEY,
        openrouter_embeddings_fallback_enabled=True,
        environment="test",
    )
    request = _embedding_request(request_id="req-gemini-exhausted-openrouter")

    response = await service.embed(request)

    assert [call["retry_count"] for call in gemini_client.calls] == [0, 1]
    assert gemini_client.calls[0]["api_key"] == "AIza-emb-first"
    assert gemini_client.calls[1]["api_key"] == "AIza-emb-second"
    assert openrouter_client.calls == [{"request": request, "api_key": _OPENROUTER_API_KEY}]
    assert [failure["binding_id"] for failure in repository.failures[:2]] == ["emb-first", "emb-second"]
    assert repository.failures[-1]["binding_id"] is None
    assert repository.failures[-1]["reason"] == "no_route"
    assert response.route_label == "openrouter-embedding-fallback"
    assert response.transport_mode == "direct"


@pytest.mark.asyncio
async def test_completion_service_uses_openrouter_embedding_fallback_after_attempted_routes_exhausted() -> None:
    class _AttemptedRoutesExhaustedRepository:
        def __init__(self) -> None:
            self.failures: list[dict[str, Any]] = []

        async def acquire_route(self, request: GatewayEmbeddingRequest) -> None:
            raise GatewayError(
                reason="no_route",
                retryable=True,
                request_id=request.request_id,
                error_code=ATTEMPTED_ROUTES_EXHAUSTED_ERROR_CODE,
            )

        async def record_success(self, lease: Any, response: Any, latency_ms: int) -> None:
            del lease, response, latency_ms

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
                    "error": error,
                    "latency_ms": latency_ms,
                    "provider_called": provider_called,
                }
            )

    repository = _AttemptedRoutesExhaustedRepository()
    openrouter_client = _RecordingOpenRouterEmbeddingClient()
    service = CompletionService(
        repository=repository,
        gemini_client=object(),
        embedding_client=_RecordingGeminiEmbeddingClient(),
        openrouter_embedding_client=openrouter_client,
        openrouter_api_key=_OPENROUTER_API_KEY,
        openrouter_embeddings_fallback_enabled=True,
        environment="test",
    )
    request = _embedding_request(request_id="req-attempted-routes-openrouter")

    response = await service.embed(request)

    assert openrouter_client.calls == [{"request": request, "api_key": _OPENROUTER_API_KEY}]
    assert repository.failures[0]["lease"] is None
    assert repository.failures[0]["error"].error_code == ATTEMPTED_ROUTES_EXHAUSTED_ERROR_CODE
    assert response.route_label == "openrouter-embedding-fallback"


@pytest.mark.asyncio
async def test_completion_service_attaches_direct_metadata_when_openrouter_embedding_fallback_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FailingOpenRouterEmbeddingClient:
        async def embed(
            self,
            *,
            request: GatewayEmbeddingRequest,
            api_key: str,
        ) -> GatewayEmbeddingResponse:
            del api_key
            raise GatewayError(
                reason="provider_unavailable",
                retryable=True,
                request_id=request.request_id,
                provider_called=True,
            )

    repository = _RouteFailureRepository("no_route")
    service = CompletionService(
        repository=repository,
        gemini_client=object(),
        embedding_client=_RecordingGeminiEmbeddingClient(),
        openrouter_embedding_client=_FailingOpenRouterEmbeddingClient(),
        openrouter_api_key=_OPENROUTER_API_KEY,
        openrouter_embeddings_fallback_enabled=True,
        environment="test",
    )
    caplog.set_level(logging.INFO, logger="gemini_gateway.service")

    with pytest.raises(GatewayError) as exc_info:
        await service.embed(_embedding_request(request_id="req-openrouter-fallback-fails"))

    error = exc_info.value
    assert error.reason == "provider_unavailable"
    assert error.project_label == "openrouter-fallback"
    assert error.route_label == "openrouter-embedding-fallback"
    assert error.key_label == "openrouter-api-key"
    assert error.proxy_label is None
    assert error.transport_mode == "direct"

    fallback_records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "gemini_gateway_request"
        and getattr(record, "fallback_provider", None) == "openrouter"
    ]
    assert len(fallback_records) == 1
    fallback_record = fallback_records[0]
    assert fallback_record.status == "error"
    assert fallback_record.failed_stage == "openrouter_embedding_fallback"
    assert fallback_record.project_label == "openrouter-fallback"
    assert fallback_record.route_label == "openrouter-embedding-fallback"
    assert fallback_record.key_label == "openrouter-api-key"
    assert fallback_record.proxy_label is None
    assert fallback_record.transport_mode == "direct"
