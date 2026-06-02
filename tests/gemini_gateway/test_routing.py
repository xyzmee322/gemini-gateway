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
    GatewayTTSRequest,
    GatewayTTSResponse,
    RouteCandidate,
)
from gemini_gateway.errors import GatewayError
from gemini_gateway.gemini_client import GeminiOpenAIClient
from gemini_gateway.repository import InMemoryRouteRepository, RouteScorer
from gemini_gateway.repository import _safe_provider_response_json
from gemini_gateway.service import CompletionService


def _candidate(
    binding_id: str,
    *,
    api_key: str = "AIza-key",
    proxy_url: str = "http://user:pass@127.0.0.1:8000",
    minute_tokens_reserved: int = 0,
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
        day_requests_used=0,
        cooldown_until=cooldown_until,
        half_open=half_open,
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
    response = GatewayTTSResponse(
        request_id="req-tts",
        model="google/gemini-tts",
        audio_base64=audio_base64,
        audio_mime_type="audio/wav",
        raw_response={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/wav",
                                    "data": audio_base64,
                                }
                            }
                        ]
                    }
                }
            ]
        },
    )

    payload = _safe_provider_response_json(response)
    serialized = json.dumps(payload, ensure_ascii=False)

    assert audio_base64 not in serialized
    assert payload["audio_base64"].startswith("<audio:")
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
    record_payload = json.dumps(record.__dict__, ensure_ascii=False, default=str)
    assert "AIza-super-secret" not in record_payload
    assert "user:pass" not in record_payload


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
            )

    repository = InMemoryRouteRepository(
        [_candidate("a", api_key="AIza-super-secret", proxy_url="http://user:pass@127.0.0.1:8000")]
    )
    service = CompletionService(repository=repository, gemini_client=_Client(), environment="test")
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
    payload = json.dumps(record.__dict__, ensure_ascii=False, default=str)
    assert "AIza-super-secret" not in payload
    assert "user:pass" not in payload
    assert "HTTP 429" not in payload


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
    service = CompletionService(repository=repository, gemini_client=_Client(), environment="test")

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
