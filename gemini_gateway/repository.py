from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from typing import Any, Protocol

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.audit_sanitizer import sanitize_payload_for_audit
from core.number_parsing import parse_optional_int as _safe_int
from gemini_gateway.contracts import (
    GatewayChatRequest,
    GatewayChatResponse,
    GatewayRouteRequest,
    RouteCandidate,
    RouteLease,
    SeedBinding,
    SeedConfig,
    SeedProxy,
)
from gemini_gateway.cooldown import CooldownDecision, CooldownPolicy
from gemini_gateway.errors import GatewayError, public_message_for_reason, public_provider_reason
from gemini_gateway.security import (
    ProxySecretParts,
    SecretVault,
    SecretDecryptionError,
    build_proxy_url,
    normalize_api_key,
    proxy_fingerprint_source,
)

_HALF_OPEN_TOKEN_LIMIT = 4096
_LEASE_FAILURE_COOLDOWN_REASONS = frozenset({"proxy_failed", "network_timeout"})
_PROJECT_MODEL_COOLDOWN_REASONS = frozenset({"rate_limited", "quota_exhausted"})
_MODEL_SCOPED_COOLDOWN_REASONS = _PROJECT_MODEL_COOLDOWN_REASONS
_TRANSPORT_COOLDOWN_REASONS = frozenset({"proxy_failed", "network_timeout"})
_MODEL_SCOPED_TRANSPORT_COOLDOWN_REASONS = frozenset({"network_timeout"})


@dataclass(frozen=True)
class _RouteSecrets:
    api_key: str
    proxy_url: str | None


class _RouteEligibilityQueryBuilder:
    """Собирает единые SQL-условия доступности маршрута Gemini Gateway."""

    _BASE_CONDITIONS: tuple[str, ...] = (
        "p.status = 'active'",
        "k.status = 'active'",
        "b.status = 'active'",
        "ml.status = 'active'",
        "(k.sleep_until IS NULL OR k.sleep_until <= :now)",
        "(b.sleep_until IS NULL OR b.sleep_until <= :now)",
    )
    _TRANSPORT_CONDITION = """
        (
            b.transport_mode = 'proxy'
            AND px.id IS NOT NULL
            AND px.status = 'active'
            AND (px.sleep_until IS NULL OR px.sleep_until <= :now)
        )
    """
    _COOLDOWN_CONDITION = """
        NOT EXISTS (
            SELECT 1
            FROM gemini_gateway.cooldowns cd
            WHERE cd.status = 'active'
              AND cd.sleep_until > :now
              AND (
                (cd.scope = 'project_model' AND cd.scope_key = p.id::text AND cd.model = ml.model)
                OR (cd.scope = 'key' AND cd.scope_key = k.id::text)
                OR (
                    cd.scope = 'proxy'
                    AND cd.scope_key = px.id::text
                    AND (cd.model IS NULL OR cd.model = ml.model)
                )
                OR (
                    cd.scope = 'binding'
                    AND cd.scope_key = b.id::text
                    AND (cd.model IS NULL OR cd.model = ml.model)
                )
                OR (cd.scope = 'provider_model' AND cd.scope_key = ml.model)
              )
        )
    """

    def where_clause(self, *, extra_conditions: list[str] | None = None) -> str:
        conditions = [
            *(extra_conditions or []),
            *self._BASE_CONDITIONS,
            self._TRANSPORT_CONDITION,
            self._COOLDOWN_CONDITION,
        ]
        return "WHERE\n                " + "\n                AND ".join(condition.strip() for condition in conditions)


_ROUTE_ELIGIBILITY_SQL = _RouteEligibilityQueryBuilder()
_LOCKED_ROUTE_EXTRA_CONDITIONS = [
    "b.id = :binding_id",
    "k.id = :api_key_id",
    "b.proxy_id IS NOT DISTINCT FROM CAST(:proxy_id AS bigint)",
    "p.id = :project_id",
]
_ROUTE_SLEEP_STATE_TABLES = ("key_proxy_bindings", "api_keys", "proxy_endpoints")


class RouteRepository(Protocol):
    async def acquire_route(self, request: GatewayRouteRequest) -> RouteLease:
        ...

    async def record_success(self, lease: RouteLease, response: Any, latency_ms: int) -> None:
        ...

    async def record_failure(
        self,
        lease: RouteLease | None,
        error: GatewayError,
        latency_ms: int,
        provider_called: bool,
    ) -> None:
        ...

    async def expire_elapsed_cooldowns(self, *, limit: int) -> int:
        ...

    async def expire_elapsed_route_sleeps(self, *, limit: int) -> int:
        ...

    async def complete_stale_leases(self, *, stale_after_seconds: int, limit: int) -> int:
        ...


class RouteScorer:
    """Выбирает наименее нагруженный доступный маршрут."""

    @staticmethod
    def choose(candidates: list[RouteCandidate], estimated_tokens: int, now: datetime) -> RouteCandidate | None:
        eligible = [
            candidate
            for candidate in candidates
            if RouteScorer._is_eligible(candidate=candidate, estimated_tokens=estimated_tokens, now=now)
        ]
        if not eligible:
            return None

        def score(candidate: RouteCandidate) -> tuple[float, float, int]:
            request_ratio = candidate.minute_requests_used / max(candidate.requests_per_minute, 1)
            token_ratio = candidate.minute_tokens_reserved / max(candidate.tokens_per_minute, 1)
            day_ratio = candidate.day_requests_used / max(candidate.requests_per_day, 1)
            last_used = candidate.last_used_at.timestamp() if candidate.last_used_at else 0.0
            return (max(request_ratio, token_ratio, day_ratio), last_used, -candidate.priority)

        return min(eligible, key=score)

    @staticmethod
    def _is_eligible(candidate: RouteCandidate, estimated_tokens: int, now: datetime) -> bool:
        if candidate.transport_mode != "proxy":
            return False
        cooldown_until = _aware_or_none(candidate.cooldown_until)
        if cooldown_until and cooldown_until > now:
            return False
        if candidate.half_open and estimated_tokens > _HALF_OPEN_TOKEN_LIMIT:
            return False
        if candidate.minute_requests_used >= candidate.requests_per_minute:
            return False
        if candidate.day_requests_used >= candidate.requests_per_day:
            return False
        if estimated_tokens > max(candidate.tokens_per_minute - candidate.minute_tokens_reserved, 0):
            return False
        if candidate.tokens_per_day is not None:
            return estimated_tokens <= max(candidate.tokens_per_day - candidate.day_tokens_reserved, 0)
        return True


class InMemoryRouteRepository:
    """In-memory репозиторий для быстрых unit-тестов."""

    def __init__(self, routes: list[RouteCandidate] | None = None) -> None:
        self._routes: dict[int | str, RouteCandidate] = {route.binding_id: route for route in routes or []}
        self._lock = asyncio.Lock()
        self._failure_streaks: dict[tuple[int | str, str], int] = {}
        self._attempted_bindings_by_request: dict[str, set[int | str]] = {}
        self.successes: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []

    async def acquire_route(self, request: GatewayRouteRequest) -> RouteLease:
        estimated_tokens = estimate_request_tokens(request)
        now = datetime.now(tz=UTC)

        async with self._lock:
            attempted_bindings = self._attempted_bindings_by_request.setdefault(_request_group_id(request), set())
            candidates = [
                route
                for route in self._routes.values()
                if (
                    route.model == request.model
                    and route.transport_mode == "proxy"
                    and route.proxy_url
                    and route.binding_id not in attempted_bindings
                )
            ]
            route = RouteScorer.choose(candidates, estimated_tokens, now)
            if route is None:
                raise GatewayError(
                    reason="no_route",
                    request_id=request.request_id,
                    retryable=True,
                    retry_after_seconds=_next_in_memory_retry_after_seconds(candidates, now=now),
                )

            route.minute_requests_used += 1
            route.minute_tokens_reserved += estimated_tokens
            route.day_requests_used += 1
            route.day_tokens_reserved += estimated_tokens
            route.last_used_at = now
            route.half_open = False
            attempted_bindings.add(route.binding_id)
            return _lease_from_candidate(route, estimated_tokens=estimated_tokens, attempt_id=str(uuid.uuid4()), now=now)

    async def record_success(self, lease: RouteLease, response: Any, latency_ms: int) -> None:
        usage = response.usage or {}
        actual_total_tokens = _safe_int(usage.get("total_tokens"))
        async with self._lock:
            route = self._routes.get(lease.binding_id)
            if route and actual_total_tokens is not None:
                route.minute_tokens_reserved = max(
                    0,
                    route.minute_tokens_reserved - lease.estimated_tokens + actual_total_tokens,
                )
                route.day_tokens_reserved = max(
                    0,
                    route.day_tokens_reserved - lease.estimated_tokens + actual_total_tokens,
                )
            if route:
                self._failure_streaks = {
                    key: value for key, value in self._failure_streaks.items() if key[0] != route.binding_id
                }
                route.cooldown_until = None
                route.half_open = False
            self.successes.append(
                {
                    "attempt_id": lease.attempt_id,
                    "binding_id": lease.binding_id,
                    "latency_ms": latency_ms,
                    "total_tokens": actual_total_tokens,
                }
            )

    async def record_failure(
        self,
        lease: RouteLease | None,
        error: GatewayError,
        latency_ms: int,
        provider_called: bool,
    ) -> None:
        route_failure = _should_apply_route_failure_state(error=error, provider_called=provider_called)
        async with self._lock:
            if lease is not None:
                route = self._routes.get(lease.binding_id)
                if route and not provider_called:
                    route.minute_tokens_reserved = max(0, route.minute_tokens_reserved - lease.estimated_tokens)
                    route.day_tokens_reserved = max(0, route.day_tokens_reserved - lease.estimated_tokens)
                if route and route_failure:
                    streak_key = (route.binding_id, _cooldown_reason_family(error.reason))
                    streak = self._failure_streaks.get(streak_key, 0)
                    decision = CooldownPolicy().next_decision(
                        reason=error.reason,
                        current_level=streak,
                        retry_after_seconds=error.retry_after_seconds,
                    )
                    if decision is not None:
                        _attach_cooldown_metadata(error=error, decision=decision, scopes=decision.scopes)
                        self._failure_streaks[streak_key] = decision.level
                        route.cooldown_until = decision.sleep_until
                        route.half_open = not decision.disable_needs_review
            self.failures.append(
                {
                    "attempt_id": lease.attempt_id if lease else None,
                    "binding_id": lease.binding_id if lease else None,
                    "latency_ms": latency_ms,
                    "reason": error.reason,
                    "provider_called": provider_called,
                }
            )

    async def complete_stale_leases(self, *, stale_after_seconds: int, limit: int) -> int:
        del stale_after_seconds, limit
        return 0

    async def expire_elapsed_cooldowns(self, *, limit: int) -> int:
        del limit
        return 0

    async def expire_elapsed_route_sleeps(self, *, limit: int) -> int:
        del limit
        return 0

    async def list_route_candidates(self, model: str, now: datetime) -> list[RouteCandidate]:
        del now
        async with self._lock:
            return [route.model_copy(deep=True) for route in self._routes.values() if route.model == model]

    async def health_check(self, *, require_routes: bool = False) -> dict[str, Any]:
        del require_routes
        async with self._lock:
            return {"ok": True, "checks": {"database": True, "schema": True, "routes": bool(self._routes)}}


class PostgresGatewayRepository:
    """БД-репозиторий маршрутов с атомарным резервированием квот."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        secret_vault: SecretVault,
        cooldown_policy: CooldownPolicy | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._secret_vault = secret_vault
        self._cooldown_policy = cooldown_policy or CooldownPolicy()

    async def acquire_route(self, request: GatewayRouteRequest) -> RouteLease:
        estimated_tokens = estimate_request_tokens(request)
        now = datetime.now(tz=UTC)
        candidates = await self.list_route_candidates(request.model, now)
        attempted_binding_ids = await self._attempted_binding_ids(request=request)
        if attempted_binding_ids:
            candidates = [
                candidate
                for candidate in candidates
                if str(candidate.binding_id) not in attempted_binding_ids
            ]

        while candidates:
            candidate = RouteScorer.choose(candidates, estimated_tokens, now)
            if candidate is None:
                break
            lease = await self._try_reserve_candidate(
                candidate=candidate,
                request=request,
                estimated_tokens=estimated_tokens,
                now=now,
            )
            if lease is not None:
                return lease
            candidates = [item for item in candidates if item.binding_id != candidate.binding_id]

        retry_after_seconds = await self._next_route_retry_after_seconds(
            model=request.model,
            estimated_tokens=estimated_tokens,
            now=now,
        )
        await self._record_skipped_no_route(request=request, estimated_tokens=estimated_tokens)
        raise GatewayError(
            reason="no_route",
            request_id=request.request_id,
            retryable=True,
            retry_after_seconds=retry_after_seconds,
        )

    async def _next_route_retry_after_seconds(self, *, model: str, estimated_tokens: int, now: datetime) -> int | None:
        query = text(
            """
            WITH route_base AS (
                SELECT
                    b.id AS binding_id,
                    p.id AS project_id,
                    k.id AS api_key_id,
                    px.id AS proxy_id,
                    ml.model AS model,
                    ml.requests_per_minute,
                    ml.tokens_per_minute,
                    ml.requests_per_day,
                    ml.tokens_per_day,
                    COALESCE(qm.requests_used, 0) AS minute_requests_used,
                    COALESCE(qm.input_tokens_reserved, 0) AS minute_tokens_reserved,
                    COALESCE(qd.requests_used, 0) AS day_requests_used,
                    COALESCE(qd.input_tokens_reserved, 0) AS day_tokens_reserved,
                    k.sleep_until AS key_sleep_until,
                    b.sleep_until AS binding_sleep_until,
                    px.sleep_until AS proxy_sleep_until,
                    active_cooldowns.cooldown_until
                FROM gemini_gateway.key_proxy_bindings b
                JOIN gemini_gateway.api_keys k ON k.id = b.api_key_id
                JOIN gemini_gateway.proxy_endpoints px ON px.id = b.proxy_id
                JOIN gemini_gateway.google_projects p ON p.id = k.project_id
                JOIN gemini_gateway.model_limits ml ON ml.project_id = p.id AND ml.model = :model
                LEFT JOIN gemini_gateway.quota_windows qm
                    ON qm.project_id = p.id
                    AND qm.model = ml.model
                    AND qm.window_kind = 'minute'
                    AND qm.window_start = date_trunc('minute', CAST(:now AS timestamptz))
                LEFT JOIN gemini_gateway.quota_windows qd
                    ON qd.project_id = p.id
                    AND qd.model = ml.model
                    AND qd.window_kind = 'day'
                    AND qd.window_start = date_trunc('day', CAST(:now AS timestamptz))
                LEFT JOIN LATERAL (
                    SELECT MAX(cd.sleep_until) AS cooldown_until
                    FROM gemini_gateway.cooldowns cd
                    WHERE cd.status = 'active'
                      AND cd.sleep_until > :now
                      AND (
                        (cd.scope = 'project_model' AND cd.scope_key = p.id::text AND cd.model = ml.model)
                        OR (cd.scope = 'key' AND cd.scope_key = k.id::text)
                        OR (
                            cd.scope = 'proxy'
                            AND cd.scope_key = px.id::text
                            AND (cd.model IS NULL OR cd.model = ml.model)
                        )
                        OR (
                            cd.scope = 'binding'
                            AND cd.scope_key = b.id::text
                            AND (cd.model IS NULL OR cd.model = ml.model)
                        )
                        OR (cd.scope = 'provider_model' AND cd.scope_key = ml.model)
                      )
                ) active_cooldowns ON TRUE
                WHERE p.status = 'active'
                  AND k.status = 'active'
                  AND b.status = 'active'
                  AND px.status = 'active'
                  AND ml.status = 'active'
                  AND b.transport_mode = 'proxy'
            ),
            route_retry_after AS (
                SELECT
                    rb.binding_id,
                    (
                        :estimated_tokens > rb.tokens_per_minute
                        OR (
                            rb.tokens_per_day IS NOT NULL
                            AND :estimated_tokens > rb.tokens_per_day
                        )
                    ) AS permanently_too_large,
                    (
                        SELECT MAX(unblock_at)
                        FROM (
                            VALUES
                                (CASE WHEN rb.key_sleep_until > :now THEN rb.key_sleep_until END),
                                (CASE WHEN rb.binding_sleep_until > :now THEN rb.binding_sleep_until END),
                                (CASE WHEN rb.proxy_sleep_until > :now THEN rb.proxy_sleep_until END),
                                (rb.cooldown_until),
                                (
                                    CASE
                                        WHEN rb.minute_requests_used >= rb.requests_per_minute
                                            THEN date_trunc('minute', CAST(:now AS timestamptz)) + interval '1 minute'
                                    END
                                ),
                                (
                                    CASE
                                        WHEN :estimated_tokens <= rb.tokens_per_minute
                                             AND :estimated_tokens > GREATEST(rb.tokens_per_minute - rb.minute_tokens_reserved, 0)
                                            THEN date_trunc('minute', CAST(:now AS timestamptz)) + interval '1 minute'
                                    END
                                ),
                                (
                                    CASE
                                        WHEN rb.day_requests_used >= rb.requests_per_day
                                            THEN date_trunc('day', CAST(:now AS timestamptz)) + interval '1 day'
                                    END
                                ),
                                (
                                    CASE
                                        WHEN rb.tokens_per_day IS NOT NULL
                                             AND :estimated_tokens <= rb.tokens_per_day
                                             AND :estimated_tokens > GREATEST(rb.tokens_per_day - rb.day_tokens_reserved, 0)
                                            THEN date_trunc('day', CAST(:now AS timestamptz)) + interval '1 day'
                                    END
                                )
                        ) AS route_blocks(unblock_at)
                        WHERE unblock_at IS NOT NULL
                    ) AS retry_after_at
                FROM route_base rb
            )
            SELECT MIN(retry_after_at)
            FROM route_retry_after
            WHERE permanently_too_large = FALSE
              AND retry_after_at IS NOT NULL
              AND NOT EXISTS (
                SELECT 1
                FROM route_retry_after
                WHERE permanently_too_large = FALSE
                  AND retry_after_at IS NULL
              )
            """
        )
        async with self._session_factory() as session:
            sleep_until = await session.scalar(
                query,
                {"model": model, "estimated_tokens": estimated_tokens, "now": now},
            )
        return _retry_after_seconds_until(sleep_until, now=now)

    async def list_route_candidates(self, model: str, now: datetime) -> list[RouteCandidate]:
        query = text(
            f"""
            SELECT
                b.id AS binding_id,
                p.id AS project_id,
                k.id AS api_key_id,
                px.id AS proxy_id,
                p.label AS project_label,
                k.label AS key_label,
                px.label AS proxy_label,
                b.label AS route_label,
                b.transport_mode,
                ml.model,
                ml.requests_per_minute,
                ml.tokens_per_minute,
                ml.requests_per_day,
                ml.tokens_per_day,
                COALESCE(qm.requests_used, 0) AS minute_requests_used,
                COALESCE(qm.input_tokens_reserved, 0) AS minute_tokens_reserved,
                COALESCE(qd.requests_used, 0) AS day_requests_used,
                COALESCE(qd.input_tokens_reserved, 0) AS day_tokens_reserved,
                GREATEST(k.sleep_until, px.sleep_until, b.sleep_until) AS cooldown_until,
                b.last_success_at AS last_used_at
            FROM gemini_gateway.key_proxy_bindings b
            JOIN gemini_gateway.api_keys k ON k.id = b.api_key_id
            LEFT JOIN gemini_gateway.proxy_endpoints px ON px.id = b.proxy_id
            JOIN gemini_gateway.google_projects p ON p.id = k.project_id
            JOIN gemini_gateway.model_limits ml ON ml.project_id = p.id AND ml.model = :model
            LEFT JOIN gemini_gateway.quota_windows qm
                ON qm.project_id = p.id
                AND qm.model = ml.model
                AND qm.window_kind = 'minute'
                AND qm.window_start = :minute_start
            LEFT JOIN gemini_gateway.quota_windows qd
                ON qd.project_id = p.id
                AND qd.model = ml.model
                AND qd.window_kind = 'day'
                AND qd.window_start = :day_start
            {_ROUTE_ELIGIBILITY_SQL.where_clause()}
            """
        )
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    query,
                    {
                        "model": model,
                        "minute_start": _minute_start(now),
                        "day_start": _day_start(now),
                        "now": now,
                    },
                )
            ).mappings()
            candidates: list[RouteCandidate] = []
            for row in rows:
                candidates.append(
                    RouteCandidate(
                        binding_id=row["binding_id"],
                        project_id=row["project_id"],
                        api_key_id=row["api_key_id"],
                        proxy_id=row["proxy_id"],
                        api_key="",
                        proxy_url=None,
                        model=row["model"],
                        route_label=row["route_label"],
                        project_label=row["project_label"],
                        key_label=row["key_label"],
                        proxy_label=row["proxy_label"],
                        transport_mode=row["transport_mode"],
                        requests_per_minute=row["requests_per_minute"],
                        tokens_per_minute=row["tokens_per_minute"],
                        requests_per_day=row["requests_per_day"],
                        tokens_per_day=row["tokens_per_day"],
                        minute_requests_used=row["minute_requests_used"],
                        minute_tokens_reserved=row["minute_tokens_reserved"],
                        day_requests_used=row["day_requests_used"],
                        day_tokens_reserved=row["day_tokens_reserved"],
                        cooldown_until=row["cooldown_until"],
                        last_used_at=row["last_used_at"],
                    )
                )
            return candidates

    async def record_success(self, lease: RouteLease, response: GatewayChatResponse, latency_ms: int) -> None:
        usage = response.usage or {}
        prompt_tokens = _safe_int(usage.get("prompt_tokens"))
        completion_tokens = _safe_int(usage.get("completion_tokens"))
        total_tokens = _safe_int(usage.get("total_tokens"))
        minute_start = _minute_start(lease.leased_at)
        day_start = _day_start(lease.leased_at)
        async with self._session_factory() as session, session.begin():
            await session.execute(
                text(
                    """
                    UPDATE gemini_gateway.route_attempts
                    SET status = 'success',
                        latency_ms = :latency_ms,
                        prompt_tokens = :prompt_tokens,
                        completion_tokens = :completion_tokens,
                        total_tokens = :total_tokens,
                        generation_id = :generation_id,
                        finish_reason = :finish_reason,
                        provider_response_json = CAST(:provider_response_json AS jsonb),
                        updated_at = now()
                    WHERE id = :attempt_id
                    """
                ),
                {
                    "attempt_id": lease.attempt_id,
                    "latency_ms": latency_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "generation_id": response.generation_id,
                    "finish_reason": response.finish_reason,
                    "provider_response_json": json.dumps(_safe_provider_response_json(response), ensure_ascii=False),
                },
            )
            await self._reconcile_windows(
                session=session,
                lease=lease,
                minute_start=minute_start,
                day_start=day_start,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            await _clear_model_cooldowns(session=session, lease=lease)
            await _clear_transport_cooldowns(session=session, lease=lease)
            await session.execute(
                text(
                    """
                    UPDATE gemini_gateway.key_proxy_bindings
                    SET consecutive_failures = 0,
                        cooldown_level = 0,
                        sleep_until = NULL,
                        last_success_at = now(),
                        updated_at = now()
                    WHERE id = :binding_id
                    """
                ),
                {"binding_id": lease.binding_id},
            )
            if lease.proxy_id is not None:
                await session.execute(
                    text(
                        """
                        UPDATE gemini_gateway.proxy_endpoints
                        SET cooldown_level = 0,
                            sleep_until = NULL,
                            last_success_at = now(),
                            updated_at = now()
                        WHERE id = :proxy_id
                        """
                    ),
                    {"proxy_id": lease.proxy_id},
                )

    async def record_failure(
        self,
        lease: RouteLease | None,
        error: GatewayError,
        latency_ms: int,
        provider_called: bool,
    ) -> None:
        route_failure = _should_apply_route_failure_state(error=error, provider_called=provider_called)
        async with self._session_factory() as session, session.begin():
            if lease is not None:
                await session.execute(
                    text(
                        """
                        UPDATE gemini_gateway.route_attempts
                        SET status = 'failed',
                            error_type = :error_type,
                            error_message_safe = :error_message_safe,
                            retryable = :retryable,
                            latency_ms = :latency_ms,
                            provider_response_json = CAST(:provider_response_json AS jsonb),
                            updated_at = now()
                        WHERE id = :attempt_id
                        """
                    ),
                    {
                        "attempt_id": lease.attempt_id,
                        "error_type": error.reason,
                        "error_message_safe": _public_route_error_message(error),
                        "retryable": error.retryable,
                        "latency_ms": latency_ms,
                        "provider_response_json": json.dumps(
                            _safe_provider_failure_json(error),
                            ensure_ascii=False,
                        ),
                    },
                )
                if not provider_called:
                    await self._release_reserved_tokens(session=session, lease=lease)
                if route_failure:
                    await self._apply_failure_state(session=session, lease=lease, error=error)

    async def cleanup_route_attempts(self, ttl_days: int) -> int:
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                text(
                    """
                    DELETE FROM gemini_gateway.route_attempts
                    WHERE created_at < now() - (:ttl_days * interval '1 day')
                    """
                ),
                {"ttl_days": ttl_days},
            )
            return int(result.rowcount or 0)

    async def expire_elapsed_cooldowns(self, *, limit: int) -> int:
        if limit <= 0:
            return 0

        async with self._session_factory() as session, session.begin():
            rows = (
                await session.execute(
                    text(
                        """
                        WITH expired AS (
                            SELECT id
                            FROM gemini_gateway.cooldowns
                            WHERE status = 'active'
                              AND sleep_until <= now()
                            ORDER BY sleep_until ASC, id ASC
                            LIMIT :limit
                            FOR UPDATE SKIP LOCKED
                        )
                        UPDATE gemini_gateway.cooldowns cd
                        SET status = 'expired',
                            updated_at = now()
                        FROM expired
                        WHERE cd.id = expired.id
                        RETURNING cd.id
                        """
                    ),
                    {"limit": limit},
                )
            ).all()
            return len(rows)

    async def expire_elapsed_route_sleeps(self, *, limit: int) -> int:
        if limit <= 0:
            return 0

        async with self._session_factory() as session, session.begin():
            expired = 0
            for table_name in _ROUTE_SLEEP_STATE_TABLES:
                expired += await _clear_elapsed_sleep_state(
                    session=session,
                    table_name=table_name,
                    limit=limit,
                )
            return expired

    async def complete_stale_leases(self, *, stale_after_seconds: int, limit: int) -> int:
        if stale_after_seconds <= 0 or limit <= 0:
            return 0

        async with self._session_factory() as session, session.begin():
            rows = (
                await session.execute(
                    text(
                        """
                        SELECT id, project_id, model, created_at, reserved_input_tokens
                        FROM gemini_gateway.route_attempts
                        WHERE status = 'leased'
                          AND created_at <= now() - (:stale_after_seconds * interval '1 second')
                        ORDER BY created_at ASC, id ASC
                        LIMIT :limit
                        FOR UPDATE SKIP LOCKED
                        """
                    ),
                    {
                        "stale_after_seconds": stale_after_seconds,
                        "limit": limit,
                    },
                )
            ).mappings().all()
            for row in rows:
                reserved_tokens = int(row["reserved_input_tokens"] or 0)
                await _release_reserved_quota(
                    session=session,
                    project_id=int(row["project_id"]),
                    model=str(row["model"]),
                    leased_at=row["created_at"],
                    tokens=reserved_tokens,
                )
                await session.execute(
                    text(
                        """
                        UPDATE gemini_gateway.route_attempts
                        SET status = 'failed',
                            error_type = 'stale_lease',
                            error_message_safe = 'Gateway route lease expired before completion.',
                            retryable = true,
                            updated_at = now()
                        WHERE id = :attempt_id
                        """
                    ),
                    {"attempt_id": row["id"]},
                )
            return len(rows)

    async def health_check(self, *, require_routes: bool = False) -> dict[str, Any]:
        checks: dict[str, bool] = {"database": False, "schema": False, "routes": not require_routes}
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
                checks["database"] = True
                table_count = await session.scalar(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM information_schema.tables
                        WHERE table_schema = 'gemini_gateway'
                          AND table_name IN (
                            'google_projects',
                            'api_keys',
                            'proxy_endpoints',
                            'key_proxy_bindings',
                            'model_limits',
                            'quota_windows',
                            'cooldowns',
                            'route_attempts'
                          )
                        """
                    )
                )
                checks["schema"] = int(table_count or 0) == 8
                if require_routes:
                    route_count = await session.scalar(
                        text(
                            f"""
                            SELECT COUNT(*)
                            FROM gemini_gateway.key_proxy_bindings b
                            JOIN gemini_gateway.api_keys k ON k.id = b.api_key_id
                            LEFT JOIN gemini_gateway.proxy_endpoints px ON px.id = b.proxy_id
                            JOIN gemini_gateway.google_projects p ON p.id = k.project_id
                            JOIN gemini_gateway.model_limits ml ON ml.project_id = p.id
                            {_ROUTE_ELIGIBILITY_SQL.where_clause()}
                            """
                        ),
                        {"now": datetime.now(tz=UTC)},
                    )
                    checks["routes"] = int(route_count or 0) > 0
        except Exception:
            return {"ok": False, "checks": checks}
        return {"ok": all(checks.values()), "checks": checks}

    async def _try_reserve_candidate(
        self,
        *,
        candidate: RouteCandidate,
        request: GatewayRouteRequest,
        estimated_tokens: int,
        now: datetime,
    ) -> RouteLease | None:
        minute_start = _minute_start(now)
        day_start = _day_start(now)
        async with self._session_factory() as session:
            try:
                async with session.begin():
                    route_state = await self._load_locked_reservable_route(
                        session=session,
                        candidate=candidate,
                        now=now,
                    )
                    if route_state is None:
                        return None
                    route_secrets = self._decrypt_locked_route_secrets(route_state)
                    if route_secrets is None:
                        return None
                    minute_reserved = await _reserve_quota_window(
                        session=session,
                        project_id=int(candidate.project_id),
                        model=candidate.model,
                        window_kind="minute",
                        window_start=minute_start,
                        estimated_tokens=estimated_tokens,
                        request_limit=int(route_state["requests_per_minute"]),
                        token_limit=int(route_state["tokens_per_minute"]),
                    )
                    if not minute_reserved:
                        return None
                    day_token_limit = int(route_state["tokens_per_day"] or 2_147_483_647)
                    day_reserved = await _reserve_quota_window(
                        session=session,
                        project_id=int(candidate.project_id),
                        model=candidate.model,
                        window_kind="day",
                        window_start=day_start,
                        estimated_tokens=estimated_tokens,
                        request_limit=int(route_state["requests_per_day"]),
                        token_limit=day_token_limit,
                    )
                    if not day_reserved:
                        raise _ReservationFailed

                    attempt_id = await session.scalar(
                        text(
                            """
                            INSERT INTO gemini_gateway.route_attempts (
                                request_id,
                                soybob_request_id,
                                source_service,
                                chat_id,
                                telegram_message_id,
                                project_id,
                                api_key_id,
                                proxy_id,
                                binding_id,
                                route_label,
                                model,
                                status,
                                retry_count,
                                estimated_input_tokens,
                                reserved_input_tokens
                            )
                            VALUES (
                                :request_id,
                                :soybob_request_id,
                                :source_service,
                                :chat_id,
                                :telegram_message_id,
                                :project_id,
                                :api_key_id,
                                :proxy_id,
                                :binding_id,
                                :route_label,
                                :model,
                                'leased',
                                :retry_count,
                                :estimated_input_tokens,
                                :reserved_input_tokens
                            )
                            RETURNING id
                            """
                        ),
                        {
                            "request_id": request.request_id,
                            "soybob_request_id": _request_group_id(request),
                            "source_service": request.source_service,
                            "chat_id": request.chat_id,
                            "telegram_message_id": request.telegram_message_id,
                            "project_id": candidate.project_id,
                            "api_key_id": candidate.api_key_id,
                            "proxy_id": candidate.proxy_id,
                            "binding_id": candidate.binding_id,
                            "route_label": candidate.route_label,
                            "model": candidate.model,
                            "retry_count": int(request.retry_count or 0),
                            "estimated_input_tokens": estimated_tokens,
                            "reserved_input_tokens": estimated_tokens,
                        },
                    )
                    if attempt_id is None:
                        return None
                    return _lease_from_candidate(
                        candidate,
                        route_secrets=route_secrets,
                        estimated_tokens=estimated_tokens,
                        attempt_id=attempt_id,
                        now=now,
                    )
            except _ReservationFailed:
                return None

    async def _load_locked_reservable_route(
        self,
        *,
        session: AsyncSession,
        candidate: RouteCandidate,
        now: datetime,
    ) -> dict[str, Any] | None:
        where_clause = _ROUTE_ELIGIBILITY_SQL.where_clause(extra_conditions=_LOCKED_ROUTE_EXTRA_CONDITIONS)
        row = (
            await session.execute(
                text(
                    f"""
                    SELECT
                        k.encrypted_api_key,
                        b.transport_mode,
                        px.scheme,
                        px.host,
                        px.port,
                        px.encrypted_username,
                        px.encrypted_password,
                        ml.requests_per_minute,
                        ml.tokens_per_minute,
                        ml.requests_per_day,
                        ml.tokens_per_day
                    FROM gemini_gateway.key_proxy_bindings b
                    JOIN gemini_gateway.api_keys k ON k.id = b.api_key_id
                    LEFT JOIN gemini_gateway.proxy_endpoints px ON px.id = b.proxy_id
                    JOIN gemini_gateway.google_projects p ON p.id = k.project_id
                    JOIN gemini_gateway.model_limits ml ON ml.project_id = p.id AND ml.model = :model
                    {where_clause}
                    FOR UPDATE OF b, k, p, ml SKIP LOCKED
                    """
                ),
                {
                    "binding_id": candidate.binding_id,
                    "api_key_id": candidate.api_key_id,
                    "proxy_id": candidate.proxy_id,
                    "project_id": candidate.project_id,
                    "model": candidate.model,
                    "now": now,
                },
            )
        ).mappings().first()
        return dict(row) if row is not None else None

    def _decrypt_locked_route_secrets(self, route_state: dict[str, Any]) -> _RouteSecrets | None:
        try:
            api_key = self._secret_vault.decrypt(route_state["encrypted_api_key"])
            if api_key is None:
                return None
            proxy_url: str | None = None
            if route_state["transport_mode"] == "proxy":
                username = self._secret_vault.decrypt(route_state["encrypted_username"])
                password = self._secret_vault.decrypt(route_state["encrypted_password"])
                proxy_url = build_proxy_url(
                    ProxySecretParts(
                        scheme=route_state["scheme"],
                        host=route_state["host"],
                        port=int(route_state["port"]),
                        username=username,
                        password=password,
                    )
                )
            return _RouteSecrets(api_key=api_key, proxy_url=proxy_url)
        except (SecretDecryptionError, ValueError):
            return None

    async def _record_skipped_no_route(self, *, request: GatewayRouteRequest, estimated_tokens: int) -> None:
        async with self._session_factory() as session, session.begin():
            await session.execute(
                text(
                    """
                    INSERT INTO gemini_gateway.route_attempts (
                        request_id,
                        soybob_request_id,
                        source_service,
                        chat_id,
                        telegram_message_id,
                        model,
                        status,
                        error_type,
                        error_message_safe,
                        retryable,
                        retry_count,
                        estimated_input_tokens
                    )
                    VALUES (
                        :request_id,
                        :soybob_request_id,
                        :source_service,
                        :chat_id,
                        :telegram_message_id,
                        :model,
                        'skipped_no_route',
                        'no_route',
                        :error_message_safe,
                        true,
                        :retry_count,
                        :estimated_tokens
                    )
                    """
                ),
                {
                    "request_id": request.request_id,
                    "soybob_request_id": _request_group_id(request),
                    "source_service": request.source_service,
                    "chat_id": request.chat_id,
                    "telegram_message_id": request.telegram_message_id,
                    "model": request.model,
                    "error_message_safe": public_message_for_reason("no_route"),
                    "retry_count": int(request.retry_count or 0),
                    "estimated_tokens": estimated_tokens,
                },
            )

    async def _attempted_binding_ids(self, *, request: GatewayRouteRequest) -> set[str]:
        query = text(
            """
            SELECT DISTINCT binding_id::text
            FROM gemini_gateway.route_attempts
            WHERE soybob_request_id = :soybob_request_id
              AND model = :model
              AND binding_id IS NOT NULL
            """
        )
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    query,
                    {
                        "soybob_request_id": _request_group_id(request),
                        "model": request.model,
                    },
                )
            ).all()
        return {str(row[0]) for row in rows if row[0] is not None}

    async def _release_reserved_tokens(self, *, session: AsyncSession, lease: RouteLease) -> None:
        await _release_reserved_quota(
            session=session,
            project_id=int(lease.project_id),
            model=lease.model,
            leased_at=lease.leased_at,
            tokens=lease.estimated_tokens,
        )

    async def _reconcile_windows(
        self,
        *,
        session: AsyncSession,
        lease: RouteLease,
        minute_start: datetime,
        day_start: datetime,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
    ) -> None:
        actual_prompt = prompt_tokens or lease.estimated_tokens
        actual_completion = completion_tokens or 0
        actual_total = total_tokens or actual_prompt + actual_completion
        for kind, start in (("minute", minute_start), ("day", day_start)):
            await session.execute(
                text(
                    """
                    UPDATE gemini_gateway.quota_windows
                    SET input_tokens_reserved = GREATEST(0, input_tokens_reserved - :reserved + :actual_total),
                        input_tokens_used = input_tokens_used + :actual_prompt,
                        output_tokens_used = output_tokens_used + :actual_completion,
                        total_tokens_used = total_tokens_used + :actual_total,
                        updated_at = now()
                    WHERE project_id = :project_id
                      AND model = :model
                      AND window_kind = :window_kind
                      AND window_start = :window_start
                    """
                ),
                {
                    "reserved": lease.estimated_tokens,
                    "actual_prompt": actual_prompt,
                    "actual_completion": actual_completion,
                    "actual_total": actual_total,
                    "project_id": lease.project_id,
                    "model": lease.model,
                    "window_kind": kind,
                    "window_start": start,
                },
            )

    async def _apply_failure_state(self, *, session: AsyncSession, lease: RouteLease, error: GatewayError) -> None:
        if error.reason == "auth_failed":
            await session.execute(
                text(
                    """
                    UPDATE gemini_gateway.api_keys
                    SET status = 'disabled_needs_review',
                        last_failure_at = now(),
                        last_error_type = :error_type,
                        last_error_message_safe = :error_message_safe,
                        updated_at = now()
                    WHERE id = :api_key_id
                    """
                ),
                {
                    "api_key_id": lease.api_key_id,
                    "error_type": error.reason,
                    "error_message_safe": _public_route_error_message(error),
                },
            )
            return

        current_level = await self._cooldown_level_for_failure(session=session, lease=lease, error=error)
        decision = self._cooldown_policy.next_decision(
            reason=error.reason,
            current_level=current_level,
            retry_after_seconds=error.retry_after_seconds,
        )
        if decision is None:
            return
        scopes = tuple(scope for scope in decision.scopes if scope != "proxy" or lease.proxy_id is not None)
        _attach_cooldown_metadata(error=error, decision=decision, scopes=scopes)
        for scope in scopes:
            scope_key, model = _cooldown_scope_key(scope=scope, lease=lease, error=error)
            await _upsert_cooldown(
                session=session,
                scope=scope,
                scope_key=scope_key,
                model=model,
                error=error,
                level=decision.level,
                sleep_until=decision.sleep_until,
                half_open_after=decision.half_open_after,
            )
        if error.reason in _MODEL_SCOPED_TRANSPORT_COOLDOWN_REASONS:
            return
        if error.reason in _MODEL_SCOPED_COOLDOWN_REASONS:
            return
        await session.execute(
            text(
                """
                UPDATE gemini_gateway.key_proxy_bindings
                SET consecutive_failures = consecutive_failures + 1,
                    cooldown_level = :cooldown_level,
                    sleep_until = :sleep_until,
                    last_failure_at = now(),
                    last_error_type = :error_type,
                    last_error_message_safe = :error_message_safe,
                    updated_at = now()
                WHERE id = :binding_id
                """
            ),
            {
                "binding_id": lease.binding_id,
                "cooldown_level": decision.level,
                "sleep_until": decision.sleep_until,
                "error_type": error.reason,
                "error_message_safe": _public_route_error_message(error),
            },
        )
        if error.reason in _TRANSPORT_COOLDOWN_REASONS and lease.proxy_id is not None:
            await session.execute(
                text(
                    """
                    UPDATE gemini_gateway.proxy_endpoints
                    SET cooldown_level = :cooldown_level,
                        sleep_until = :sleep_until,
                        last_failure_at = now(),
                        last_error_type = :error_type,
                        last_error_message_safe = :error_message_safe,
                        updated_at = now()
                    WHERE id = :proxy_id
                    """
                ),
                {
                    "proxy_id": lease.proxy_id,
                    "cooldown_level": decision.level,
                    "sleep_until": decision.sleep_until,
                    "error_type": error.reason,
                    "error_message_safe": _public_route_error_message(error),
                },
            )

    async def _cooldown_level_for_failure(
        self,
        *,
        session: AsyncSession,
        lease: RouteLease,
        error: GatewayError,
    ) -> int:
        if error.reason in _PROJECT_MODEL_COOLDOWN_REASONS:
            return await _active_cooldown_level(
                session=session,
                scope="project_model",
                scope_key=str(lease.project_id),
                model=lease.model,
            )
        if error.reason in _MODEL_SCOPED_TRANSPORT_COOLDOWN_REASONS:
            return await _active_cooldown_level(
                session=session,
                scope="binding",
                scope_key=str(lease.binding_id),
                model=lease.model,
            )
        return await self._binding_cooldown_level(session, lease.binding_id, reason=error.reason)

    async def _binding_cooldown_level(
        self,
        session: AsyncSession,
        binding_id: int | str,
        *,
        reason: str,
    ) -> int:
        row = (
            await session.execute(
                text(
                    """
                    SELECT
                        CASE
                            WHEN sleep_until IS NOT NULL AND sleep_until > now() THEN cooldown_level
                            ELSE 0
                        END AS active_cooldown_level,
                        last_error_type
                    FROM gemini_gateway.key_proxy_bindings
                    WHERE id = :binding_id
                    """
                ),
                {"binding_id": binding_id},
            )
        ).mappings().first()
        if row is None:
            return 0
        last_error_type = row["last_error_type"]
        if last_error_type and _cooldown_reason_family(str(last_error_type)) != _cooldown_reason_family(reason):
            return 0
        return int(row["active_cooldown_level"] or 0)

    async def _upsert_projects(self, session: AsyncSession, config: SeedConfig) -> dict[str, int]:
        project_ids: dict[str, int] = {}
        for project in config.projects:
            project_id = await session.scalar(
                text(
                    """
                    INSERT INTO gemini_gateway.google_projects (
                        label, owner_name, owner_contact, google_project_ref, tier
                    )
                    VALUES (:label, :owner_name, :owner_contact, :google_project_ref, :tier)
                    ON CONFLICT (label) DO UPDATE SET
                        owner_name = EXCLUDED.owner_name,
                        owner_contact = EXCLUDED.owner_contact,
                        google_project_ref = EXCLUDED.google_project_ref,
                        tier = EXCLUDED.tier,
                        updated_at = now()
                    RETURNING id
                    """
                ),
                project.model_dump(),
            )
            if project_id is None:
                continue
            project_ids[project.label] = int(project_id)
            for limit in project.model_limits:
                await session.execute(
                    text(
                        """
                        INSERT INTO gemini_gateway.model_limits (
                            project_id, model, requests_per_minute, tokens_per_minute, requests_per_day, tokens_per_day
                        )
                        VALUES (
                            :project_id, :model, :requests_per_minute, :tokens_per_minute, :requests_per_day, :tokens_per_day
                        )
                        ON CONFLICT (project_id, model) DO UPDATE SET
                            requests_per_minute = EXCLUDED.requests_per_minute,
                            tokens_per_minute = EXCLUDED.tokens_per_minute,
                            requests_per_day = EXCLUDED.requests_per_day,
                            tokens_per_day = EXCLUDED.tokens_per_day,
                            status = 'active',
                            updated_at = now()
                        """
                    ),
                    {"project_id": project_id, **limit.model_dump()},
                )
        return project_ids

    async def _upsert_keys(
        self,
        session: AsyncSession,
        config: SeedConfig,
        projects: dict[str, int],
    ) -> dict[tuple[str, str], int]:
        key_ids: dict[tuple[str, str], int] = {}
        for key in config.api_keys:
            api_key = normalize_api_key(key.api_key)
            key_fingerprint = self._secret_vault.fingerprint("api-key", api_key)
            payload = {
                "project_id": projects[key.project_label],
                "label": key.label,
                "encrypted_api_key": self._secret_vault.encrypt(api_key),
                "key_fingerprint": key_fingerprint,
            }
            key_id = await session.scalar(
                text(
                    """
                    UPDATE gemini_gateway.api_keys
                    SET project_id = :project_id,
                        label = :label,
                        encrypted_api_key = :encrypted_api_key,
                        status = 'active',
                        updated_at = now()
                    WHERE key_fingerprint = :key_fingerprint
                    RETURNING id
                    """
                ),
                payload,
            )
            if key_id is None:
                key_id = await session.scalar(
                    text(
                        """
                    INSERT INTO gemini_gateway.api_keys (
                        project_id, label, encrypted_api_key, key_fingerprint
                    )
                    VALUES (:project_id, :label, :encrypted_api_key, :key_fingerprint)
                    ON CONFLICT (project_id, label) DO UPDATE SET
                        encrypted_api_key = EXCLUDED.encrypted_api_key,
                        key_fingerprint = EXCLUDED.key_fingerprint,
                        status = 'active',
                        updated_at = now()
                    RETURNING id
                    """
                    ),
                    payload,
                )
            if key_id is not None:
                key_ids[(key.project_label, key.label)] = int(key_id)
        return key_ids

    async def _upsert_proxies(self, session: AsyncSession, config: SeedConfig) -> dict[str, int]:
        proxy_ids: dict[str, int] = {}
        for proxy in config.proxies:
            proxy_id = await self._upsert_proxy(session, proxy)
            proxy_ids[proxy.label] = proxy_id
        return proxy_ids

    async def _upsert_proxy(self, session: AsyncSession, proxy: SeedProxy) -> int:
        parts = ProxySecretParts(
            scheme=proxy.scheme,
            host=proxy.host,
            port=proxy.port,
            username=proxy.username,
            password=proxy.password,
        )
        normalized_source = proxy_fingerprint_source(parts)
        proxy_id = await session.scalar(
            text(
                """
                INSERT INTO gemini_gateway.proxy_endpoints (
                    label, scheme, host, port, encrypted_username, encrypted_password, proxy_fingerprint
                )
                VALUES (
                    :label, :scheme, :host, :port, :encrypted_username, :encrypted_password, :proxy_fingerprint
                )
                ON CONFLICT (label) DO UPDATE SET
                    scheme = EXCLUDED.scheme,
                    host = EXCLUDED.host,
                    port = EXCLUDED.port,
                    encrypted_username = EXCLUDED.encrypted_username,
                    encrypted_password = EXCLUDED.encrypted_password,
                    proxy_fingerprint = EXCLUDED.proxy_fingerprint,
                    status = 'active',
                    updated_at = now()
                RETURNING id
                """
            ),
            {
                "label": proxy.label,
                "scheme": proxy.scheme,
                "host": proxy.host,
                "port": proxy.port,
                "encrypted_username": self._secret_vault.encrypt(proxy.username) if proxy.username else None,
                "encrypted_password": self._secret_vault.encrypt(proxy.password) if proxy.password else None,
                "proxy_fingerprint": self._secret_vault.fingerprint("proxy", normalized_source),
            },
        )
        if proxy_id is None:
            raise GatewayError(reason="request_failed", request_id="seed")
        return int(proxy_id)

    async def _upsert_bindings(
        self,
        session: AsyncSession,
        bindings: list[SeedBinding],
        keys: dict[tuple[str, str], int],
        proxies: dict[str, int],
    ) -> None:
        for binding in bindings:
            api_key_id = _resolve_seed_binding_key_id(binding=binding, keys=keys)
            proxy_id = _resolve_seed_binding_proxy_id(binding=binding, proxies=proxies)
            payload = {
                "label": binding.label,
                "api_key_id": api_key_id,
                "proxy_id": proxy_id,
                "transport_mode": binding.transport_mode,
            }
            binding_id = await session.scalar(
                text(
                    """
                    UPDATE gemini_gateway.key_proxy_bindings
                    SET label = :label,
                        api_key_id = :api_key_id,
                        proxy_id = :proxy_id,
                        transport_mode = :transport_mode,
                        status = 'active',
                        updated_at = now()
                    WHERE status = 'active'
                      AND (
                        api_key_id = :api_key_id
                        OR (
                            CAST(:proxy_id AS bigint) IS NOT NULL
                            AND proxy_id = CAST(:proxy_id AS bigint)
                        )
                      )
                    RETURNING id
                    """
                ),
                payload,
            )
            if binding_id is not None:
                continue
            await session.execute(
                text(
                    """
                    INSERT INTO gemini_gateway.key_proxy_bindings (label, api_key_id, proxy_id, transport_mode)
                    VALUES (:label, :api_key_id, :proxy_id, :transport_mode)
                    ON CONFLICT (label) DO UPDATE SET
                        api_key_id = EXCLUDED.api_key_id,
                        proxy_id = EXCLUDED.proxy_id,
                        transport_mode = EXCLUDED.transport_mode,
                        status = 'active',
                        updated_at = now()
                    """
                ),
                payload,
            )


class PostgresGatewaySeedRepository(PostgresGatewayRepository):
    """Seed/admin writes для gateway без публикации seed API в runtime repository."""

    async def upsert_seed_config(self, config: SeedConfig) -> None:
        async with self._session_factory() as session, session.begin():
            projects = await self._upsert_projects(session, config)
            keys = await self._upsert_keys(session, config, projects)
            proxies = await self._upsert_proxies(session, config)
            await self._upsert_bindings(session, config.bindings, keys, proxies)


def _resolve_seed_binding_key_id(*, binding: SeedBinding, keys: dict[tuple[str, str], int]) -> int:
    if binding.project_label is not None:
        return keys[(binding.project_label, binding.api_key_label)]

    matches = [key_id for (project_label, key_label), key_id in keys.items() if key_label == binding.api_key_label]
    if len(matches) != 1:
        raise GatewayError(reason="request_failed", request_id="seed")
    return matches[0]


def _resolve_seed_binding_proxy_id(*, binding: SeedBinding, proxies: dict[str, int]) -> int | None:
    if binding.transport_mode == "direct":
        return None
    if binding.proxy_label is None:
        raise GatewayError(reason="request_failed", request_id="seed")
    return proxies[binding.proxy_label]


class _ReservationFailed(RuntimeError):
    pass


async def _reserve_quota_window(
    *,
    session: AsyncSession,
    project_id: int,
    model: str,
    window_kind: str,
    window_start: datetime,
    estimated_tokens: int,
    request_limit: int,
    token_limit: int,
) -> bool:
    row_id = await session.scalar(
        text(
            """
            INSERT INTO gemini_gateway.quota_windows (
                project_id, model, window_kind, window_start, requests_used, input_tokens_reserved
            )
            VALUES (:project_id, :model, :window_kind, :window_start, 1, :estimated_tokens)
            ON CONFLICT (project_id, model, window_kind, window_start)
            DO UPDATE SET
                requests_used = gemini_gateway.quota_windows.requests_used + 1,
                input_tokens_reserved = gemini_gateway.quota_windows.input_tokens_reserved + :estimated_tokens,
                updated_at = now()
            WHERE
                gemini_gateway.quota_windows.requests_used + 1 <= :request_limit
                AND gemini_gateway.quota_windows.input_tokens_reserved + :estimated_tokens <= :token_limit
            RETURNING id
            """
        ),
        {
            "project_id": project_id,
            "model": model,
            "window_kind": window_kind,
            "window_start": window_start,
            "estimated_tokens": estimated_tokens,
            "request_limit": request_limit,
            "token_limit": token_limit,
        },
    )
    return row_id is not None


async def _release_reserved_quota(
    *,
    session: AsyncSession,
    project_id: int,
    model: str,
    leased_at: datetime,
    tokens: int,
) -> None:
    for kind, start in (("minute", _minute_start(leased_at)), ("day", _day_start(leased_at))):
        await session.execute(
            text(
                """
                UPDATE gemini_gateway.quota_windows
                SET requests_used = GREATEST(0, requests_used - 1),
                    input_tokens_reserved = GREATEST(0, input_tokens_reserved - :tokens),
                    updated_at = now()
                WHERE project_id = :project_id
                  AND model = :model
                  AND window_kind = :window_kind
                  AND window_start = :window_start
                """
            ),
            {
                "tokens": tokens,
                "project_id": project_id,
                "model": model,
                "window_kind": kind,
                "window_start": start,
            },
        )


async def _upsert_cooldown(
    *,
    session: AsyncSession,
    scope: str,
    scope_key: str,
    model: str | None,
    error: GatewayError,
    level: int,
    sleep_until: datetime,
    half_open_after: datetime,
) -> None:
    if model is None:
        conflict_clause = "(scope, scope_key, status) WHERE model IS NULL AND status = 'active'"
    else:
        conflict_clause = "(scope, scope_key, model, status) WHERE model IS NOT NULL AND status = 'active'"
    await session.execute(
        text(
            f"""
            INSERT INTO gemini_gateway.cooldowns (
                scope, scope_key, model, error_type, cooldown_level, sleep_until, half_open_after, status
            )
            VALUES (:scope, :scope_key, :model, :error_type, :cooldown_level, :sleep_until, :half_open_after, 'active')
            ON CONFLICT {conflict_clause}
            DO UPDATE SET
                error_type = EXCLUDED.error_type,
                cooldown_level = EXCLUDED.cooldown_level,
                sleep_until = EXCLUDED.sleep_until,
                half_open_after = EXCLUDED.half_open_after,
                updated_at = now()
            """
        ),
        {
            "scope": scope,
            "scope_key": scope_key,
            "model": model,
            "error_type": error.reason,
            "cooldown_level": level,
            "sleep_until": sleep_until,
            "half_open_after": half_open_after,
        },
    )


async def _clear_elapsed_sleep_state(*, session: AsyncSession, table_name: str, limit: int) -> int:
    """Сбрасывает прошедший sleep на route-сущностях без стирания причины."""

    if table_name not in _ROUTE_SLEEP_STATE_TABLES:
        raise ValueError("Unsupported route sleep table.")

    rows = (
        await session.execute(
            text(
                f"""
                WITH expired AS (
                    SELECT id
                    FROM gemini_gateway.{table_name}
                    WHERE status = 'active'
                      AND sleep_until IS NOT NULL
                      AND sleep_until <= now()
                    ORDER BY sleep_until ASC, id ASC
                    LIMIT :limit
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE gemini_gateway.{table_name} target
                SET cooldown_level = 0,
                    sleep_until = NULL,
                    updated_at = now()
                FROM expired
                WHERE target.id = expired.id
                RETURNING target.id
                """
            ),
            {"limit": limit},
        )
    ).all()
    return len(rows)


async def _active_cooldown_level(
    *,
    session: AsyncSession,
    scope: str,
    scope_key: str,
    model: str | None,
) -> int:
    if model is None:
        model_condition = "model IS NULL"
        params: dict[str, Any] = {"scope": scope, "scope_key": scope_key}
    else:
        model_condition = "model = :model"
        params = {"scope": scope, "scope_key": scope_key, "model": model}
    level = await session.scalar(
        text(
            f"""
            SELECT cooldown_level
            FROM gemini_gateway.cooldowns
            WHERE scope = :scope
              AND scope_key = :scope_key
              AND {model_condition}
              AND status = 'active'
              AND sleep_until > now()
            """
        ),
        params,
    )
    return int(level or 0)


async def _clear_model_cooldowns(*, session: AsyncSession, lease: RouteLease) -> None:
    await session.execute(
        text(
            """
            UPDATE gemini_gateway.cooldowns
            SET status = 'cleared',
                updated_at = now()
            WHERE (
                    (
                        scope = 'project_model'
                        AND scope_key = :project_id
                        AND model = :model
                    )
                    OR (
                        scope = 'provider_model'
                        AND scope_key = :model
                        AND model IS NULL
                    )
                  )
              AND status = 'active'
            """
        ),
        {"project_id": str(lease.project_id), "model": lease.model},
    )


async def _clear_transport_cooldowns(*, session: AsyncSession, lease: RouteLease) -> None:
    await session.execute(
        text(
            """
            UPDATE gemini_gateway.cooldowns
            SET status = 'cleared',
                updated_at = now()
            WHERE status = 'active'
              AND (
                (
                  model IS NULL
                  AND (
                    (
                        scope = 'binding'
                        AND scope_key = :binding_id
                    )
                    OR (
                        scope = 'proxy'
                        AND scope_key = :proxy_id
                    )
                  )
                )
                OR (
                  model = :model
                  AND (
                        (
                            scope = 'binding'
                            AND scope_key = :binding_id
                        )
                        OR (
                            scope = 'proxy'
                            AND scope_key = :proxy_id
                        )
                  )
                )
              )
            """
        ),
        {
            "binding_id": str(lease.binding_id),
            "model": lease.model,
            "proxy_id": str(lease.proxy_id) if lease.proxy_id is not None else None,
        },
    )


def estimate_request_tokens(request: GatewayRouteRequest) -> int:
    explicit_estimate = _safe_int(request.estimated_input_tokens)
    if explicit_estimate is not None and explicit_estimate > 0:
        return explicit_estimate

    text_size = 0
    messages = getattr(request, "messages", None)
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                text_size += _content_size(message.get("content"))
        return max(1, (text_size // 4) + 128)
    text_size = _content_size(getattr(request, "text", ""))
    return max(1, (text_size // 4) + 128)


def _request_group_id(request: GatewayRouteRequest) -> str:
    """Возвращает стабильный id одного LLM-вызова для обхода route pool без повторов."""

    return request.soybob_request_id or request.request_id


def _lease_from_candidate(
    candidate: RouteCandidate,
    *,
    route_secrets: _RouteSecrets | None = None,
    estimated_tokens: int,
    attempt_id: int | str,
    now: datetime,
) -> RouteLease:
    api_key = route_secrets.api_key if route_secrets is not None else candidate.api_key
    proxy_url = route_secrets.proxy_url if route_secrets is not None else candidate.proxy_url
    return RouteLease(
        attempt_id=attempt_id,
        binding_id=candidate.binding_id,
        project_id=candidate.project_id,
        api_key_id=candidate.api_key_id,
        proxy_id=candidate.proxy_id,
        api_key=api_key,
        proxy_url=proxy_url,
        model=candidate.model,
        route_label=candidate.route_label,
        project_label=candidate.project_label,
        key_label=candidate.key_label,
        proxy_label=candidate.proxy_label,
        transport_mode=candidate.transport_mode,
        estimated_tokens=estimated_tokens,
        leased_at=now,
    )


def _cooldown_scope_key(*, scope: str, lease: RouteLease, error: GatewayError) -> tuple[str, str | None]:
    if scope == "project_model":
        return str(lease.project_id), lease.model
    if scope == "key":
        return str(lease.api_key_id), None
    if scope == "proxy":
        return str(lease.proxy_id), _transport_cooldown_model(lease=lease, error=error)
    if scope == "binding":
        return str(lease.binding_id), _transport_cooldown_model(lease=lease, error=error)
    if scope == "provider_model":
        return lease.model, None
    return str(lease.binding_id), None


def _transport_cooldown_model(*, lease: RouteLease, error: GatewayError) -> str | None:
    if error.reason in _MODEL_SCOPED_TRANSPORT_COOLDOWN_REASONS:
        return lease.model
    return None


def _content_size(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict):
                total += _content_size(part.get("text") or part.get("content") or "")
            else:
                total += _content_size(part)
        return total
    return len(str(content)) if content is not None else 0


def _aware_or_none(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _retry_after_seconds_until(sleep_until: datetime | None, *, now: datetime) -> int | None:
    aware_sleep_until = _aware_or_none(sleep_until)
    if aware_sleep_until is None:
        return None
    aware_now = _aware_or_none(now) or datetime.now(tz=UTC)
    seconds = ceil((aware_sleep_until - aware_now).total_seconds())
    return max(1, seconds) if seconds > 0 else None


def _next_in_memory_retry_after_seconds(candidates: list[RouteCandidate], *, now: datetime) -> int | None:
    sleep_until_values = [
        cooldown_until
        for candidate in candidates
        if (cooldown_until := _aware_or_none(candidate.cooldown_until)) is not None and cooldown_until > now
    ]
    if not sleep_until_values:
        return None
    return _retry_after_seconds_until(min(sleep_until_values), now=now)


def _should_apply_route_failure_state(*, error: GatewayError, provider_called: bool) -> bool:
    """Определяет, нужно ли охлаждать маршрут после ошибки lease/provider."""

    if error.reason == "auth_failed":
        return provider_called
    if not error.retryable:
        return False
    return provider_called or error.reason in _LEASE_FAILURE_COOLDOWN_REASONS


def _cooldown_reason_family(reason: str) -> str:
    if reason in _TRANSPORT_COOLDOWN_REASONS:
        return "transport"
    return reason


def _attach_cooldown_metadata(
    *,
    error: GatewayError,
    decision: CooldownDecision,
    scopes: tuple[str, ...],
) -> None:
    """Добавляет cooldown decision к ошибке для последующего wide event."""

    error.cooldown_scope = ",".join(scopes) if scopes else None
    error.cooldown_level = decision.level
    error.sleep_until = decision.sleep_until
    if error.retry_after_seconds is None:
        error.retry_after_seconds = decision.seconds


def _safe_provider_response_json(response: Any) -> dict[str, Any]:
    """Готовит provider response для БД без base64-медиа и секретных полей."""

    raw_response = getattr(response, "raw_response", None)
    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json")
    elif isinstance(raw_response, dict) and raw_response:
        payload = dict(raw_response)
    elif isinstance(response, dict):
        payload = dict(response)
    else:
        payload = {key: value for key, value in vars(response).items() if not key.startswith("_")}
    payload.pop("raw_response", None)
    return sanitize_payload_for_audit(payload)


def _safe_provider_failure_json(error: GatewayError) -> dict[str, Any]:
    """Готовит диагностический payload ошибки без raw provider текста."""

    payload: dict[str, Any] = {
        "reason": error.reason,
        "retryable": error.retryable,
    }
    if error.provider_status_code is not None:
        payload["provider_status_code"] = error.provider_status_code
    provider_reason = public_provider_reason(error.provider_message_safe)
    if provider_reason is not None:
        payload["provider_reason"] = provider_reason
    if error.retry_after_seconds is not None:
        payload["retry_after_seconds"] = error.retry_after_seconds
    return payload


def _public_route_error_message(error: GatewayError) -> str:
    """Возвращает стабильный текст ошибки без provider-фрагментов."""

    return public_message_for_reason(error.reason)


def _minute_start(value: datetime) -> datetime:
    aware = _aware_or_none(value) or datetime.now(tz=UTC)
    return aware.replace(second=0, microsecond=0)


def _day_start(value: datetime) -> datetime:
    aware = _aware_or_none(value) or datetime.now(tz=UTC)
    return aware.replace(hour=0, minute=0, second=0, microsecond=0)
