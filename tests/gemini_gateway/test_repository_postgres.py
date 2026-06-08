from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from gemini_gateway.contracts import (
    GatewayChatRequest,
    GatewayChatResponse,
    SeedApiKey,
    SeedBinding,
    SeedConfig,
    SeedModelLimit,
    SeedProject,
    SeedProxy,
)
from gemini_gateway.errors import GatewayError, public_message_for_reason
from gemini_gateway.repository import PostgresGatewayRepository, PostgresGatewaySeedRepository
from gemini_gateway.security import SecretVault
from tests.db_test_policy import default_test_postgres_dsn, skip_or_fail_db_unavailable

TEST_POSTGRES_DSN = default_test_postgres_dsn("GEMINI_GATEWAY_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.db
_ENCRYPTION_KEY = "fTypay4X7jiYzb-rZUnV74QuKmcRQev9i12cZrFI3E8="
_HMAC_KEY = "h" * 32


@dataclass(frozen=True)
class PostgresTestContext:
    session_factory: async_sessionmaker[AsyncSession]
    repository: PostgresGatewayRepository
    seed_repository: PostgresGatewaySeedRepository
    suffix: str


async def _engine_or_skip() -> AsyncEngine:
    engine = create_async_engine(TEST_POSTGRES_DSN, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            schema_ready = await connection.scalar(text("SELECT to_regclass('gemini_gateway.route_attempts')"))
    except Exception as exc:  # noqa: BLE001
        await engine.dispose()
        skip_or_fail_db_unavailable(f"Postgres gateway schema is unavailable: {exc}")

    if schema_ready is None:
        await engine.dispose()
        skip_or_fail_db_unavailable("Postgres gateway schema is not migrated")

    return engine


def _repository(session_factory: async_sessionmaker[AsyncSession]) -> PostgresGatewayRepository:
    return PostgresGatewayRepository(
        session_factory=session_factory,
        secret_vault=SecretVault(
            encryption_key=_ENCRYPTION_KEY,
            hmac_key=_HMAC_KEY,
        ),
    )


def _seed_repository(session_factory: async_sessionmaker[AsyncSession]) -> PostgresGatewaySeedRepository:
    return PostgresGatewaySeedRepository(
        session_factory=session_factory,
        secret_vault=SecretVault(
            encryption_key=_ENCRYPTION_KEY,
            hmac_key=_HMAC_KEY,
        ),
    )


class _CountingSecretVault:
    def __init__(self) -> None:
        self._inner = SecretVault(encryption_key=_ENCRYPTION_KEY, hmac_key=_HMAC_KEY)
        self.decrypt_calls = 0

    def encrypt(self, value: str) -> str:
        return self._inner.encrypt(value)

    def fingerprint(self, namespace: str, value: str) -> str:
        return self._inner.fingerprint(namespace, value)

    def decrypt(self, encrypted_value: str | None) -> str | None:
        self.decrypt_calls += 1
        return self._inner.decrypt(encrypted_value)


async def _delete_postgres_test_rows(session_factory: async_sessionmaker[AsyncSession], suffix: str) -> None:
    suffix_pattern = f"%-{suffix}"
    request_pattern = f"pg-test-{suffix}-%"

    async with session_factory() as session, session.begin():
        await session.execute(
            text(
                """
                DELETE FROM gemini_gateway.route_attempts
                WHERE request_id LIKE :request_pattern
                   OR model = :model
                   OR project_id IN (
                       SELECT id FROM gemini_gateway.google_projects WHERE label LIKE :suffix_pattern
                   )
                   OR api_key_id IN (
                       SELECT id FROM gemini_gateway.api_keys WHERE label LIKE :suffix_pattern
                   )
                   OR proxy_id IN (
                       SELECT id FROM gemini_gateway.proxy_endpoints WHERE label LIKE :suffix_pattern
                   )
                   OR binding_id IN (
                       SELECT id FROM gemini_gateway.key_proxy_bindings WHERE label LIKE :suffix_pattern
                   )
                """
            ),
            {
                "model": _model(suffix),
                "request_pattern": request_pattern,
                "suffix_pattern": suffix_pattern,
            },
        )
        await session.execute(
            text(
                """
                DELETE FROM gemini_gateway.cooldowns cd
                WHERE (
                       cd.scope = 'project_model'
                       AND cd.scope_key IN (
                           SELECT id::text FROM gemini_gateway.google_projects WHERE label LIKE :suffix_pattern
                       )
                   )
                   OR (
                       cd.scope = 'key'
                       AND cd.scope_key IN (
                           SELECT id::text FROM gemini_gateway.api_keys WHERE label LIKE :suffix_pattern
                       )
                   )
                   OR (
                       cd.scope = 'proxy'
                       AND cd.scope_key IN (
                           SELECT id::text FROM gemini_gateway.proxy_endpoints WHERE label LIKE :suffix_pattern
                       )
                   )
                   OR (
                       cd.scope = 'binding'
                       AND cd.scope_key IN (
                           SELECT id::text FROM gemini_gateway.key_proxy_bindings WHERE label LIKE :suffix_pattern
                       )
                   )
                   OR (
                       cd.scope = 'provider_model'
                       AND (
                           cd.scope_key = :model
                           OR cd.model = :model
                           OR cd.scope_key LIKE :suffix_pattern
                           OR cd.model LIKE :suffix_pattern
                       )
                   )
                """
            ),
            {"model": _model(suffix), "suffix_pattern": suffix_pattern},
        )
        await session.execute(
            text(
                """
                DELETE FROM gemini_gateway.quota_windows
                WHERE project_id IN (
                   SELECT id FROM gemini_gateway.google_projects WHERE label LIKE :suffix_pattern
                )
                """
            ),
            {"suffix_pattern": suffix_pattern},
        )
        await session.execute(
            text("DELETE FROM gemini_gateway.key_proxy_bindings WHERE label LIKE :suffix_pattern"),
            {"suffix_pattern": suffix_pattern},
        )
        await session.execute(
            text("DELETE FROM gemini_gateway.proxy_endpoints WHERE label LIKE :suffix_pattern"),
            {"suffix_pattern": suffix_pattern},
        )
        await session.execute(
            text("DELETE FROM gemini_gateway.api_keys WHERE label LIKE :suffix_pattern"),
            {"suffix_pattern": suffix_pattern},
        )
        await session.execute(
            text("DELETE FROM gemini_gateway.google_projects WHERE label LIKE :suffix_pattern"),
            {"suffix_pattern": suffix_pattern},
        )


@pytest_asyncio.fixture
async def postgres_context() -> AsyncIterator[PostgresTestContext]:
    engine = await _engine_or_skip()
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    context = PostgresTestContext(
        session_factory=session_factory,
        repository=_repository(session_factory),
        seed_repository=_seed_repository(session_factory),
        suffix=uuid.uuid4().hex[:10],
    )

    try:
        yield context
    finally:
        await _delete_postgres_test_rows(context.session_factory, context.suffix)
        await engine.dispose()


def _seed_config(
    suffix: str,
    *,
    requests_per_minute: int = 10,
    tokens_per_minute: int = 1_000,
    requests_per_day: int = 100,
    models: tuple[str, ...] | None = None,
) -> SeedConfig:
    model_names = models or (_model(suffix),)
    return SeedConfig(
        projects=[
            SeedProject(
                label=f"pg-project-{suffix}",
                owner_name="Postgres Test",
                model_limits=[
                    SeedModelLimit(
                        model=model,
                        requests_per_minute=requests_per_minute,
                        tokens_per_minute=tokens_per_minute,
                        requests_per_day=requests_per_day,
                    )
                    for model in model_names
                ],
            )
        ],
        api_keys=[
            SeedApiKey(
                project_label=f"pg-project-{suffix}",
                label=f"pg-key-{suffix}",
                api_key=f"AIza-postgres-test-{suffix}",
            )
        ],
        proxies=[
            SeedProxy(
                label=f"pg-proxy-{suffix}",
                host=f"proxy-{suffix}.local",
                port=8080,
                username=f"user-{suffix}",
                password=f"pass-{suffix}",
            )
        ],
        bindings=[
            SeedBinding(
                label=f"pg-route-{suffix}",
                project_label=f"pg-project-{suffix}",
                api_key_label=f"pg-key-{suffix}",
                proxy_label=f"pg-proxy-{suffix}",
            )
        ],
    )


def _pool_seed_config(suffix: str, *, route_count: int, model: str | None = None) -> SeedConfig:
    model_name = model or _model(suffix)
    return SeedConfig(
        projects=[
            SeedProject(
                label=f"pg-project-{index}-{suffix}",
                owner_name="Postgres Pool Test",
                model_limits=[
                    SeedModelLimit(
                        model=model_name,
                        requests_per_minute=10,
                        tokens_per_minute=10_000,
                        requests_per_day=100,
                    )
                ],
            )
            for index in range(route_count)
        ],
        api_keys=[
            SeedApiKey(
                project_label=f"pg-project-{index}-{suffix}",
                label=f"pg-key-{index}-{suffix}",
                api_key=f"AIza-postgres-pool-{index}-{suffix}",
            )
            for index in range(route_count)
        ],
        proxies=[
            SeedProxy(
                label=f"pg-proxy-{index}-{suffix}",
                host=f"proxy-{index}-{suffix}.local",
                port=8080,
                username=f"user-{index}-{suffix}",
                password=f"pass-{index}-{suffix}",
            )
            for index in range(route_count)
        ],
        bindings=[
            SeedBinding(
                label=f"pg-route-{index}-{suffix}",
                project_label=f"pg-project-{index}-{suffix}",
                api_key_label=f"pg-key-{index}-{suffix}",
                proxy_label=f"pg-proxy-{index}-{suffix}",
            )
            for index in range(route_count)
        ],
    )


def _request(suffix: str, *, estimated_tokens: int = 100, model: str | None = None) -> GatewayChatRequest:
    return GatewayChatRequest(
        request_id=f"pg-test-{suffix}-{uuid.uuid4().hex}",
        source_service="test",
        model=model or _model(suffix),
        messages=[{"role": "user", "content": "hello"}],
        estimated_input_tokens=estimated_tokens,
    )


def _model(suffix: str) -> str:
    return f"google/gemini-postgres-test-{suffix}"


@pytest.mark.asyncio
async def test_postgres_attempt_stores_request_metadata(postgres_context: PostgresTestContext) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))
    request = _request(suffix)
    request.chat_id = -1002450899820
    request.telegram_message_id = 89522

    await repository.acquire_route(request)

    async with session_factory() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT source_service, chat_id, telegram_message_id
                    FROM gemini_gateway.route_attempts
                    WHERE request_id = :request_id
                    """
                ),
                {"request_id": request.request_id},
            )
        ).mappings().one()

    assert dict(row) == {
        "source_service": "test",
        "chat_id": -1002450899820,
        "telegram_message_id": 89522,
    }


@pytest.mark.asyncio
async def test_postgres_attempt_stores_gateway_retry_count(postgres_context: PostgresTestContext) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))
    request = _request(suffix)
    request.retry_count = 3

    await repository.acquire_route(request)

    async with session_factory() as session:
        retry_count = await session.scalar(
            text(
                """
                SELECT retry_count
                FROM gemini_gateway.route_attempts
                WHERE request_id = :request_id
                """
            ),
            {"request_id": request.request_id},
        )

    assert retry_count == 3


@pytest.mark.asyncio
async def test_postgres_acquire_route_does_not_reuse_binding_inside_soybob_request(
    postgres_context: PostgresTestContext,
) -> None:
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    group_id = f"pg-pool-group-{suffix}"
    await postgres_context.seed_repository.upsert_seed_config(_pool_seed_config(suffix, route_count=2))

    first_request = _request(suffix)
    first_request.soybob_request_id = group_id
    second_request = _request(suffix)
    second_request.soybob_request_id = group_id
    overflow_request = _request(suffix)
    overflow_request.soybob_request_id = group_id

    first_lease = await repository.acquire_route(first_request)
    second_lease = await repository.acquire_route(second_request)

    with pytest.raises(GatewayError) as exc_info:
        await repository.acquire_route(overflow_request)

    assert second_lease.binding_id != first_lease.binding_id
    assert exc_info.value.reason == "no_route"


@pytest.mark.asyncio
async def test_postgres_route_candidates_do_not_decrypt_secrets_before_lease(
    postgres_context: PostgresTestContext,
) -> None:
    counting_vault = _CountingSecretVault()
    repository = PostgresGatewayRepository(
        session_factory=postgres_context.session_factory,
        secret_vault=counting_vault,  # type: ignore[arg-type]
    )
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))

    [candidate] = await repository.list_route_candidates(_model(suffix), datetime.now(tz=UTC))

    assert counting_vault.decrypt_calls == 0
    assert candidate.api_key == ""
    assert candidate.proxy_url is None

    lease = await repository.acquire_route(_request(suffix))

    assert counting_vault.decrypt_calls == 3
    assert lease.api_key == f"AIza-postgres-test-{suffix}"
    assert lease.proxy_url == f"http://user-{suffix}:pass-{suffix}@proxy-{suffix}.local:8080"


@pytest.mark.asyncio
async def test_postgres_concurrent_lease_reserves_single_request_window(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix, requests_per_minute=1))

    async def acquire() -> Any:
        try:
            return await repository.acquire_route(_request(suffix))
        except GatewayError as exc:
            return exc

    results = await asyncio.gather(acquire(), acquire())

    leases = [result for result in results if not isinstance(result, GatewayError)]
    errors = [result for result in results if isinstance(result, GatewayError)]
    assert len(leases) == 1
    assert len(errors) == 1
    assert errors[0].reason == "quota_exhausted"
    assert errors[0].error_code == "quota_exhausted"
    assert errors[0].quota_scope == "minute"

    async with session_factory() as session:
        no_route_attempt = (
            await session.execute(
                text(
                    """
                    SELECT status, error_type, error_message_safe, retryable
                    FROM gemini_gateway.route_attempts
                    WHERE status = 'skipped_no_route'
                      AND model = :model
                    """
                ),
                {"model": _model(suffix)},
            )
        ).mappings().one()
        rows = (
            await session.execute(
                text(
                    """
                    SELECT window_kind, requests_used
                    FROM gemini_gateway.quota_windows qw
                    JOIN gemini_gateway.google_projects gp ON gp.id = qw.project_id
                    WHERE gp.label = :project_label
                    ORDER BY window_kind
                    """
                ),
                {"project_label": f"pg-project-{suffix}"},
            )
        ).mappings().all()

    assert dict(no_route_attempt) == {
        "status": "skipped_no_route",
        "error_type": "quota_exhausted",
        "error_message_safe": public_message_for_reason("quota_exhausted"),
        "retryable": True,
    }
    assert {row["window_kind"]: row["requests_used"] for row in rows} == {"day": 1, "minute": 1}


@pytest.mark.asyncio
async def test_postgres_reports_quota_exhausted_when_day_quota_is_exhausted(
    postgres_context: PostgresTestContext,
) -> None:
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix, requests_per_day=1))

    await repository.acquire_route(_request(suffix))

    with pytest.raises(GatewayError) as exc_info:
        await repository.acquire_route(_request(suffix))

    assert exc_info.value.reason == "quota_exhausted"
    assert exc_info.value.error_code == "quota_exhausted"
    assert exc_info.value.quota_scope == "day"
    assert exc_info.value.quota_reset_at is not None
    assert exc_info.value.eligible_routes_count == 1
    assert exc_info.value.exhausted_routes_count == 1
    assert exc_info.value.retry_after_seconds is not None
    assert 0 < exc_info.value.retry_after_seconds <= 24 * 60 * 60


@pytest.mark.asyncio
async def test_postgres_skipped_no_route_stores_gateway_retry_count(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix, requests_per_day=1))

    await repository.acquire_route(_request(suffix))
    request = _request(suffix)
    request.retry_count = 5

    with pytest.raises(GatewayError):
        await repository.acquire_route(request)

    async with session_factory() as session:
        retry_count = await session.scalar(
            text(
                """
                SELECT retry_count
                FROM gemini_gateway.route_attempts
                WHERE request_id = :request_id
                  AND status = 'skipped_no_route'
                """
            ),
            {"request_id": request.request_id},
        )

    assert retry_count == 5


@pytest.mark.asyncio
async def test_postgres_complete_stale_leases_marks_attempt_and_releases_reservation(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix, tokens_per_minute=10_000))

    request = _request(suffix, estimated_tokens=123)
    lease = await repository.acquire_route(request)
    async with session_factory() as session, session.begin():
        stale_times = (
            await session.execute(
                text(
                    """
                    UPDATE gemini_gateway.route_attempts
                    SET created_at = TIMESTAMPTZ '2000-01-01 00:00:00+00'
                    WHERE id = :attempt_id
                    RETURNING date_trunc('minute', created_at) AS minute_start,
                              date_trunc('day', created_at) AS day_start
                    """
                ),
                {"attempt_id": lease.attempt_id},
            )
        ).mappings().one()
        await session.execute(
            text(
                """
                UPDATE gemini_gateway.quota_windows
                SET window_start = CASE
                    WHEN window_kind = 'minute' THEN CAST(:minute_start AS timestamptz)
                    ELSE CAST(:day_start AS timestamptz)
                END
                WHERE project_id = :project_id
                  AND model = :model
                """
            ),
            {
                "project_id": lease.project_id,
                "model": lease.model,
                "minute_start": stale_times["minute_start"],
                "day_start": stale_times["day_start"],
            },
        )

    completed_count = await repository.complete_stale_leases(stale_after_seconds=3600, limit=1)

    async with session_factory() as session:
        attempt = (
            await session.execute(
                text(
                    """
                    SELECT status, error_type, error_message_safe, retryable
                    FROM gemini_gateway.route_attempts
                    WHERE id = :attempt_id
                    """
                ),
                {"attempt_id": lease.attempt_id},
            )
        ).mappings().one()
        quota_rows = (
            await session.execute(
                text(
                    """
                    SELECT window_kind, requests_used, input_tokens_reserved
                    FROM gemini_gateway.quota_windows
                    WHERE project_id = :project_id
                      AND model = :model
                    ORDER BY window_kind
                    """
                ),
                {"project_id": lease.project_id, "model": lease.model},
            )
        ).mappings().all()

    assert completed_count == 1
    assert dict(attempt) == {
        "status": "failed",
        "error_type": "stale_lease",
        "error_message_safe": "Gateway route lease expired before completion.",
        "retryable": True,
    }
    assert {
        row["window_kind"]: (row["requests_used"], row["input_tokens_reserved"])
        for row in quota_rows
    } == {"day": (0, 0), "minute": (0, 0)}


@pytest.mark.asyncio
async def test_postgres_expire_elapsed_cooldowns_marks_only_elapsed_active_rows(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    model = _model(suffix)
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))

    async with session_factory() as session, session.begin():
        route_ids = (
            await session.execute(
                text(
                    """
                    SELECT p.id AS project_id, k.id AS api_key_id
                    FROM gemini_gateway.google_projects p
                    JOIN gemini_gateway.api_keys k ON k.project_id = p.id
                    WHERE p.label = :project_label
                      AND k.label = :key_label
                    """
                ),
                {
                    "project_label": f"pg-project-{suffix}",
                    "key_label": f"pg-key-{suffix}",
                },
            )
        ).mappings().one()
        await session.execute(
            text(
                """
                INSERT INTO gemini_gateway.cooldowns (
                    scope, scope_key, model, error_type, cooldown_level, sleep_until, half_open_after, status
                )
                VALUES
                    (
                        'project_model',
                        :project_id,
                        :model,
                        'network_timeout',
                        1,
                        TIMESTAMPTZ '1000-01-01 00:00:00+00',
                        TIMESTAMPTZ '1000-01-01 00:00:00+00',
                        'active'
                    ),
                    (
                        'provider_model',
                        :model,
                        NULL,
                        'provider_unavailable',
                        1,
                        now() + interval '1 day',
                        now() + interval '1 day',
                        'active'
                    ),
                    (
                        'key',
                        :api_key_id,
                        NULL,
                        'network_timeout',
                        1,
                        TIMESTAMPTZ '1900-01-01 00:00:00+00',
                        TIMESTAMPTZ '1900-01-01 00:00:00+00',
                        'cleared'
                    )
                """
            ),
            {
                "project_id": str(route_ids["project_id"]),
                "api_key_id": str(route_ids["api_key_id"]),
                "model": model,
            },
        )

    expired_count = await repository.expire_elapsed_cooldowns(limit=1)

    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT scope, status
                    FROM gemini_gateway.cooldowns
                    WHERE (
                           scope = 'project_model'
                           AND scope_key = :project_id
                           AND model = :model
                       )
                       OR (
                           scope = 'provider_model'
                           AND scope_key = :model
                       )
                       OR (
                           scope = 'key'
                           AND scope_key = :api_key_id
                       )
                    ORDER BY scope
                    """
                ),
                {
                    "project_id": str(route_ids["project_id"]),
                    "api_key_id": str(route_ids["api_key_id"]),
                    "model": model,
                },
            )
        ).mappings().all()

    assert expired_count == 1
    assert {row["scope"]: row["status"] for row in rows} == {
        "key": "cleared",
        "project_model": "expired",
        "provider_model": "active",
    }


@pytest.mark.asyncio
async def test_postgres_expire_elapsed_route_sleeps_clears_stale_binding_state(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))

    request = _request(suffix)
    lease = await repository.acquire_route(request)
    await repository.record_failure(
        lease,
        GatewayError(
            reason="proxy_failed",
            retryable=True,
            request_id=request.request_id,
        ),
        latency_ms=10,
        provider_called=False,
    )

    async with session_factory() as session, session.begin():
        await session.execute(
            text(
                """
                UPDATE gemini_gateway.key_proxy_bindings
                SET sleep_until = TIMESTAMPTZ '1000-01-01 00:00:00+00'
                WHERE id = :binding_id
                """
            ),
            {"binding_id": lease.binding_id},
        )

    expired_count = await repository.expire_elapsed_route_sleeps(limit=1)

    async with session_factory() as session:
        binding = (
            await session.execute(
                text(
                    """
                    SELECT cooldown_level, sleep_until, last_error_type
                    FROM gemini_gateway.key_proxy_bindings
                    WHERE id = :binding_id
                    """
                ),
                {"binding_id": lease.binding_id},
            )
        ).mappings().one()

    assert expired_count >= 1
    assert dict(binding) == {
        "cooldown_level": 0,
        "sleep_until": None,
        "last_error_type": "proxy_failed",
    }


@pytest.mark.asyncio
async def test_postgres_lease_rechecks_stale_binding_status_inside_transaction(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))

    request = _request(suffix)
    [candidate] = await repository.list_route_candidates(request.model, datetime.now(tz=UTC))

    async with session_factory() as session, session.begin():
        await session.execute(
            text("UPDATE gemini_gateway.key_proxy_bindings SET status = 'paused' WHERE id = :binding_id"),
            {"binding_id": candidate.binding_id},
        )

    lease = await repository._try_reserve_candidate(
        candidate=candidate,
        request=request,
        estimated_tokens=100,
        now=datetime.now(tz=UTC),
    )

    assert lease is None


@pytest.mark.asyncio
async def test_postgres_rate_limit_cooldown_is_model_scoped(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    main_model = f"google/gemini-postgres-main-{suffix}"
    gate_model = f"google/gemini-postgres-gate-{suffix}"
    await postgres_context.seed_repository.upsert_seed_config(
        _seed_config(suffix, models=(main_model, gate_model), tokens_per_minute=10_000)
    )

    main_request = _request(suffix, model=main_model)
    main_lease = await repository.acquire_route(main_request)
    await repository.record_failure(
        main_lease,
        GatewayError(
            reason="rate_limited",
            retryable=True,
            provider_status_code=429,
            request_id=main_request.request_id,
        ),
        latency_ms=10,
        provider_called=True,
    )

    async with session_factory() as session:
        binding = (
            await session.execute(
                text(
                    """
                    SELECT cooldown_level, sleep_until, consecutive_failures
                    FROM gemini_gateway.key_proxy_bindings
                    WHERE id = :binding_id
                    """
                ),
                {"binding_id": main_lease.binding_id},
            )
        ).mappings().one()
        cooldown = (
            await session.execute(
                text(
                    """
                    SELECT scope, scope_key, model, error_type, cooldown_level
                    FROM gemini_gateway.cooldowns
                    WHERE scope = 'project_model'
                      AND scope_key = :project_id
                      AND model = :model
                      AND status = 'active'
                    """
                ),
                {"project_id": str(main_lease.project_id), "model": main_model},
            )
        ).mappings().one()

    assert dict(binding) == {
        "cooldown_level": 0,
        "sleep_until": None,
        "consecutive_failures": 0,
    }
    assert dict(cooldown) == {
        "scope": "project_model",
        "scope_key": str(main_lease.project_id),
        "model": main_model,
        "error_type": "rate_limited",
        "cooldown_level": 1,
    }

    with pytest.raises(GatewayError) as exc_info:
        await repository.acquire_route(_request(suffix, model=main_model))
    assert exc_info.value.reason == "cooldown_active"
    assert exc_info.value.error_code == "cooldown_active"
    assert exc_info.value.sleep_until is not None
    assert exc_info.value.retry_after_seconds is not None
    assert exc_info.value.retry_after_seconds > 0

    gate_lease = await repository.acquire_route(_request(suffix, model=gate_model))
    assert gate_lease.binding_id == main_lease.binding_id
    assert gate_lease.model == gate_model


@pytest.mark.asyncio
async def test_postgres_network_timeout_cooldown_is_model_scoped(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    main_model = f"google/gemini-postgres-main-{suffix}"
    gate_model = f"google/gemini-postgres-gate-{suffix}"
    await postgres_context.seed_repository.upsert_seed_config(
        _seed_config(suffix, models=(main_model, gate_model), tokens_per_minute=10_000)
    )

    main_request = _request(suffix, model=main_model)
    main_lease = await repository.acquire_route(main_request)
    await repository.record_failure(
        main_lease,
        GatewayError(
            reason="network_timeout",
            retryable=True,
            provider_message_safe="read_timeout",
            request_id=main_request.request_id,
        ),
        latency_ms=30_000,
        provider_called=True,
    )

    async with session_factory() as session:
        binding = (
            await session.execute(
                text(
                    """
                    SELECT cooldown_level, sleep_until, consecutive_failures, last_error_type
                    FROM gemini_gateway.key_proxy_bindings
                    WHERE id = :binding_id
                    """
                ),
                {"binding_id": main_lease.binding_id},
            )
        ).mappings().one()
        cooldowns = [
            dict(row)
            for row in (
                await session.execute(
                    text(
                        """
                        SELECT scope, scope_key, model, error_type, cooldown_level
                        FROM gemini_gateway.cooldowns
                        WHERE status = 'active'
                          AND scope IN ('binding', 'proxy')
                          AND scope_key IN (:binding_id, :proxy_id)
                        ORDER BY scope
                        """
                    ),
                    {"binding_id": str(main_lease.binding_id), "proxy_id": str(main_lease.proxy_id)},
                )
            ).mappings()
        ]

    assert dict(binding) == {
        "cooldown_level": 0,
        "sleep_until": None,
        "consecutive_failures": 0,
        "last_error_type": None,
    }
    assert cooldowns == [
        {
            "scope": "binding",
            "scope_key": str(main_lease.binding_id),
            "model": main_model,
            "error_type": "network_timeout",
            "cooldown_level": 1,
        },
        {
            "scope": "proxy",
            "scope_key": str(main_lease.proxy_id),
            "model": main_model,
            "error_type": "network_timeout",
            "cooldown_level": 1,
        },
    ]

    with pytest.raises(GatewayError) as exc_info:
        await repository.acquire_route(_request(suffix, model=main_model))
    assert exc_info.value.reason == "cooldown_active"
    assert exc_info.value.error_code == "cooldown_active"
    assert exc_info.value.sleep_until is not None
    assert exc_info.value.retry_after_seconds is not None

    gate_lease = await repository.acquire_route(_request(suffix, model=gate_model))
    assert gate_lease.binding_id == main_lease.binding_id
    await repository.record_success(
        gate_lease,
        GatewayChatResponse(
            request_id=str(gate_lease.attempt_id),
            model=gate_model,
            choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        ),
        latency_ms=10,
    )

    async with session_factory() as session:
        active_main_cooldowns = await session.scalar(
            text(
                """
                SELECT COUNT(*)
                FROM gemini_gateway.cooldowns
                WHERE status = 'active'
                  AND model = :model
                  AND scope IN ('binding', 'proxy')
                  AND scope_key IN (:binding_id, :proxy_id)
                """
            ),
            {
                "model": main_model,
                "binding_id": str(main_lease.binding_id),
                "proxy_id": str(main_lease.proxy_id),
            },
        )

    assert active_main_cooldowns == 2


@pytest.mark.asyncio
async def test_postgres_provider_unavailable_cools_failed_binding_without_blocking_model_pool(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    main_model = f"google/gemini-postgres-main-{suffix}"
    await postgres_context.seed_repository.upsert_seed_config(
        _pool_seed_config(suffix, route_count=2, model=main_model)
    )

    main_request = _request(suffix, model=main_model)
    main_lease = await repository.acquire_route(main_request)
    await repository.record_failure(
        main_lease,
        GatewayError(
            reason="provider_unavailable",
            retryable=True,
            provider_status_code=503,
            request_id=main_request.request_id,
        ),
        latency_ms=10,
        provider_called=True,
    )

    async with session_factory() as session:
        binding = (
            await session.execute(
                text(
                    """
                    SELECT cooldown_level, sleep_until, consecutive_failures
                    FROM gemini_gateway.key_proxy_bindings
                    WHERE id = :binding_id
                    """
                ),
                {"binding_id": main_lease.binding_id},
            )
        ).mappings().one()
        cooldown = (
            await session.execute(
                text(
                    """
                    SELECT scope, scope_key, model, error_type, cooldown_level
                    FROM gemini_gateway.cooldowns
                    WHERE scope = 'binding'
                      AND scope_key = :binding_id
                      AND model IS NULL
                      AND status = 'active'
                    """
                ),
                {"binding_id": str(main_lease.binding_id)},
            )
        ).mappings().one()
        provider_model_cooldowns = await session.scalar(
            text(
                """
                SELECT COUNT(*)
                FROM gemini_gateway.cooldowns
                WHERE scope = 'provider_model'
                  AND scope_key = :model
                  AND model IS NULL
                  AND status = 'active'
                """
            ),
            {"model": main_model},
        )

    assert binding["cooldown_level"] == 1
    assert binding["sleep_until"] is not None
    assert binding["consecutive_failures"] == 1
    assert dict(cooldown) == {
        "scope": "binding",
        "scope_key": str(main_lease.binding_id),
        "model": None,
        "error_type": "provider_unavailable",
        "cooldown_level": 1,
    }
    assert provider_model_cooldowns == 0

    retry_lease = await repository.acquire_route(_request(suffix, model=main_model))
    assert retry_lease.binding_id != main_lease.binding_id
    assert retry_lease.model == main_model


@pytest.mark.asyncio
async def test_postgres_binding_cooldown_ladder_resets_when_error_reason_changes(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))

    request = _request(suffix)
    lease = await repository.acquire_route(request)
    provider_error = GatewayError(
        reason="provider_unavailable",
        retryable=True,
        request_id=request.request_id,
    )
    for _ in range(5):
        await repository.record_failure(lease, provider_error, latency_ms=10, provider_called=False)

    proxy_error = GatewayError(
        reason="proxy_failed",
        retryable=True,
        request_id=request.request_id,
    )
    await repository.record_failure(lease, proxy_error, latency_ms=10, provider_called=False)

    async with session_factory() as session:
        binding = (
            await session.execute(
                text(
                    """
                    SELECT cooldown_level, last_error_type
                    FROM gemini_gateway.key_proxy_bindings
                    WHERE id = :binding_id
                    """
                ),
                {"binding_id": lease.binding_id},
            )
        ).mappings().one()
        cooldown_levels = {
            row["scope"]: row["cooldown_level"]
            for row in (
                await session.execute(
                    text(
                        """
                        SELECT scope, cooldown_level
                        FROM gemini_gateway.cooldowns
                        WHERE status = 'active'
                          AND scope IN ('proxy', 'binding')
                          AND scope_key IN (:binding_id, :proxy_id)
                        """
                    ),
                    {"binding_id": str(lease.binding_id), "proxy_id": str(lease.proxy_id)},
                )
            ).mappings()
        }

    assert dict(binding) == {
        "cooldown_level": 1,
        "last_error_type": "proxy_failed",
    }
    assert cooldown_levels == {
        "binding": 1,
        "proxy": 1,
    }


@pytest.mark.asyncio
async def test_postgres_network_timeout_cooldown_ladder_ignores_elapsed_model_row(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))

    request = _request(suffix)
    lease = await repository.acquire_route(request)
    async with session_factory() as session, session.begin():
        await session.execute(
            text(
                """
                INSERT INTO gemini_gateway.cooldowns (
                    scope, scope_key, model, error_type, cooldown_level, sleep_until, half_open_after, status
                )
                VALUES (
                    'binding',
                    :binding_id,
                    :model,
                    'network_timeout',
                    4,
                    now() - INTERVAL '1 minute',
                    now() - INTERVAL '1 minute',
                    'active'
                )
                """
            ),
            {"binding_id": str(lease.binding_id), "model": lease.model},
        )

    error = GatewayError(reason="network_timeout", retryable=True, request_id=request.request_id)
    await repository.record_failure(lease, error, latency_ms=10, provider_called=False)

    async with session_factory() as session:
        binding = (
            await session.execute(
                text(
                    """
                    SELECT cooldown_level, sleep_until, consecutive_failures, last_error_type
                    FROM gemini_gateway.key_proxy_bindings
                    WHERE id = :binding_id
                    """
                ),
                {"binding_id": lease.binding_id},
            )
        ).mappings().one()
        cooldowns = {
            row["scope"]: {"model": row["model"], "cooldown_level": row["cooldown_level"]}
            for row in (
                await session.execute(
                    text(
                        """
                        SELECT scope, model, cooldown_level
                        FROM gemini_gateway.cooldowns
                        WHERE status = 'active'
                          AND scope IN ('proxy', 'binding')
                          AND scope_key IN (:binding_id, :proxy_id)
                        """
                    ),
                    {"binding_id": str(lease.binding_id), "proxy_id": str(lease.proxy_id)},
                )
            ).mappings()
        }

    assert dict(binding) == {
        "cooldown_level": 0,
        "sleep_until": None,
        "consecutive_failures": 0,
        "last_error_type": None,
    }
    assert cooldowns == {
        "binding": {"model": lease.model, "cooldown_level": 1},
        "proxy": {"model": lease.model, "cooldown_level": 1},
    }


@pytest.mark.asyncio
async def test_postgres_project_model_cooldown_ladder_ignores_elapsed_cooldown_row(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))

    request = _request(suffix)
    lease = await repository.acquire_route(request)
    async with session_factory() as session, session.begin():
        await session.execute(
            text(
                """
                INSERT INTO gemini_gateway.cooldowns (
                    scope, scope_key, model, error_type, cooldown_level, sleep_until, half_open_after, status
                )
                VALUES (
                    'project_model',
                    :project_id,
                    :model,
                    'rate_limited',
                    4,
                    now() - INTERVAL '1 minute',
                    now() - INTERVAL '1 minute',
                    'active'
                )
                ON CONFLICT DO NOTHING
                """
            ),
            {"project_id": str(lease.project_id), "model": lease.model},
        )

    error = GatewayError(reason="rate_limited", retryable=True, request_id=request.request_id)
    await repository.record_failure(lease, error, latency_ms=10, provider_called=True)

    async with session_factory() as session:
        cooldown_level = await session.scalar(
            text(
                """
                SELECT cooldown_level
                FROM gemini_gateway.cooldowns
                WHERE scope = 'project_model'
                  AND scope_key = :project_id
                  AND model = :model
                  AND status = 'active'
                """
            ),
            {"project_id": str(lease.project_id), "model": lease.model},
        )

    assert cooldown_level == 1


@pytest.mark.asyncio
async def test_postgres_transport_failure_updates_proxy_sleep_state(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))

    lease = await repository.acquire_route(_request(suffix))
    await repository.record_failure(
        lease,
        GatewayError(
            reason="proxy_failed",
            retryable=True,
            request_id=str(lease.attempt_id),
        ),
        latency_ms=10,
        provider_called=False,
    )

    async with session_factory() as session:
        proxy = (
            await session.execute(
                text(
                    """
                    SELECT cooldown_level, sleep_until, last_error_type
                    FROM gemini_gateway.proxy_endpoints
                    WHERE id = :proxy_id
                    """
                ),
                {"proxy_id": lease.proxy_id},
            )
        ).mappings().one()

    assert proxy["cooldown_level"] == 1
    assert proxy["sleep_until"] is not None
    assert proxy["last_error_type"] == "proxy_failed"


@pytest.mark.asyncio
async def test_postgres_transport_cooldowns_clear_on_route_success_after_sleep(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))

    failed_lease = await repository.acquire_route(_request(suffix))
    await repository.record_failure(
        failed_lease,
        GatewayError(
            reason="proxy_failed",
            retryable=True,
            request_id=str(failed_lease.attempt_id),
        ),
        latency_ms=10,
        provider_called=False,
    )
    async with session_factory() as session, session.begin():
        await session.execute(
            text(
                """
                UPDATE gemini_gateway.cooldowns
                SET sleep_until = TIMESTAMPTZ '1900-01-01 00:00:00+00',
                    half_open_after = TIMESTAMPTZ '1900-01-01 00:00:00+00'
                WHERE status = 'active'
                  AND (
                    (scope = 'binding' AND scope_key = :binding_id)
                    OR (scope = 'proxy' AND scope_key = :proxy_id)
                  )
                """
            ),
            {"binding_id": str(failed_lease.binding_id), "proxy_id": str(failed_lease.proxy_id)},
        )
        await session.execute(
            text(
                """
                UPDATE gemini_gateway.key_proxy_bindings
                SET sleep_until = TIMESTAMPTZ '1900-01-01 00:00:00+00'
                WHERE id = :binding_id
                """
            ),
            {"binding_id": failed_lease.binding_id},
        )
        await session.execute(
            text(
                """
                UPDATE gemini_gateway.proxy_endpoints
                SET sleep_until = TIMESTAMPTZ '1900-01-01 00:00:00+00'
                WHERE id = :proxy_id
                """
            ),
            {"proxy_id": failed_lease.proxy_id},
        )

    recovered_lease = await repository.acquire_route(_request(suffix))
    await repository.record_success(
        recovered_lease,
        GatewayChatResponse(
            request_id=str(recovered_lease.attempt_id),
            model=recovered_lease.model,
            choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        ),
        latency_ms=10,
    )

    async with session_factory() as session:
        cooldowns = {
            row["status"]: row["count"]
            for row in (
                await session.execute(
                    text(
                        """
                        SELECT status, COUNT(*) AS count
                        FROM gemini_gateway.cooldowns
                        WHERE scope IN ('binding', 'proxy')
                          AND scope_key IN (:binding_id, :proxy_id)
                        GROUP BY status
                        """
                    ),
                    {"binding_id": str(failed_lease.binding_id), "proxy_id": str(failed_lease.proxy_id)},
                )
            ).mappings()
        }

    assert cooldowns.get("active", 0) == 0
    assert cooldowns["cleared"] == 2


@pytest.mark.asyncio
async def test_postgres_failure_persists_public_error_message_without_provider_text(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))

    request = _request(suffix)
    lease = await repository.acquire_route(request)
    await repository.record_failure(
        lease,
        GatewayError(
            reason="auth_failed",
            retryable=False,
            provider_status_code=401,
            provider_message_safe="raw prompt fragment with api_key=AIza-secret",
            request_id=request.request_id,
        ),
        latency_ms=10,
        provider_called=True,
    )

    async with session_factory() as session:
        attempt_error_message = await session.scalar(
            text(
                """
                SELECT error_message_safe
                FROM gemini_gateway.route_attempts
                WHERE id = :attempt_id
                """
            ),
            {"attempt_id": lease.attempt_id},
        )
        key_error_message = await session.scalar(
            text(
                """
                SELECT last_error_message_safe
                FROM gemini_gateway.api_keys
                WHERE id = :api_key_id
                """
            ),
            {"api_key_id": lease.api_key_id},
        )

    assert attempt_error_message == public_message_for_reason("auth_failed")
    assert key_error_message == public_message_for_reason("auth_failed")


@pytest.mark.asyncio
async def test_postgres_failure_persists_safe_provider_diagnostics(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))

    request = _request(suffix)
    lease = await repository.acquire_route(request)
    await repository.record_failure(
        lease,
        GatewayError(
            reason="network_timeout",
            retryable=True,
            provider_status_code=408,
            provider_message_safe="read_timeout",
            retry_after_seconds=30,
            request_id=request.request_id,
        ),
        latency_ms=10,
        provider_called=False,
    )

    async with session_factory() as session:
        provider_response_json = await session.scalar(
            text(
                """
                SELECT provider_response_json
                FROM gemini_gateway.route_attempts
                WHERE id = :attempt_id
                """
            ),
            {"attempt_id": lease.attempt_id},
        )

    assert provider_response_json == {
        "reason": "network_timeout",
        "retryable": True,
        "provider_reason": "read_timeout",
        "provider_status_code": 408,
        "retry_after_seconds": 30,
    }


@pytest.mark.asyncio
async def test_postgres_rate_limit_cooldown_escalates_and_clears_on_success(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))

    request = _request(suffix)
    lease = await repository.acquire_route(request)
    error = GatewayError(
        reason="rate_limited",
        retryable=True,
        provider_status_code=429,
        request_id=request.request_id,
    )

    await repository.record_failure(lease, error, latency_ms=10, provider_called=True)
    await repository.record_failure(lease, error, latency_ms=10, provider_called=True)

    async with session_factory() as session:
        cooldown_level = await session.scalar(
            text(
                """
                SELECT cooldown_level
                FROM gemini_gateway.cooldowns
                WHERE scope = 'project_model'
                  AND scope_key = :project_id
                  AND model = :model
                  AND status = 'active'
                """
            ),
            {"project_id": str(lease.project_id), "model": lease.model},
        )

    assert cooldown_level == 2

    await repository.record_success(
        lease,
        GatewayChatResponse(
            request_id=request.request_id,
            model=request.model,
            choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            raw_response={"id": "gen-pg-rate-limit-success"},
        ),
        latency_ms=10,
    )

    async with session_factory() as session:
        active_cooldowns = await session.scalar(
            text(
                """
                SELECT COUNT(*)
                FROM gemini_gateway.cooldowns
                WHERE scope = 'project_model'
                  AND scope_key = :project_id
                  AND model = :model
                  AND status = 'active'
                """
            ),
            {"project_id": str(lease.project_id), "model": lease.model},
        )
        cleared_cooldowns = await session.scalar(
            text(
                """
                SELECT COUNT(*)
                FROM gemini_gateway.cooldowns
                WHERE scope = 'project_model'
                  AND scope_key = :project_id
                  AND model = :model
                  AND status = 'cleared'
                """
            ),
            {"project_id": str(lease.project_id), "model": lease.model},
        )

    assert active_cooldowns == 0
    assert cleared_cooldowns == 1


@pytest.mark.asyncio
async def test_postgres_success_reconciles_admission_budget_with_total_tokens(
    postgres_context: PostgresTestContext,
) -> None:
    session_factory = postgres_context.session_factory
    repository = postgres_context.repository
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix, tokens_per_minute=1_000))

    request = _request(suffix, estimated_tokens=100)
    lease = await repository.acquire_route(request)
    await repository.record_success(
        lease,
        GatewayChatResponse(
            request_id=request.request_id,
            model=request.model,
            choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            usage={"prompt_tokens": 20, "completion_tokens": 180, "total_tokens": 200},
            raw_response={"id": "gen-pg"},
        ),
        latency_ms=10,
    )

    async with session_factory() as session:
        reserved = await session.scalar(
            text(
                """
                SELECT input_tokens_reserved
                FROM gemini_gateway.quota_windows qw
                JOIN gemini_gateway.google_projects gp ON gp.id = qw.project_id
                WHERE gp.label = :project_label
                  AND qw.window_kind = 'minute'
                """
            ),
            {"project_label": f"pg-project-{suffix}"},
        )

    assert reserved == 200


@pytest.mark.asyncio
async def test_postgres_rejects_direct_route_without_proxy(postgres_context: PostgresTestContext) -> None:
    session_factory = postgres_context.session_factory
    suffix = postgres_context.suffix
    await postgres_context.seed_repository.upsert_seed_config(_seed_config(suffix))

    with pytest.raises(IntegrityError):
        async with session_factory() as session, session.begin():
            await session.execute(
                text(
                    """
                    UPDATE gemini_gateway.key_proxy_bindings
                    SET transport_mode = 'direct',
                        proxy_id = NULL,
                        updated_at = now()
                    WHERE label = :label
                    """
                ),
                {"label": f"pg-route-{suffix}"},
            )
