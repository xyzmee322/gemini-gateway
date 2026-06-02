from datetime import UTC, datetime
from typing import Any

import pytest

from gemini_gateway.contracts import SeedApiKey, SeedBinding, SeedConfig, SeedModelLimit, SeedProject, SeedProxy
from gemini_gateway.repository import PostgresGatewayRepository, PostgresGatewaySeedRepository, _RouteEligibilityQueryBuilder
from gemini_gateway.security import SecretVault

_ENCRYPTION_KEY = "fTypay4X7jiYzb-rZUnV74QuKmcRQev9i12cZrFI3E8="
_HMAC_KEY = "h" * 32


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


def _secret_vault() -> SecretVault:
    return SecretVault(encryption_key=_ENCRYPTION_KEY, hmac_key=_HMAC_KEY)


def test_route_eligibility_builder_reuses_reservable_predicates() -> None:
    builder = _RouteEligibilityQueryBuilder()

    candidate_where = _normalize_sql(builder.where_clause(extra_conditions=["ml.model = :model"]))
    locked_where = _normalize_sql(
        builder.where_clause(
            extra_conditions=[
                "b.id = :binding_id",
                "k.id = :api_key_id",
                "b.proxy_id IS NOT DISTINCT FROM CAST(:proxy_id AS bigint)",
                "p.id = :project_id",
                "ml.model = :model",
            ]
        )
    )
    health_where = _normalize_sql(builder.where_clause())

    for sql in (candidate_where, locked_where, health_where):
        assert "p.status = 'active'" in sql
        assert "(k.sleep_until IS NULL OR k.sleep_until <= :now)" in sql
        assert "(b.sleep_until IS NULL OR b.sleep_until <= :now)" in sql
        assert "gemini_gateway.cooldowns cd" in sql
        assert "(cd.scope = 'provider_model' AND cd.scope_key = ml.model)" in sql


class _HealthSession:
    def __init__(self) -> None:
        self.scalar_calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> "_HealthSession":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def execute(self, *_: object, **__: object) -> object:
        return object()

    async def scalar(self, statement: object, params: dict[str, Any] | None = None) -> int:
        sql = str(statement)
        payload = params or {}
        self.scalar_calls.append((sql, payload))

        if "information_schema.tables" in sql:
            return 7
        if "gemini_gateway.key_proxy_bindings" in sql:
            has_reservable_predicates = (
                "gemini_gateway.cooldowns" in sql
                and "sleep_until" in sql
                and isinstance(payload.get("now"), datetime)
            )
            return 0 if has_reservable_predicates else 1
        return 0


class _HealthSessionFactory:
    def __init__(self) -> None:
        self.session = _HealthSession()

    def __call__(self) -> _HealthSession:
        return self.session


class _SeedKeySession:
    def __init__(self) -> None:
        self.scalar_calls: list[tuple[str, dict[str, Any]]] = []

    async def scalar(self, statement: object, params: dict[str, Any] | None = None) -> int:
        sql = str(statement)
        payload = params or {}
        self.scalar_calls.append((sql, payload))
        if "WHERE key_fingerprint = :key_fingerprint" in sql:
            return 42
        return 99


class _SeedBindingSession:
    def __init__(self) -> None:
        self.scalar_calls: list[tuple[str, dict[str, Any]]] = []
        self.execute_calls: list[tuple[str, dict[str, Any]]] = []

    async def scalar(self, statement: object, params: dict[str, Any] | None = None) -> int:
        sql = str(statement)
        payload = params or {}
        self.scalar_calls.append((sql, payload))
        if "WHERE status = 'active'" in sql and "api_key_id = :api_key_id" in sql:
            return 77
        return 0

    async def execute(self, statement: object, params: dict[str, Any] | None = None) -> object:
        self.execute_calls.append((str(statement), params or {}))
        return object()


@pytest.mark.asyncio
async def test_health_check_require_routes_uses_reservable_route_filters() -> None:
    session_factory = _HealthSessionFactory()
    repository = PostgresGatewayRepository(
        session_factory=session_factory,  # type: ignore[arg-type]
        secret_vault=_secret_vault(),
    )

    health = await repository.health_check(require_routes=True)

    route_sql, route_params = session_factory.session.scalar_calls[-1]
    assert health["checks"]["routes"] is False
    assert "gemini_gateway.cooldowns" in route_sql
    assert "sleep_until" in route_sql
    assert route_params["now"].tzinfo is UTC


@pytest.mark.asyncio
async def test_health_check_schema_requires_cooldowns_table() -> None:
    session_factory = _HealthSessionFactory()
    repository = PostgresGatewayRepository(
        session_factory=session_factory,  # type: ignore[arg-type]
        secret_vault=_secret_vault(),
    )

    health = await repository.health_check(require_routes=False)

    schema_sql, _ = session_factory.session.scalar_calls[0]
    assert "'cooldowns'" in schema_sql
    assert health["checks"]["schema"] is False


@pytest.mark.asyncio
async def test_seed_key_upsert_reuses_existing_key_by_fingerprint() -> None:
    session = _SeedKeySession()
    repository = PostgresGatewaySeedRepository(
        session_factory=lambda: None,  # type: ignore[arg-type]
        secret_vault=_secret_vault(),
    )
    config = SeedConfig(
        projects=[
            SeedProject(
                label="local-dev",
                owner_name="Local Dev",
                model_limits=[
                    SeedModelLimit(
                        model="google/gemini-3.5-flash",
                        requests_per_minute=10,
                        tokens_per_minute=100000,
                        requests_per_day=1000,
                    )
                ],
            )
        ],
        api_keys=[
            SeedApiKey(
                project_label="local-dev",
                label="local-key",
                api_key="AIza-existing-key",
            )
        ],
        bindings=[
            SeedBinding(
                label="local-route",
                project_label="local-dev",
                api_key_label="local-key",
                proxy_label="proxy-a",
            )
        ],
        proxies=[
            SeedProxy(label="proxy-a", host="127.0.0.1", port=8080),
        ],
    )

    key_ids = await repository._upsert_keys(session, config, {"local-dev": 7})  # type: ignore[arg-type]

    assert key_ids == {("local-dev", "local-key"): 42}
    assert len(session.scalar_calls) == 1
    fingerprint_sql, fingerprint_params = session.scalar_calls[0]
    assert "UPDATE gemini_gateway.api_keys" in fingerprint_sql
    assert "WHERE key_fingerprint = :key_fingerprint" in fingerprint_sql
    assert fingerprint_params["project_id"] == 7


@pytest.mark.asyncio
async def test_seed_binding_upsert_reuses_existing_active_binding_by_key() -> None:
    session = _SeedBindingSession()
    repository = PostgresGatewaySeedRepository(
        session_factory=lambda: None,  # type: ignore[arg-type]
        secret_vault=_secret_vault(),
    )
    binding = SeedBinding(
        label="local-proxy-route",
        project_label="local-dev",
        api_key_label="local-key",
        proxy_label="local-proxy",
    )

    await repository._upsert_bindings(  # type: ignore[arg-type]
        session,
        [binding],
        {("local-dev", "local-key"): 42},
        {"local-proxy": 24},
    )

    assert len(session.scalar_calls) == 1
    assert session.execute_calls == []
    binding_sql, binding_params = session.scalar_calls[0]
    assert "UPDATE gemini_gateway.key_proxy_bindings" in binding_sql
    assert "WHERE status = 'active'" in binding_sql
    assert "api_key_id = :api_key_id" in binding_sql
    assert "CAST(:proxy_id AS bigint)" in binding_sql
    assert binding_params["api_key_id"] == 42
    assert binding_params["proxy_id"] == 24
