from __future__ import annotations

import asyncio

from core.logger import configure_logging
from gemini_gateway.api import create_app
from gemini_gateway.config import get_gateway_settings
from gemini_gateway.cooldown import CooldownPolicy
from gemini_gateway.db.session import create_gateway_engine
from gemini_gateway.embedding_client import GeminiEmbeddingClient
from gemini_gateway.gemini_client import GeminiOpenAIClient
from gemini_gateway.http_client_pool import GatewayHttpClientPool
from gemini_gateway.repository import PostgresGatewayRepository
from gemini_gateway.retention import GatewayRetentionService
from gemini_gateway.security import SecretVault
from gemini_gateway.service import CompletionService
from gemini_gateway.tts_client import GeminiTTSClient
from sqlalchemy.ext.asyncio import async_sessionmaker


def build_app():
    """Создает FastAPI app для запуска через uvicorn."""

    settings = get_gateway_settings()
    configure_logging(settings.service_name)

    engine = create_gateway_engine(settings)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    vault = SecretVault(encryption_key=settings.encryption_key, hmac_key=settings.hmac_key)
    repository = PostgresGatewayRepository(
        session_factory=session_factory,
        secret_vault=vault,
        cooldown_policy=CooldownPolicy(),
    )
    http_client_pool = GatewayHttpClientPool(timeout=settings.default_request_timeout_seconds)
    gemini_client = GeminiOpenAIClient(
        base_url=settings.upstream_base_url,
        timeout=settings.default_request_timeout_seconds,
        client_pool=http_client_pool,
    )
    tts_client = GeminiTTSClient(
        base_url=_native_gemini_base_url(settings.upstream_base_url),
        timeout=settings.default_request_timeout_seconds,
        client_pool=http_client_pool,
    )
    embedding_client = GeminiEmbeddingClient(
        base_url=_native_gemini_base_url(settings.upstream_base_url),
        timeout=settings.default_request_timeout_seconds,
        client_pool=http_client_pool,
    )
    service = CompletionService(
        repository=repository,
        gemini_client=gemini_client,
        tts_client=tts_client,
        embedding_client=embedding_client,
        service_name=settings.service_name,
        environment=settings.environment,
    )
    app = create_app(
        auth_token=settings.internal_auth_token.get_secret_value(),
        completion_service=service,
        readiness_check=lambda: service.health_check(require_routes=settings.require_seeded_routes),
        service_name=settings.service_name,
        environment=settings.environment,
    )
    retention_service = GatewayRetentionService(
        repository=repository,
        ttl_days=settings.route_attempts_ttl_days,
        interval_seconds=settings.retention_interval_seconds,
        service_name=settings.service_name,
        environment=settings.environment,
    )
    retention_stop_event = asyncio.Event()
    retention_task: asyncio.Task[None] | None = None

    @app.on_event("startup")
    async def start_retention() -> None:
        nonlocal retention_task
        retention_task = asyncio.create_task(retention_service.run_periodically(retention_stop_event))

    @app.on_event("shutdown")
    async def dispose_engine() -> None:
        retention_stop_event.set()
        if retention_task is not None:
            await retention_task
        await http_client_pool.aclose()
        await engine.dispose()

    return app


def _native_gemini_base_url(upstream_base_url: str) -> str:
    return upstream_base_url.rstrip("/").removesuffix("/openai")


app = build_app()
