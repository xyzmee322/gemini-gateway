from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from gemini_gateway.config import GeminiGatewaySettings


def create_gateway_engine(settings: GeminiGatewaySettings) -> AsyncEngine:
    return create_async_engine(settings.postgres_dsn, pool_pre_ping=True, echo=False)


def create_gateway_session_factory(settings: GeminiGatewaySettings) -> async_sessionmaker[AsyncSession]:
    engine = create_gateway_engine(settings)
    return async_sessionmaker(bind=engine, expire_on_commit=False)

