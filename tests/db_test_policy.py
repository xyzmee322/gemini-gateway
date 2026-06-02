from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pytest


def default_test_postgres_dsn(
    env_var: str,
    *,
    env: Mapping[str, str] | None = None,
    inside_container: bool | None = None,
) -> str:
    """Возвращает DSN тестовой Postgres под host- и docker-запуск."""

    values = os.environ if env is None else env
    explicit_dsn = values.get(env_var)
    if explicit_dsn:
        return explicit_dsn

    if inside_container is None:
        inside_container = Path("/.dockerenv").exists()

    user = values.get("POSTGRES_USER") or "gemini_gateway"
    password = values.get("POSTGRES_PASSWORD") or "gemini_gateway"
    database = values.get("POSTGRES_DB") or "gemini_gateway"

    if inside_container:
        host = values.get("POSTGRES_HOST") or "postgres"
        port = values.get("POSTGRES_PORT") or "5432"
    else:
        host = "127.0.0.1"
        port = values.get("POSTGRES_BIND_PORT") or "5434"

    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"


def skip_or_fail_db_unavailable(reason: str) -> None:
    """В CI и обязательном DB-режиме недоступная БД должна быть явным провалом."""

    if os.getenv("RUN_DB_TESTS") == "1" or os.getenv("CI", "").lower() == "true":
        pytest.fail(reason)
    pytest.skip(reason)
