from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
import re
from typing import Any

import yaml

from gemini_gateway.config import GeminiGatewaySettings, get_gateway_settings
from gemini_gateway.contracts import SeedConfig
from gemini_gateway.cooldown import CooldownPolicy
from gemini_gateway.db.session import create_gateway_session_factory
from gemini_gateway.repository import PostgresGatewaySeedRepository
from gemini_gateway.security import SecretVault

_ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_LOGGER = logging.getLogger(__name__)


class MissingSeedEnvironmentError(ValueError):
    """Ошибка отсутствующей переменной окружения seed-файла."""

    def __init__(self, variable_name: str) -> None:
        self.variable_name = variable_name
        super().__init__(f"seed environment variable is not set: {variable_name}")


def load_seed_config(path: Path) -> SeedConfig:
    """Читает seed-файл без вывода секретов в stdout."""

    with path.open("r", encoding="utf-8") as file:
        raw_data: Any = yaml.safe_load(file)
    if not isinstance(raw_data, dict):
        raise ValueError("seed file must contain an object")
    return SeedConfig.model_validate(_expand_seed_environment(raw_data))


def _expand_seed_environment(value: Any) -> Any:
    """Разворачивает ${ENV_VAR} в seed YAML без логирования секретов."""

    if isinstance(value, dict):
        return {key: _expand_seed_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_seed_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        environment_value = os.getenv(variable_name)
        if not environment_value:
            raise MissingSeedEnvironmentError(variable_name)
        return environment_value

    return _ENV_PLACEHOLDER_RE.sub(replace, value)


def _build_seed_repository(settings: GeminiGatewaySettings) -> PostgresGatewaySeedRepository:
    vault = SecretVault(encryption_key=settings.encryption_key, hmac_key=settings.hmac_key)
    return PostgresGatewaySeedRepository(
        session_factory=create_gateway_session_factory(settings),
        secret_vault=vault,
        cooldown_policy=CooldownPolicy(),
    )


def _health_has_routes(health: dict[str, Any]) -> bool:
    checks = health.get("checks")
    if not isinstance(checks, dict):
        return False
    return checks.get("routes") is True


async def _can_skip_missing_dev_seed_env(
    *,
    settings: GeminiGatewaySettings,
    repository: PostgresGatewaySeedRepository,
    error: MissingSeedEnvironmentError,
) -> bool:
    if settings.environment != "development":
        return False
    try:
        health = await repository.health_check(require_routes=True)
    except Exception as health_error:
        _LOGGER.warning(
            "gemini_gateway_seed_route_check_failed",
            extra={
                "missing_variable": error.variable_name,
                "error_type": type(health_error).__name__,
            },
        )
        return False
    return _health_has_routes(health)


async def import_seed_file(path: Path) -> None:
    settings = get_gateway_settings()
    repository = _build_seed_repository(settings)
    try:
        seed_config = load_seed_config(path)
    except MissingSeedEnvironmentError as error:
        if await _can_skip_missing_dev_seed_env(settings=settings, repository=repository, error=error):
            _LOGGER.warning(
                "gemini_gateway_seed_skipped_missing_env_existing_routes",
                extra={"missing_variable": error.variable_name},
            )
            return
        raise
    await repository.upsert_seed_config(seed_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Gemini Gateway seed config")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    asyncio.run(import_seed_file(args.path))


if __name__ == "__main__":
    main()
