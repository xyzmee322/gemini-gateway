from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from gemini_gateway.errors import public_message_for_reason

_LOGGER = logging.getLogger(__name__)
_ELAPSED_COOLDOWN_CLEANUP_LIMIT = 100
_STALE_ROUTE_LEASE_SECONDS = 3600
_STALE_ROUTE_LEASE_CLEANUP_LIMIT = 100


class GatewayRetentionService:
    """Периодически очищает устаревшие записи gateway."""

    def __init__(
        self,
        *,
        repository: Any,
        ttl_days: int,
        interval_seconds: float,
        service_name: str = "gemini-gateway",
        environment: str = "development",
    ) -> None:
        self._repository = repository
        self._ttl_days = ttl_days
        self._interval_seconds = interval_seconds
        self._service_name = service_name
        self._environment = environment

    async def run_once(self) -> int:
        await self._expire_elapsed_cooldowns()
        await self._expire_elapsed_route_sleeps()
        await self._complete_stale_leases()
        return await self._cleanup_route_attempts()

    async def _expire_elapsed_cooldowns(self) -> int:
        expire_elapsed_cooldowns = getattr(self._repository, "expire_elapsed_cooldowns", None)
        if expire_elapsed_cooldowns is None:
            return 0
        expired = expire_elapsed_cooldowns(limit=_ELAPSED_COOLDOWN_CLEANUP_LIMIT)
        if inspect.isawaitable(expired):
            expired = await expired
        return int(expired or 0)

    async def _expire_elapsed_route_sleeps(self) -> int:
        expire_elapsed_route_sleeps = getattr(self._repository, "expire_elapsed_route_sleeps", None)
        if expire_elapsed_route_sleeps is None:
            return 0
        expired = expire_elapsed_route_sleeps(limit=_ELAPSED_COOLDOWN_CLEANUP_LIMIT)
        if inspect.isawaitable(expired):
            expired = await expired
        return int(expired or 0)

    async def _complete_stale_leases(self) -> int:
        complete_stale_leases = getattr(self._repository, "complete_stale_leases", None)
        if complete_stale_leases is None:
            return 0
        completed = complete_stale_leases(
            stale_after_seconds=_STALE_ROUTE_LEASE_SECONDS,
            limit=_STALE_ROUTE_LEASE_CLEANUP_LIMIT,
        )
        if inspect.isawaitable(completed):
            completed = await completed
        return int(completed or 0)

    async def _cleanup_route_attempts(self) -> int:
        deleted = self._repository.cleanup_route_attempts(self._ttl_days)
        if inspect.isawaitable(deleted):
            deleted = await deleted
        return int(deleted or 0)

    async def run_periodically(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                expired_cooldowns = await self._expire_elapsed_cooldowns()
                expired_route_sleeps = await self._expire_elapsed_route_sleeps()
                completed_stale_leases = await self._complete_stale_leases()
                deleted = await self._cleanup_route_attempts()
                _LOGGER.info(
                    "gemini_gateway_retention_complete",
                    extra={
                        "event": "gemini_gateway_retention",
                        "service": self._service_name,
                        "environment": self._environment,
                        "status": "success",
                        "ttl_days": self._ttl_days,
                        "interval_seconds": self._interval_seconds,
                        "expired_cooldowns": expired_cooldowns,
                        "expired_route_sleeps": expired_route_sleeps,
                        "completed_stale_leases": completed_stale_leases,
                        "deleted_route_attempts": deleted,
                    },
                )
            except Exception as exc:
                _LOGGER.warning(
                    "gemini_gateway_retention_failed",
                    extra={
                        "event": "gemini_gateway_retention_failed",
                        "service": self._service_name,
                        "environment": self._environment,
                        "status": "error",
                        "reason": "cleanup_failed",
                        "retryable": True,
                        "failed_stage": "retention_cleanup",
                        "ttl_days": self._ttl_days,
                        "interval_seconds": self._interval_seconds,
                        "error_type": type(exc).__name__,
                        "error_message": public_message_for_reason("request_failed"),
                    },
                )

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._interval_seconds)
            except TimeoutError:
                continue
