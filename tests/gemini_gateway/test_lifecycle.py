from __future__ import annotations

import asyncio
import logging

import pytest
import httpx

from gemini_gateway.errors import public_message_for_reason
from gemini_gateway.http_client_pool import GatewayHttpClientPool
from gemini_gateway.retention import GatewayRetentionService


@pytest.mark.asyncio
async def test_retention_service_completes_stale_leases_before_deleting_old_attempts() -> None:
    class _Repository:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, int | None]] = []

        async def expire_elapsed_cooldowns(self, *, limit: int) -> int:
            self.calls.append(("expire_elapsed_cooldowns", limit, None))
            return 3

        async def expire_elapsed_route_sleeps(self, *, limit: int) -> int:
            self.calls.append(("expire_elapsed_route_sleeps", limit, None))
            return 4

        async def complete_stale_leases(self, *, stale_after_seconds: int, limit: int) -> int:
            self.calls.append(("complete_stale_leases", stale_after_seconds, limit))
            return 2

        async def cleanup_route_attempts(self, ttl_days: int) -> int:
            self.calls.append(("cleanup_route_attempts", ttl_days, None))
            return 5

    repository = _Repository()
    service = GatewayRetentionService(
        repository=repository,
        ttl_days=7,
        interval_seconds=0.01,
        service_name="gemini-gateway",
        environment="test",
    )

    deleted = await service.run_once()

    assert deleted == 5
    assert repository.calls == [
        ("expire_elapsed_cooldowns", 100, None),
        ("expire_elapsed_route_sleeps", 100, None),
        ("complete_stale_leases", 3600, 100),
        ("cleanup_route_attempts", 7, None),
    ]


@pytest.mark.asyncio
async def test_retention_service_runs_periodically_and_stops_cleanly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Repository:
        def __init__(self) -> None:
            self.calls = 0

        async def cleanup_route_attempts(self, ttl_days: int) -> int:
            assert ttl_days == 7
            self.calls += 1
            return 0

    repository = _Repository()
    service = GatewayRetentionService(
        repository=repository,
        ttl_days=7,
        interval_seconds=0.01,
        service_name="gemini-gateway",
        environment="test",
    )
    stop_event = asyncio.Event()

    with caplog.at_level(logging.INFO, logger="gemini_gateway.retention"):
        task = asyncio.create_task(service.run_periodically(stop_event))
        await asyncio.sleep(0.035)
        stop_event.set()
        await asyncio.wait_for(task, timeout=1)

    assert repository.calls >= 2
    complete_records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "gemini_gateway_retention"
    ]
    assert complete_records
    assert complete_records[0].service == "gemini-gateway"
    assert complete_records[0].environment == "test"
    assert complete_records[0].status == "success"
    assert complete_records[0].ttl_days == 7
    assert complete_records[0].interval_seconds == 0.01
    assert complete_records[0].expired_cooldowns == 0
    assert complete_records[0].expired_route_sleeps == 0
    assert complete_records[0].deleted_route_attempts == 0


@pytest.mark.asyncio
async def test_retention_service_logs_safe_failure_without_raw_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FailingRepository:
        def __init__(self) -> None:
            self.calls = 0

        async def cleanup_route_attempts(self, ttl_days: int) -> int:
            assert ttl_days == 7
            self.calls += 1
            raise RuntimeError("gateway retention failed SECRET_TOKEN")

    repository = _FailingRepository()
    service = GatewayRetentionService(
        repository=repository,
        ttl_days=7,
        interval_seconds=0.01,
        service_name="gemini-gateway",
        environment="test",
    )
    stop_event = asyncio.Event()

    async def stop_soon() -> None:
        await asyncio.sleep(0.02)
        stop_event.set()

    stopper = asyncio.create_task(stop_soon())
    with caplog.at_level(logging.WARNING, logger="gemini_gateway.retention"):
        await service.run_periodically(stop_event)
    await asyncio.wait_for(stopper, timeout=1)

    assert repository.calls >= 1
    failure_records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "gemini_gateway_retention_failed"
    ]
    assert failure_records
    assert failure_records[0].service == "gemini-gateway"
    assert failure_records[0].environment == "test"
    assert failure_records[0].status == "error"
    assert failure_records[0].reason == "cleanup_failed"
    assert failure_records[0].retryable is True
    assert failure_records[0].failed_stage == "retention_cleanup"
    assert failure_records[0].ttl_days == 7
    assert failure_records[0].interval_seconds == 0.01
    assert failure_records[0].error_type == "RuntimeError"
    assert failure_records[0].error_message == public_message_for_reason("request_failed")
    assert "SECRET_TOKEN" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
async def test_gateway_http_client_pool_reuses_clients_by_proxy_and_closes_them() -> None:
    pool = GatewayHttpClientPool(timeout=1.0, transport=httpx.MockTransport(lambda request: httpx.Response(200)))

    first = pool.get(proxy_url="http://proxy-a")
    second = pool.get(proxy_url="http://proxy-a")

    assert first is second
    await pool.aclose()
    assert first.is_closed is True


def test_gateway_http_client_pool_rejects_missing_proxy_url() -> None:
    pool = GatewayHttpClientPool(timeout=1.0, transport=httpx.MockTransport(lambda request: httpx.Response(200)))

    with pytest.raises(ValueError, match="proxy_url is required"):
        pool.get(proxy_url=None)
