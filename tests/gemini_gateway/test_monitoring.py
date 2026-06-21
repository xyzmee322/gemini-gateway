from __future__ import annotations

from typing import Any

import pytest

from gemini_gateway.monitoring import (
    MonitorWindow,
    fetch_proxy_summary,
    fetch_proxy_timeseries,
    summarize_proxy_overview_rows,
    summarize_proxy_timeseries_rows,
)


class _EmptyResult:
    def mappings(self) -> "_EmptyResult":
        return self

    def all(self) -> list[dict[str, Any]]:
        return []


class _CapturedSession:
    def __init__(self) -> None:
        self.statements: list[str] = []

    async def __aenter__(self) -> "_CapturedSession":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def execute(self, statement: Any, params: dict[str, Any]) -> _EmptyResult:
        del params
        self.statements.append(str(statement))
        return _EmptyResult()


class _CapturedSessionFactory:
    def __init__(self) -> None:
        self.session = _CapturedSession()

    def __call__(self) -> _CapturedSession:
        return self.session


def test_summarize_proxy_overview_rows_returns_safe_proxy_metrics() -> None:
    rows = [
        {
            "proxy_id": 1,
            "proxy_label": "proxy-a",
            "route_label": "route-a",
            "total_requests": 2,
            "success_count": 1,
            "failed_count": 1,
            "read_timeout_count": 1,
            "p95_latency_ms": 92010,
            "avg_provider_total_ms": 120,
            "p95_provider_total_ms": 180,
            "avg_request_prepare_ms": 10,
            "p95_request_prepare_ms": 11,
            "avg_response_headers_ms": 20,
            "p95_response_headers_ms": 21,
            "avg_response_body_ms": 80,
            "p95_response_body_ms": 120,
            "avg_response_parse_ms": 6,
            "p95_response_parse_ms": 8,
        },
        {
            "proxy_id": 2,
            "proxy_label": "proxy-b",
            "route_label": "route-b",
            "total_requests": 1,
            "success_count": 1,
            "failed_count": 0,
            "read_timeout_count": 0,
            "p95_latency_ms": 120,
        },
    ]

    summary = summarize_proxy_overview_rows(rows)

    assert summary["total_requests"] == 3
    assert summary["proxy_count"] == 2
    assert summary["proxies"][0]["failure_rate"] == 0.5
    assert summary["proxies"][0]["read_timeout_count"] == 1
    assert summary["proxies"][0]["max_route_p95_latency_ms"] == 92010
    assert "p95_latency_ms" not in summary["proxies"][0]
    assert summary["proxies"][0]["routes"][0]["p95_latency_ms"] == 92010
    assert summary["proxies"][0]["avg_provider_total_ms"] == 120
    assert summary["proxies"][0]["max_route_p95_provider_total_ms"] == 180
    assert summary["proxies"][0]["avg_request_prepare_ms"] == 10
    assert summary["proxies"][0]["max_route_p95_request_prepare_ms"] == 11
    assert summary["proxies"][0]["avg_response_headers_ms"] == 20
    assert summary["proxies"][0]["max_route_p95_response_headers_ms"] == 21
    assert summary["proxies"][0]["avg_response_body_ms"] == 80
    assert summary["proxies"][0]["max_route_p95_response_body_ms"] == 120
    assert summary["proxies"][0]["avg_response_parse_ms"] == 6
    assert summary["proxies"][0]["max_route_p95_response_parse_ms"] == 8


@pytest.mark.asyncio
async def test_fetch_proxy_summary_ignores_elapsed_active_proxy_cooldowns() -> None:
    session_factory = _CapturedSessionFactory()

    await fetch_proxy_summary(session_factory, MonitorWindow())

    assert len(session_factory.session.statements) == 1
    assert "AND sleep_until > now()" in session_factory.session.statements[0]
    assert "CAST(:model AS text) IS NULL" in session_factory.session.statements[0]
    assert "CAST(:proxy_label AS text) IS NULL" in session_factory.session.statements[0]


def test_summarize_proxy_timeseries_rows_includes_stage_timing_metrics() -> None:
    summary = summarize_proxy_timeseries_rows(
        [
            {
                "bucket_start": "2026-06-20T10:00:00Z",
                "proxy_label": "proxy-a",
                "route_label": "route-a",
                "total_requests": 1,
                "success_count": 1,
                "failed_count": 0,
                "avg_provider_total_ms": 130,
                "p95_provider_total_ms": 170,
                "avg_request_prepare_ms": 9,
                "p95_request_prepare_ms": 12,
                "avg_response_headers_ms": 19,
                "p95_response_headers_ms": 22,
                "avg_response_body_ms": 70,
                "p95_response_body_ms": 99,
                "avg_response_parse_ms": 5,
                "p95_response_parse_ms": 7,
            }
        ],
        bucket_seconds=60,
    )

    point = summary["series"][0]
    assert point["avg_provider_total_ms"] == 130
    assert point["p95_provider_total_ms"] == 170
    assert point["avg_request_prepare_ms"] == 9
    assert point["p95_request_prepare_ms"] == 12
    assert point["avg_response_headers_ms"] == 19
    assert point["p95_response_headers_ms"] == 22
    assert point["avg_response_body_ms"] == 70
    assert point["p95_response_body_ms"] == 99
    assert point["avg_response_parse_ms"] == 5
    assert point["p95_response_parse_ms"] == 7


@pytest.mark.asyncio
async def test_fetch_proxy_timeseries_ignores_elapsed_active_proxy_cooldowns() -> None:
    session_factory = _CapturedSessionFactory()

    await fetch_proxy_timeseries(session_factory, MonitorWindow())

    assert len(session_factory.session.statements) == 1
    assert "AND sleep_until > now()" in session_factory.session.statements[0]
    assert "CAST(:model AS text) IS NULL" in session_factory.session.statements[0]
    assert "CAST(:proxy_label AS text) IS NULL" in session_factory.session.statements[0]


def test_summarize_proxy_overview_rows_names_proxy_p95_timing_as_max_route_p95() -> None:
    summary = summarize_proxy_overview_rows(
        [
            {
                "proxy_id": 1,
                "proxy_label": "proxy-a",
                "route_label": "route-a",
                "total_requests": 2,
                "success_count": 2,
                "failed_count": 0,
                "p95_provider_total_ms": 180,
                "p95_request_prepare_ms": 18,
                "p95_response_headers_ms": 28,
                "p95_response_body_ms": 120,
                "p95_response_parse_ms": 12,
            },
            {
                "proxy_id": 1,
                "proxy_label": "proxy-a",
                "route_label": "route-b",
                "total_requests": 1,
                "success_count": 1,
                "failed_count": 0,
                "p95_provider_total_ms": 240,
                "p95_request_prepare_ms": 16,
                "p95_response_headers_ms": 35,
                "p95_response_body_ms": 90,
                "p95_response_parse_ms": 15,
            },
        ]
    )

    proxy = summary["proxies"][0]
    assert proxy["max_route_p95_provider_total_ms"] == 240
    assert proxy["max_route_p95_request_prepare_ms"] == 18
    assert proxy["max_route_p95_response_headers_ms"] == 35
    assert proxy["max_route_p95_response_body_ms"] == 120
    assert proxy["max_route_p95_response_parse_ms"] == 15
    assert "p95_provider_total_ms" not in proxy
    assert "p95_request_prepare_ms" not in proxy
    assert "p95_response_headers_ms" not in proxy
    assert "p95_response_body_ms" not in proxy
    assert "p95_response_parse_ms" not in proxy
    assert proxy["routes"][0]["p95_provider_total_ms"] == 180
    assert proxy["routes"][1]["p95_provider_total_ms"] == 240
