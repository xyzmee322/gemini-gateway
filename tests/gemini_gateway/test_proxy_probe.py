from __future__ import annotations

import json

import httpx
import pytest

from scripts.probe_gemini_gateway_proxies import (
    ProxyProbeTarget,
    _build_fetch_proxy_probe_targets_query,
    _parse_args,
    probe_proxy_targets,
)


@pytest.mark.asyncio
async def test_probe_proxy_targets_reports_success_without_proxy_secret() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    targets = [
        ProxyProbeTarget(
            route_label="Keyproxy4",
            key_label="Keyproxy4-key",
            proxy_label="Keyproxy4-proxy",
            key_status="active",
            binding_status="active",
            proxy_status="active",
            proxy_url="http://user:secret@127.0.0.1:8080",
        )
    ]

    report = await probe_proxy_targets(
        targets,
        target_url="https://generativelanguage.googleapis.com/",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )

    assert report["status"] == "ok"
    assert report["ok_count"] == 1
    assert report["error_count"] == 0
    assert report["results"][0]["status"] == "ok"
    assert report["results"][0]["http_status_family"] == "2xx"
    serialized = json.dumps(report, ensure_ascii=False)
    assert "proxy_url" not in serialized
    assert "user" not in serialized
    assert "secret" not in serialized
    assert "127.0.0.1" not in serialized


@pytest.mark.asyncio
async def test_probe_proxy_targets_maps_timeout_to_safe_reason() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("raw timeout details", request=request)

    targets = [
        ProxyProbeTarget(
            route_label="Keyproxy5",
            key_label="Keyproxy5-key",
            proxy_label="Keyproxy5-proxy",
            key_status="active",
            binding_status="active",
            proxy_status="active",
            proxy_url="http://user:secret@127.0.0.1:8080",
        )
    ]

    report = await probe_proxy_targets(
        targets,
        target_url="https://generativelanguage.googleapis.com/",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )

    assert report["status"] == "error"
    assert report["ok_count"] == 0
    assert report["error_count"] == 1
    assert report["reason_counts"] == {"network_timeout": 1}
    assert report["timeout_reason_counts"] == {"read_timeout": 1}
    assert report["results"][0]["reason"] == "network_timeout"
    assert report["results"][0]["timeout_reason"] == "read_timeout"
    assert "raw timeout details" not in json.dumps(report, ensure_ascii=False)


@pytest.mark.asyncio
async def test_probe_proxy_targets_maps_connect_timeout_to_safe_subtype() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("raw connect details", request=request)

    targets = [
        ProxyProbeTarget(
            route_label="Keyproxy8",
            key_label="Keyproxy8-key",
            proxy_label="Keyproxy8-proxy",
            key_status="active",
            binding_status="active",
            proxy_status="active",
            proxy_url="http://user:secret@127.0.0.1:8080",
        )
    ]

    report = await probe_proxy_targets(
        targets,
        target_url="https://generativelanguage.googleapis.com/",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )

    assert report["reason_counts"] == {"network_timeout": 1}
    assert report["timeout_reason_counts"] == {"connect_timeout": 1}
    assert report["results"][0]["reason"] == "network_timeout"
    assert report["results"][0]["timeout_reason"] == "connect_timeout"
    assert "raw connect details" not in json.dumps(report, ensure_ascii=False)


@pytest.mark.asyncio
async def test_probe_proxy_targets_rejects_missing_proxy_secret_without_http_call() -> None:
    called = False

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200)

    targets = [
        ProxyProbeTarget(
            route_label="Keyproxy6",
            key_label="Keyproxy6-key",
            proxy_label="Keyproxy6-proxy",
            key_status="active",
            binding_status="active",
            proxy_status="active",
            proxy_url=None,
        )
    ]

    report = await probe_proxy_targets(
        targets,
        target_url="https://generativelanguage.googleapis.com/",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )

    assert called is False
    assert report["status"] == "error"
    assert report["reason_counts"] == {"secret_unavailable": 1}
    assert report["results"][0]["reason"] == "secret_unavailable"


def test_fetch_proxy_probe_targets_query_can_filter_by_model_pool() -> None:
    query = str(
        _build_fetch_proxy_probe_targets_query(
            model="google/gemini-3.5-flash",
            only_active_routes=True,
            route_labels=None,
        )
    )

    assert "JOIN gemini_gateway.google_projects p ON p.id = k.project_id" in query
    assert "JOIN gemini_gateway.model_limits ml ON ml.project_id = p.id AND ml.model = :model" in query
    assert "AND p.status = 'active'" in query
    assert "AND ml.status = 'active'" in query


def test_fetch_proxy_probe_targets_query_can_filter_by_route_labels() -> None:
    query = str(
        _build_fetch_proxy_probe_targets_query(
            model="google/gemini-3.5-flash",
            only_active_routes=True,
            route_labels=("Keyproxy4", "Keyproxy8"),
        )
    )

    assert "AND b.label = ANY(CAST(:route_labels AS text[]))" in query


def test_parse_args_accepts_model_filter() -> None:
    args = _parse_args(["--model", "google/gemini-3.1-flash-lite"])

    assert args.model == "google/gemini-3.1-flash-lite"


def test_parse_args_accepts_route_label_filter() -> None:
    args = _parse_args(["--route-label", "Keyproxy4", "--route-label", "Keyproxy8"])

    assert args.route_labels == ["Keyproxy4", "Keyproxy8"]


def test_parse_args_uses_conservative_default_concurrency() -> None:
    args = _parse_args([])

    assert args.concurrency == 1


def test_parse_args_uses_conservative_default_timeout() -> None:
    args = _parse_args([])

    assert args.timeout_seconds == 20.0
