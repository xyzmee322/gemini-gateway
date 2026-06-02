from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from scripts.audit_gemini_gateway_routes import (
    summarize_recent_attempt_rows,
    summarize_recent_request_group_rows,
    summarize_route_pool_rows,
)


class _EmptyRows:
    def mappings(self) -> "_EmptyRows":
        return self

    def all(self) -> list[dict[str, Any]]:
        return []


class _CaptureSession:
    def __init__(self) -> None:
        self.sql: str | None = None
        self.params: dict[str, Any] | None = None

    async def __aenter__(self) -> "_CaptureSession":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def execute(self, statement: object, params: dict[str, Any] | None = None) -> _EmptyRows:
        self.sql = " ".join(str(statement).split())
        self.params = params or {}
        return _EmptyRows()

    async def rollback(self) -> None:
        return None


class _CaptureSessionFactory:
    def __init__(self) -> None:
        self.session = _CaptureSession()

    def __call__(self) -> _CaptureSession:
        return self.session


@pytest.mark.asyncio
async def test_fetch_route_pool_rows_scopes_transport_cooldowns_by_model() -> None:
    from scripts.audit_gemini_gateway_routes import fetch_route_pool_rows

    now = datetime(2026, 6, 1, 14, 50, tzinfo=UTC)
    session_factory = _CaptureSessionFactory()

    await fetch_route_pool_rows(session_factory, model="google/gemini-3.5-flash", now=now)  # type: ignore[arg-type]

    assert session_factory.session.sql is not None
    assert session_factory.session.sql.count("AND (cd.model IS NULL OR cd.model = rb.model)") >= 2
    assert session_factory.session.params == {"model": "google/gemini-3.5-flash", "now": now}


def test_summarize_route_pool_rows_groups_reservable_disabled_and_cooldown_routes() -> None:
    now = datetime(2026, 5, 31, 13, 50, tzinfo=UTC)

    report = summarize_route_pool_rows(
        model="google/gemini-3.5-flash",
        rows=[
            {
                "route_label": "Keyproxy4",
                "key_label": "Keyproxy4-key",
                "proxy_label": "Keyproxy4-proxy",
                "project_status": "active",
                "key_status": "active",
                "binding_status": "active",
                "proxy_status": "active",
                "model_limit_status": "active",
                "transport_mode": "proxy",
                "key_sleep_until": None,
                "binding_sleep_until": None,
                "proxy_sleep_until": None,
                "cooldown_scopes": [],
                "cooldown_until": None,
            },
            {
                "route_label": "Keyproxy5",
                "key_label": "Keyproxy5-key",
                "proxy_label": "Keyproxy5-proxy",
                "project_status": "active",
                "key_status": "active",
                "binding_status": "active",
                "proxy_status": "active",
                "model_limit_status": "active",
                "transport_mode": "proxy",
                "key_sleep_until": None,
                "binding_sleep_until": None,
                "proxy_sleep_until": None,
                "cooldown_scopes": ["proxy", "binding"],
                "cooldown_until": now + timedelta(seconds=90),
            },
            {
                "route_label": "Keyproxy9",
                "key_label": "Keyproxy9-key",
                "proxy_label": "Keyproxy9-proxy",
                "project_status": "active",
                "key_status": "disabled_needs_review",
                "binding_status": "active",
                "proxy_status": "active",
                "model_limit_status": "active",
                "transport_mode": "proxy",
                "key_sleep_until": None,
                "binding_sleep_until": None,
                "proxy_sleep_until": None,
                "cooldown_scopes": [],
                "cooldown_until": None,
            },
        ],
        now=now,
        sample_limit=5,
    )

    assert report["status"] == "ok"
    assert report["route_total"] == 3
    assert report["reservable_count"] == 1
    assert report["blocked_by_cooldown_count"] == 1
    assert report["disabled_key_count"] == 1
    assert report["retry_after_seconds"] == 90
    assert report["blocked_route_samples"] == [
        {
            "route_label": "Keyproxy5",
            "key_label": "Keyproxy5-key",
            "proxy_label": "Keyproxy5-proxy",
            "reasons": ["cooldown:binding", "cooldown:proxy"],
        },
        {
            "route_label": "Keyproxy9",
            "key_label": "Keyproxy9-key",
            "proxy_label": "Keyproxy9-proxy",
            "reasons": ["key_status:disabled_needs_review"],
        },
    ]


def test_summarize_route_pool_rows_reports_manual_review_routes_separately() -> None:
    now = datetime(2026, 5, 31, 13, 50, tzinfo=UTC)

    report = summarize_route_pool_rows(
        model="google/gemini-embedding-2",
        rows=[
            {
                "route_label": "Keyproxy2",
                "key_label": "Keyproxy2-key",
                "proxy_label": "Keyproxy2-proxy",
                "project_status": "active",
                "key_status": "disabled_needs_review",
                "binding_status": "active",
                "proxy_status": "active",
                "model_limit_status": "active",
                "transport_mode": "proxy",
                "key_sleep_until": None,
                "binding_sleep_until": None,
                "proxy_sleep_until": None,
                "cooldown_scopes": [],
                "cooldown_until": None,
            },
            {
                "route_label": "Keyproxy4",
                "key_label": "Keyproxy4-key",
                "proxy_label": "Keyproxy4-proxy",
                "project_status": "active",
                "key_status": "active",
                "binding_status": "active",
                "proxy_status": "active",
                "model_limit_status": "active",
                "transport_mode": "proxy",
                "key_sleep_until": None,
                "binding_sleep_until": None,
                "proxy_sleep_until": None,
                "cooldown_scopes": ["binding"],
                "cooldown_until": now + timedelta(seconds=30),
            },
        ],
        now=now,
        sample_limit=1,
    )

    assert report["manual_review_required_count"] == 1
    assert report["manual_review_reason_counts"] == {"key_status:disabled_needs_review": 1}
    assert report["manual_review_route_samples"] == [
        {
            "route_label": "Keyproxy2",
            "key_label": "Keyproxy2-key",
            "proxy_label": "Keyproxy2-proxy",
            "reasons": ["key_status:disabled_needs_review"],
        }
    ]


def test_summarize_route_pool_rows_reports_no_route_when_every_route_is_blocked() -> None:
    now = datetime(2026, 5, 31, 13, 50, tzinfo=UTC)

    report = summarize_route_pool_rows(
        model="google/gemini-3.1-flash-lite",
        rows=[
            {
                "route_label": "Keyproxy4",
                "key_label": "Keyproxy4-key",
                "proxy_label": "Keyproxy4-proxy",
                "project_status": "active",
                "key_status": "active",
                "binding_status": "active",
                "proxy_status": "active",
                "model_limit_status": "active",
                "transport_mode": "proxy",
                "key_sleep_until": None,
                "binding_sleep_until": now + timedelta(seconds=12),
                "proxy_sleep_until": None,
                "cooldown_scopes": [],
                "cooldown_until": None,
            }
        ],
        now=now,
    )

    assert report["status"] == "no_route"
    assert report["reservable_count"] == 0
    assert report["blocked_by_sleep_count"] == 1
    assert report["retry_after_seconds"] == 12


def test_summarize_route_pool_rows_counts_cooldown_reasons_per_route() -> None:
    now = datetime(2026, 5, 31, 13, 50, tzinfo=UTC)

    report = summarize_route_pool_rows(
        model="google/gemini-3.1-flash-lite",
        rows=[
            {
                "route_label": "Keyproxy4",
                "key_label": "Keyproxy4-key",
                "proxy_label": "Keyproxy4-proxy",
                "project_status": "active",
                "key_status": "active",
                "binding_status": "active",
                "proxy_status": "active",
                "model_limit_status": "active",
                "transport_mode": "proxy",
                "key_sleep_until": None,
                "binding_sleep_until": now + timedelta(seconds=3600),
                "proxy_sleep_until": now + timedelta(seconds=3600),
                "cooldown_scopes": ["binding", "proxy"],
                "cooldown_until": now + timedelta(seconds=3600),
                "cooldown_details": [
                    {
                        "scope": "binding",
                        "reason": "network_timeout",
                        "level": 4,
                        "sleep_until": (now + timedelta(seconds=3600)).isoformat(),
                    },
                    {
                        "scope": "proxy",
                        "reason": "network_timeout",
                        "level": 4,
                        "sleep_until": now + timedelta(seconds=3600),
                    },
                ],
            },
            {
                "route_label": "Keyproxy5",
                "key_label": "Keyproxy5-key",
                "proxy_label": "Keyproxy5-proxy",
                "project_status": "active",
                "key_status": "active",
                "binding_status": "active",
                "proxy_status": "active",
                "model_limit_status": "active",
                "transport_mode": "proxy",
                "key_sleep_until": None,
                "binding_sleep_until": now + timedelta(seconds=300),
                "proxy_sleep_until": now + timedelta(seconds=300),
                "cooldown_scopes": ["binding", "proxy"],
                "cooldown_until": now + timedelta(seconds=300),
                "cooldown_details": [
                    {
                        "scope": "binding",
                        "reason": "proxy_failed",
                        "level": 2,
                        "sleep_until": now + timedelta(seconds=300),
                    }
                ],
            },
        ],
        now=now,
        sample_limit=2,
    )

    assert report["status"] == "no_route"
    assert report["cooldown_reason_counts"] == {"network_timeout": 1, "proxy_failed": 1}
    assert report["cooldown_level_counts"] == {"2": 1, "4": 1}
    assert report["blocked_route_samples"][0]["cooldowns"] == [
        {
            "scope": "binding",
            "reason": "network_timeout",
            "level": 4,
            "seconds_left": 3600,
        },
        {
            "scope": "proxy",
            "reason": "network_timeout",
            "level": 4,
            "seconds_left": 3600,
        },
    ]


def test_summarize_route_pool_rows_reports_cooldown_state_drift() -> None:
    now = datetime(2026, 5, 31, 13, 50, tzinfo=UTC)

    report = summarize_route_pool_rows(
        model="google/gemini-3.1-flash-lite",
        rows=[
            {
                "route_label": "Keyproxy4",
                "key_label": "Keyproxy4-key",
                "proxy_label": "Keyproxy4-proxy",
                "project_status": "active",
                "key_status": "active",
                "binding_status": "active",
                "proxy_status": "active",
                "model_limit_status": "active",
                "transport_mode": "proxy",
                "key_sleep_until": None,
                "binding_sleep_until": now + timedelta(seconds=3600),
                "proxy_sleep_until": None,
                "cooldown_scopes": ["binding", "proxy"],
                "cooldown_until": now + timedelta(seconds=3600),
                "cooldown_details": [
                    {
                        "scope": "binding",
                        "reason": "network_timeout",
                        "level": 4,
                        "sleep_until": now + timedelta(seconds=3600),
                    },
                    {
                        "scope": "proxy",
                        "reason": "network_timeout",
                        "level": 4,
                        "sleep_until": now + timedelta(seconds=3600),
                    },
                ],
            },
        ],
        now=now,
    )

    assert report["cooldown_state_drift_count"] == 1
    assert report["cooldown_state_drift_scope_counts"] == {"proxy": 1}
    assert report["blocked_route_samples"][0]["cooldown_state_drift"] == [
        {
            "scope": "proxy",
            "reason": "network_timeout",
            "state_field": "proxy_sleep_until",
            "state": "missing_or_elapsed",
        }
    ]


def test_summarize_route_pool_rows_can_explain_no_route_for_single_retry_group() -> None:
    now = datetime(2026, 5, 31, 13, 50, tzinfo=UTC)

    report = summarize_route_pool_rows(
        model="google/gemini-3.1-flash-lite",
        rows=[
            {
                "binding_id": 4,
                "route_label": "Keyproxy4",
                "key_label": "Keyproxy4-key",
                "proxy_label": "Keyproxy4-proxy",
                "project_status": "active",
                "key_status": "active",
                "binding_status": "active",
                "proxy_status": "active",
                "model_limit_status": "active",
                "transport_mode": "proxy",
                "key_sleep_until": None,
                "binding_sleep_until": None,
                "proxy_sleep_until": None,
                "cooldown_scopes": [],
                "cooldown_until": None,
            }
        ],
        now=now,
        attempted_binding_ids={"4"},
    )

    assert report["status"] == "no_route"
    assert report["reservable_count"] == 0
    assert report["attempted_binding_count"] == 1
    assert report["blocked_route_samples"][0]["reasons"] == ["attempted_in_request"]


def test_summarize_route_pool_rows_accounts_for_route_quota_windows() -> None:
    now = datetime(2026, 5, 31, 13, 50, tzinfo=UTC)

    report = summarize_route_pool_rows(
        model="google/gemini-3.5-flash",
        rows=[
            {
                "binding_id": 4,
                "route_label": "Keyproxy4",
                "key_label": "Keyproxy4-key",
                "proxy_label": "Keyproxy4-proxy",
                "project_status": "active",
                "key_status": "active",
                "binding_status": "active",
                "proxy_status": "active",
                "model_limit_status": "active",
                "transport_mode": "proxy",
                "requests_per_minute": 1,
                "tokens_per_minute": 1000,
                "requests_per_day": 100,
                "tokens_per_day": 1000,
                "minute_requests_used": 1,
                "minute_tokens_reserved": 0,
                "day_requests_used": 1,
                "day_tokens_reserved": 0,
                "key_sleep_until": None,
                "binding_sleep_until": None,
                "proxy_sleep_until": None,
                "cooldown_scopes": [],
                "cooldown_until": None,
            },
            {
                "binding_id": 5,
                "route_label": "Keyproxy5",
                "key_label": "Keyproxy5-key",
                "proxy_label": "Keyproxy5-proxy",
                "project_status": "active",
                "key_status": "active",
                "binding_status": "active",
                "proxy_status": "active",
                "model_limit_status": "active",
                "transport_mode": "proxy",
                "requests_per_minute": 10,
                "tokens_per_minute": 100,
                "requests_per_day": 100,
                "tokens_per_day": 100,
                "minute_requests_used": 0,
                "minute_tokens_reserved": 90,
                "day_requests_used": 0,
                "day_tokens_reserved": 90,
                "key_sleep_until": None,
                "binding_sleep_until": None,
                "proxy_sleep_until": None,
                "cooldown_scopes": [],
                "cooldown_until": None,
            },
        ],
        now=now,
        estimated_tokens=20,
    )

    assert report["status"] == "no_route"
    assert report["blocked_by_quota_count"] == 2
    assert report["quota_reason_counts"] == {
        "quota:day_tokens": 1,
        "quota:minute_requests": 1,
        "quota:minute_tokens": 1,
    }
    assert report["quota_retry_after_seconds"] == 60
    assert report["quota_retry_after_at"] == "2026-05-31T13:51:00Z"
    assert report["retry_after_seconds"] == 60
    assert report["retry_after_at"] == "2026-05-31T13:51:00Z"
    assert report["quota_route_samples"] == [
        {
            "route_label": "Keyproxy4",
            "key_label": "Keyproxy4-key",
            "proxy_label": "Keyproxy4-proxy",
            "reasons": ["quota:minute_requests"],
        },
        {
            "route_label": "Keyproxy5",
            "key_label": "Keyproxy5-key",
            "proxy_label": "Keyproxy5-proxy",
            "reasons": ["quota:minute_tokens", "quota:day_tokens"],
        },
    ]
    assert report["blocked_route_samples"] == [
        {
            "route_label": "Keyproxy4",
            "key_label": "Keyproxy4-key",
            "proxy_label": "Keyproxy4-proxy",
            "reasons": ["quota:minute_requests"],
        },
        {
            "route_label": "Keyproxy5",
            "key_label": "Keyproxy5-key",
            "proxy_label": "Keyproxy5-proxy",
            "reasons": ["quota:minute_tokens", "quota:day_tokens"],
        },
    ]


def test_summarize_recent_attempt_rows_counts_safe_provider_reasons() -> None:
    report = summarize_recent_attempt_rows(
        window_minutes=120,
        rows=[
            {
                "status": "failed",
                "error_type": "network_timeout",
                "route_label": "Keyproxy4",
                "provider_reason": "read_timeout",
                "retry_count": 0,
                "count": 2,
            },
            {
                "status": "failed",
                "error_type": "network_timeout",
                "route_label": "Keyproxy5",
                "provider_response_json": {"provider_reason": "connect_timeout"},
                "retry_count": 1,
                "count": 1,
            },
            {
                "status": "success",
                "error_type": None,
                "route_label": "Keyproxy6",
                "retry_count": 2,
                "count": 3,
            },
        ],
    )

    assert report == {
        "window_minutes": 120,
        "total_count": 6,
        "status_counts": {"failed": 3, "success": 3},
        "error_type_counts": {"network_timeout": 3},
        "retry_count_counts": {"0": 2, "1": 1, "2": 3},
        "provider_reason_counts": {"connect_timeout": 1, "read_timeout": 2},
        "failed_route_counts": {"Keyproxy4": 2, "Keyproxy5": 1},
        "route_attempt_counts": {
            "Keyproxy4": {
                "total_count": 2,
                "status_counts": {"failed": 2},
                "error_type_counts": {"network_timeout": 2},
                "retry_count_counts": {"0": 2},
                "provider_reason_counts": {"read_timeout": 2},
            },
            "Keyproxy5": {
                "total_count": 1,
                "status_counts": {"failed": 1},
                "error_type_counts": {"network_timeout": 1},
                "retry_count_counts": {"1": 1},
                "provider_reason_counts": {"connect_timeout": 1},
            },
            "Keyproxy6": {
                "total_count": 3,
                "status_counts": {"success": 3},
                "error_type_counts": {},
                "retry_count_counts": {"2": 3},
                "provider_reason_counts": {},
            },
        },
    }


def test_summarize_recent_request_group_rows_reports_retry_outcomes() -> None:
    report = summarize_recent_request_group_rows(
        window_minutes=120,
        rows=[
            {
                "attempt_count": 3,
                "success_count": 1,
                "network_timeout_count": 2,
                "no_route_count": 0,
                "max_retry_count": 2,
            },
            {
                "attempt_count": 6,
                "success_count": 0,
                "network_timeout_count": 5,
                "no_route_count": 1,
                "max_retry_count": 5,
            },
            {
                "attempt_count": 1,
                "success_count": 1,
                "network_timeout_count": 0,
                "no_route_count": 0,
                "max_retry_count": 0,
            },
            {
                "attempt_count": 1,
                "success_count": 0,
                "network_timeout_count": 0,
                "no_route_count": 1,
                "max_retry_count": 0,
            },
        ],
    )

    assert report == {
        "window_minutes": 120,
        "total_group_count": 4,
        "success_group_count": 2,
        "failed_group_count": 2,
        "timeout_group_count": 2,
        "timeout_then_success_group_count": 1,
        "timeout_then_no_route_group_count": 1,
        "no_route_group_count": 2,
        "max_attempt_count": 6,
        "max_retry_count": 5,
        "attempt_count_counts": {"1": 2, "3": 1, "6": 1},
    }
