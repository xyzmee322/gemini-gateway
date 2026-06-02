from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from math import ceil
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gemini_gateway_cli_errors import gateway_cli_error_payload  # noqa: E402

_DEFAULT_MODEL = "google/gemini-3.5-flash"
_DEFAULT_SAMPLE_LIMIT = 10
_DEFAULT_RECENT_ATTEMPT_WINDOW_MINUTES = 120
_SAFE_AUDIT_ERROR = "Не удалось проверить Gemini Gateway routes. Проверьте Postgres и настройки окружения."
_MANUAL_REVIEW_REASONS = frozenset({"key_status:disabled_needs_review"})


def summarize_route_pool_rows(
    *,
    model: str,
    rows: Iterable[Mapping[str, Any]],
    now: datetime | None = None,
    sample_limit: int = _DEFAULT_SAMPLE_LIMIT,
    attempted_binding_ids: Iterable[object] | None = None,
    estimated_tokens: int = 1,
) -> dict[str, Any]:
    """Строит безопасную сводку причин недоступности route pool."""

    normalized_now = _aware(now) or datetime.now(tz=UTC)
    normalized_rows = [dict(row) for row in rows]
    attempted_ids = {str(item) for item in attempted_binding_ids or () if item is not None}
    route_states = []
    for row in normalized_rows:
        cooldowns = _route_cooldown_details(row, now=normalized_now)
        cooldown_state_drift = _route_cooldown_state_drift(row, now=normalized_now)
        state = {
            "route_label": _optional_str(row.get("route_label")),
            "key_label": _optional_str(row.get("key_label")),
            "proxy_label": _optional_str(row.get("proxy_label")),
            "reasons": _route_block_reasons(
                row,
                now=normalized_now,
                attempted_binding_ids=attempted_ids,
                estimated_tokens=max(1, estimated_tokens),
            ),
        }
        if cooldowns:
            state["cooldowns"] = cooldowns
        if cooldown_state_drift:
            state["cooldown_state_drift"] = cooldown_state_drift
        route_states.append(state)
    blocked_route_states = [state for state in route_states if state["reasons"]]
    retry_after_seconds = _retry_after_seconds(
        (
            value
            for row in normalized_rows
            for value in (
                row.get("key_sleep_until"),
                row.get("binding_sleep_until"),
                row.get("proxy_sleep_until"),
                row.get("cooldown_until"),
            )
        ),
        now=normalized_now,
    )
    quota_retry_after_seconds = _quota_retry_after_seconds(
        route_states=route_states,
        now=normalized_now,
    )
    combined_retry_after_seconds = _min_retry_after_seconds(retry_after_seconds, quota_retry_after_seconds)
    limit = max(0, sample_limit)
    manual_review_route_states = [
        state for state in blocked_route_states if _route_requires_manual_review(state)
    ]
    quota_route_states = [
        state for state in blocked_route_states if _route_has_quota_reason(state)
    ]

    return {
        "status": "ok" if len(blocked_route_states) < len(route_states) else "no_route",
        "model": model,
        "route_total": len(route_states),
        "reservable_count": len(route_states) - len(blocked_route_states),
        "blocked_count": len(blocked_route_states),
        "blocked_by_cooldown_count": _count_with_reason_prefix(blocked_route_states, "cooldown:"),
        "blocked_by_sleep_count": _count_with_reason_prefix(blocked_route_states, "sleep:"),
        "blocked_by_quota_count": _count_with_reason_prefix(blocked_route_states, "quota:"),
        "attempted_binding_count": _count_exact_reason(blocked_route_states, "attempted_in_request"),
        "disabled_key_count": _count_with_reason_prefix(blocked_route_states, "key_status:"),
        "disabled_binding_count": _count_with_reason_prefix(blocked_route_states, "binding_status:"),
        "disabled_proxy_count": _count_with_reason_prefix(blocked_route_states, "proxy_status:"),
        "inactive_project_count": _count_with_reason_prefix(blocked_route_states, "project_status:"),
        "inactive_model_limit_count": _count_with_reason_prefix(blocked_route_states, "model_limit_status:"),
        "invalid_transport_count": _count_with_reason_prefix(blocked_route_states, "transport_mode:"),
        "missing_proxy_count": _count_exact_reason(blocked_route_states, "proxy_missing"),
        "cooldown_reason_counts": _cooldown_reason_counts(blocked_route_states),
        "cooldown_level_counts": _cooldown_level_counts(blocked_route_states),
        "cooldown_state_drift_count": _cooldown_state_drift_count(blocked_route_states),
        "cooldown_state_drift_scope_counts": _cooldown_state_drift_scope_counts(blocked_route_states),
        "quota_reason_counts": _quota_reason_counts(blocked_route_states),
        "quota_retry_after_seconds": quota_retry_after_seconds,
        "quota_retry_after_at": _retry_after_at_iso(quota_retry_after_seconds, now=normalized_now),
        "quota_route_samples": quota_route_states[:limit],
        "manual_review_required_count": len(manual_review_route_states),
        "manual_review_reason_counts": _manual_review_reason_counts(manual_review_route_states),
        "manual_review_route_samples": manual_review_route_states[:limit],
        "retry_after_seconds": combined_retry_after_seconds,
        "retry_after_at": _retry_after_at_iso(combined_retry_after_seconds, now=normalized_now),
        "blocked_route_samples": blocked_route_states[:limit],
    }


async def audit_gemini_gateway_routes(
    *,
    model: str,
    sample_limit: int = _DEFAULT_SAMPLE_LIMIT,
    soybob_request_id: str | None = None,
    estimated_tokens: int = 1,
    recent_attempt_window_minutes: int = _DEFAULT_RECENT_ATTEMPT_WINDOW_MINUTES,
) -> dict[str, Any]:
    from gemini_gateway.config import get_gateway_settings
    from gemini_gateway.db.session import create_gateway_engine

    settings = get_gateway_settings()
    engine = create_gateway_engine(settings)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    now = datetime.now(tz=UTC)
    try:
        rows = await fetch_route_pool_rows(session_factory, model=model, now=now)
        attempted_binding_ids = (
            await fetch_attempted_binding_ids(
                session_factory,
                model=model,
                soybob_request_id=soybob_request_id,
            )
            if soybob_request_id
            else ()
        )
        report = summarize_route_pool_rows(
            model=model,
            rows=rows,
            now=now,
            sample_limit=sample_limit,
            attempted_binding_ids=attempted_binding_ids,
            estimated_tokens=estimated_tokens,
        )
        if recent_attempt_window_minutes > 0:
            recent_attempt_rows = await fetch_recent_attempt_rows(
                session_factory,
                model=model,
                now=now,
                window_minutes=recent_attempt_window_minutes,
            )
            report["recent_attempts"] = summarize_recent_attempt_rows(
                window_minutes=recent_attempt_window_minutes,
                rows=recent_attempt_rows,
            )
            recent_request_group_rows = await fetch_recent_request_group_rows(
                session_factory,
                model=model,
                now=now,
                window_minutes=recent_attempt_window_minutes,
            )
            report["recent_request_groups"] = summarize_recent_request_group_rows(
                window_minutes=recent_attempt_window_minutes,
                rows=recent_request_group_rows,
            )
        if soybob_request_id:
            report["soybob_request_id"] = soybob_request_id
        return report
    finally:
        await engine.dispose()


async def fetch_route_pool_rows(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    model: str,
    now: datetime,
) -> list[dict[str, Any]]:
    """Читает route pool без расшифровки API keys и proxy credentials."""

    query = text(
        """
        WITH route_base AS (
            SELECT
                b.id AS binding_id,
                p.id AS project_id,
                k.id AS api_key_id,
                px.id AS proxy_id,
                b.label AS route_label,
                p.label AS project_label,
                k.label AS key_label,
                px.label AS proxy_label,
                p.status AS project_status,
                k.status AS key_status,
                b.status AS binding_status,
                px.status AS proxy_status,
                ml.status AS model_limit_status,
                b.transport_mode AS transport_mode,
                ml.model AS model,
                ml.requests_per_minute AS requests_per_minute,
                ml.tokens_per_minute AS tokens_per_minute,
                ml.requests_per_day AS requests_per_day,
                ml.tokens_per_day AS tokens_per_day,
                COALESCE(qm.requests_used, 0) AS minute_requests_used,
                COALESCE(qm.input_tokens_reserved, 0) AS minute_tokens_reserved,
                COALESCE(qd.requests_used, 0) AS day_requests_used,
                COALESCE(qd.input_tokens_reserved, 0) AS day_tokens_reserved,
                k.sleep_until AS key_sleep_until,
                b.sleep_until AS binding_sleep_until,
                px.sleep_until AS proxy_sleep_until
            FROM gemini_gateway.key_proxy_bindings b
            JOIN gemini_gateway.api_keys k ON k.id = b.api_key_id
            JOIN gemini_gateway.google_projects p ON p.id = k.project_id
            JOIN gemini_gateway.model_limits ml ON ml.project_id = p.id AND ml.model = :model
            LEFT JOIN gemini_gateway.proxy_endpoints px ON px.id = b.proxy_id
            LEFT JOIN gemini_gateway.quota_windows qm
                ON qm.project_id = p.id
               AND qm.model = ml.model
               AND qm.window_kind = 'minute'
               AND qm.window_start = date_trunc('minute', CAST(:now AS timestamptz))
            LEFT JOIN gemini_gateway.quota_windows qd
                ON qd.project_id = p.id
               AND qd.model = ml.model
               AND qd.window_kind = 'day'
               AND qd.window_start = date_trunc('day', CAST(:now AS timestamptz))
        )
        SELECT
            rb.binding_id,
            rb.project_id,
            rb.api_key_id,
            rb.proxy_id,
            rb.route_label,
            rb.project_label,
            rb.key_label,
            rb.proxy_label,
            rb.project_status,
            rb.key_status,
            rb.binding_status,
            rb.proxy_status,
            rb.model_limit_status,
            rb.transport_mode,
            rb.model,
            rb.requests_per_minute,
            rb.tokens_per_minute,
            rb.requests_per_day,
            rb.tokens_per_day,
            rb.minute_requests_used,
            rb.minute_tokens_reserved,
            rb.day_requests_used,
            rb.day_tokens_reserved,
            rb.key_sleep_until,
            rb.binding_sleep_until,
            rb.proxy_sleep_until,
            COALESCE(array_remove(array_agg(DISTINCT cd.scope), NULL), ARRAY[]::text[]) AS cooldown_scopes,
            COALESCE(
                jsonb_agg(
                    DISTINCT jsonb_build_object(
                        'scope', cd.scope,
                        'reason', cd.error_type,
                        'level', cd.cooldown_level,
                        'sleep_until', cd.sleep_until
                    )
                ) FILTER (WHERE cd.id IS NOT NULL),
                '[]'::jsonb
            ) AS cooldown_details,
            MIN(cd.sleep_until) AS cooldown_until
        FROM route_base rb
        LEFT JOIN gemini_gateway.cooldowns cd
          ON cd.status = 'active'
         AND cd.sleep_until > :now
         AND (
                (cd.scope = 'project_model' AND cd.scope_key = rb.project_id::text AND cd.model = rb.model)
             OR (cd.scope = 'key' AND cd.scope_key = rb.api_key_id::text)
             OR (
                    cd.scope = 'proxy'
                    AND cd.scope_key = rb.proxy_id::text
                    AND (cd.model IS NULL OR cd.model = rb.model)
                )
             OR (
                    cd.scope = 'binding'
                    AND cd.scope_key = rb.binding_id::text
                    AND (cd.model IS NULL OR cd.model = rb.model)
                )
             OR (cd.scope = 'provider_model' AND cd.scope_key = rb.model)
         )
        GROUP BY
            rb.binding_id,
            rb.project_id,
            rb.api_key_id,
            rb.proxy_id,
            rb.route_label,
            rb.project_label,
            rb.key_label,
            rb.proxy_label,
            rb.project_status,
            rb.key_status,
            rb.binding_status,
            rb.proxy_status,
            rb.model_limit_status,
            rb.transport_mode,
            rb.model,
            rb.requests_per_minute,
            rb.tokens_per_minute,
            rb.requests_per_day,
            rb.tokens_per_day,
            rb.minute_requests_used,
            rb.minute_tokens_reserved,
            rb.day_requests_used,
            rb.day_tokens_reserved,
            rb.key_sleep_until,
            rb.binding_sleep_until,
            rb.proxy_sleep_until
        ORDER BY rb.route_label ASC
        """
    )
    async with session_factory() as session:
        result = await session.execute(query, {"model": model, "now": now})
        rows = [dict(row) for row in result.mappings().all()]
        await session.rollback()
        return rows


async def fetch_attempted_binding_ids(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    model: str,
    soybob_request_id: str,
) -> set[str]:
    """Возвращает binding ids, уже попробованные внутри одного LLM-запроса."""

    query = text(
        """
        SELECT DISTINCT binding_id::text
        FROM gemini_gateway.route_attempts
        WHERE soybob_request_id = :soybob_request_id
          AND model = :model
          AND binding_id IS NOT NULL
        """
    )
    async with session_factory() as session:
        rows = (
            await session.execute(
                query,
                {
                    "model": model,
                    "soybob_request_id": soybob_request_id,
                },
            )
        ).all()
        await session.rollback()
        return {str(row[0]) for row in rows if row[0] is not None}


async def fetch_recent_attempt_rows(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    model: str,
    now: datetime,
    window_minutes: int,
) -> list[dict[str, Any]]:
    """Читает безопасную агрегацию недавних route attempts без raw payload."""

    query = text(
        """
        SELECT
            status,
            error_type,
            route_label,
            retry_count,
            provider_response_json->>'provider_reason' AS provider_reason,
            COUNT(*) AS count
        FROM gemini_gateway.route_attempts
        WHERE model = :model
          AND created_at >= CAST(:now AS timestamptz) - (CAST(:window_minutes AS integer) * interval '1 minute')
        GROUP BY status, error_type, route_label, retry_count, provider_reason
        ORDER BY COUNT(*) DESC, status ASC, error_type ASC, route_label ASC, retry_count ASC
        """
    )
    async with session_factory() as session:
        result = await session.execute(
            query,
            {
                "model": model,
                "now": now,
                "window_minutes": max(1, int(window_minutes)),
            },
        )
        rows = [dict(row) for row in result.mappings().all()]
        await session.rollback()
        return rows


async def fetch_recent_request_group_rows(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    model: str,
    now: datetime,
    window_minutes: int,
) -> list[dict[str, Any]]:
    """Агрегирует outcome provider-attempt цепочек без raw payload и raw prompts."""

    query = text(
        """
        SELECT
            soybob_request_id,
            COUNT(*) AS attempt_count,
            COUNT(*) FILTER (WHERE status = 'success') AS success_count,
            COUNT(*) FILTER (WHERE status = 'failed') AS failed_count,
            COUNT(*) FILTER (WHERE error_type = 'network_timeout') AS network_timeout_count,
            COUNT(*) FILTER (WHERE status = 'skipped_no_route' OR error_type = 'no_route') AS no_route_count,
            MAX(retry_count) AS max_retry_count
        FROM gemini_gateway.route_attempts
        WHERE model = :model
          AND soybob_request_id IS NOT NULL
          AND created_at >= CAST(:now AS timestamptz) - (CAST(:window_minutes AS integer) * interval '1 minute')
        GROUP BY soybob_request_id
        ORDER BY MAX(created_at) DESC
        """
    )
    async with session_factory() as session:
        result = await session.execute(
            query,
            {
                "model": model,
                "now": now,
                "window_minutes": max(1, int(window_minutes)),
            },
        )
        rows = [dict(row) for row in result.mappings().all()]
        await session.rollback()
        return rows


def summarize_recent_attempt_rows(
    *,
    window_minutes: int,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Строит безопасную сводку недавних route attempts."""

    normalized_rows = [dict(row) for row in rows]
    total_count = sum(_count_value(row.get("count")) for row in normalized_rows)
    return {
        "window_minutes": max(1, int(window_minutes)),
        "total_count": total_count,
        "status_counts": _counts_by_field(normalized_rows, "status"),
        "error_type_counts": _counts_by_field(normalized_rows, "error_type"),
        "retry_count_counts": _counts_by_field(normalized_rows, "retry_count"),
        "provider_reason_counts": _provider_reason_counts(normalized_rows),
        "failed_route_counts": _failed_route_counts(normalized_rows),
        "route_attempt_counts": _route_attempt_counts(normalized_rows),
    }


def summarize_recent_request_group_rows(
    *,
    window_minutes: int,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Строит безопасную сводку исходов retry-цепочек по soybob_request_id."""

    normalized_rows = [dict(row) for row in rows]
    total_group_count = len(normalized_rows)
    success_group_count = sum(1 for row in normalized_rows if _int_value(row.get("success_count")) > 0)
    timeout_group_count = sum(
        1 for row in normalized_rows if _int_value(row.get("network_timeout_count")) > 0
    )
    no_route_group_count = sum(1 for row in normalized_rows if _int_value(row.get("no_route_count")) > 0)
    timeout_then_success_group_count = sum(
        1
        for row in normalized_rows
        if _int_value(row.get("network_timeout_count")) > 0
        and _int_value(row.get("success_count")) > 0
    )
    timeout_then_no_route_group_count = sum(
        1
        for row in normalized_rows
        if _int_value(row.get("network_timeout_count")) > 0
        and _int_value(row.get("no_route_count")) > 0
        and _int_value(row.get("success_count")) == 0
    )
    attempt_counts = [_int_value(row.get("attempt_count")) for row in normalized_rows]
    retry_counts = [_int_value(row.get("max_retry_count")) for row in normalized_rows]
    return {
        "window_minutes": max(1, int(window_minutes)),
        "total_group_count": total_group_count,
        "success_group_count": success_group_count,
        "failed_group_count": total_group_count - success_group_count,
        "timeout_group_count": timeout_group_count,
        "timeout_then_success_group_count": timeout_then_success_group_count,
        "timeout_then_no_route_group_count": timeout_then_no_route_group_count,
        "no_route_group_count": no_route_group_count,
        "max_attempt_count": max(attempt_counts, default=0),
        "max_retry_count": max(retry_counts, default=0),
        "attempt_count_counts": _counts_by_field(normalized_rows, "attempt_count"),
    }


def _route_block_reasons(
    row: Mapping[str, Any],
    *,
    now: datetime,
    attempted_binding_ids: set[str],
    estimated_tokens: int,
) -> list[str]:
    reasons: list[str] = []
    binding_id = row.get("binding_id")
    if binding_id is not None and str(binding_id) in attempted_binding_ids:
        reasons.append("attempted_in_request")

    for field_name, prefix in (
        ("project_status", "project_status"),
        ("key_status", "key_status"),
        ("binding_status", "binding_status"),
        ("proxy_status", "proxy_status"),
        ("model_limit_status", "model_limit_status"),
    ):
        value = _optional_str(row.get(field_name))
        if value != "active":
            reasons.append(f"{prefix}:{value or 'missing'}")

    transport_mode = _optional_str(row.get("transport_mode"))
    if transport_mode != "proxy":
        reasons.append(f"transport_mode:{transport_mode or 'missing'}")
    if row.get("proxy_id") is None and row.get("proxy_label") is None:
        reasons.append("proxy_missing")

    requests_per_minute = _optional_int(row.get("requests_per_minute"))
    requests_per_day = _optional_int(row.get("requests_per_day"))
    tokens_per_minute = _optional_int(row.get("tokens_per_minute"))
    tokens_per_day = _optional_int(row.get("tokens_per_day"))
    if requests_per_minute is not None and _int_value(row.get("minute_requests_used")) >= requests_per_minute:
        reasons.append("quota:minute_requests")
    if requests_per_day is not None and _int_value(row.get("day_requests_used")) >= requests_per_day:
        reasons.append("quota:day_requests")
    if tokens_per_minute is not None and estimated_tokens > max(tokens_per_minute - _int_value(row.get("minute_tokens_reserved")), 0):
        reasons.append("quota:minute_tokens")
    if tokens_per_day is not None and estimated_tokens > max(tokens_per_day - _int_value(row.get("day_tokens_reserved")), 0):
        reasons.append("quota:day_tokens")

    for field_name, scope in (
        ("key_sleep_until", "key"),
        ("binding_sleep_until", "binding"),
        ("proxy_sleep_until", "proxy"),
    ):
        if _is_future(row.get(field_name), now=now):
            reasons.append(f"sleep:{scope}")

    for scope in _cooldown_scopes(row.get("cooldown_scopes")):
        reasons.append(f"cooldown:{scope}")
    return reasons


def _cooldown_scopes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return sorted({str(item) for item in value if item})
    return []


def _route_cooldown_details(row: Mapping[str, Any], *, now: datetime) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for item in _cooldown_detail_items(row.get("cooldown_details")):
        scope = _optional_str(item.get("scope"))
        reason = _optional_str(item.get("reason") or item.get("error_type"))
        level = _optional_int(item.get("level") or item.get("cooldown_level"))
        sleep_until = _aware(item.get("sleep_until"))
        if scope is None or reason is None:
            continue
        detail: dict[str, Any] = {
            "scope": scope,
            "reason": reason,
        }
        if level is not None:
            detail["level"] = level
        if sleep_until is not None and sleep_until > now:
            detail["seconds_left"] = max(1, ceil((sleep_until - now).total_seconds()))
        details.append(detail)
    return sorted(details, key=lambda detail: (str(detail.get("scope")), str(detail.get("reason"))))


def _route_cooldown_state_drift(row: Mapping[str, Any], *, now: datetime) -> list[dict[str, Any]]:
    drifts: list[dict[str, Any]] = []
    for item in _cooldown_detail_items(row.get("cooldown_details")):
        scope = _optional_str(item.get("scope"))
        reason = _optional_str(item.get("reason") or item.get("error_type"))
        cooldown_sleep_until = _aware(item.get("sleep_until"))
        state_field = _state_sleep_field_for_scope(scope)
        if scope is None or reason is None or state_field is None:
            continue
        if cooldown_sleep_until is None or cooldown_sleep_until <= now:
            continue

        state_sleep_until = _aware(row.get(state_field))
        drift_state = _cooldown_state_drift_state(
            state_sleep_until=state_sleep_until,
            cooldown_sleep_until=cooldown_sleep_until,
            now=now,
        )
        if drift_state is None:
            continue
        drifts.append(
            {
                "scope": scope,
                "reason": reason,
                "state_field": state_field,
                "state": drift_state,
            }
        )
    return sorted(drifts, key=lambda item: (item["scope"], item["reason"], item["state_field"]))


def _cooldown_detail_items(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return _cooldown_detail_items(decoded)
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Iterable):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _state_sleep_field_for_scope(scope: str | None) -> str | None:
    return {
        "key": "key_sleep_until",
        "binding": "binding_sleep_until",
        "proxy": "proxy_sleep_until",
    }.get(scope or "")


def _cooldown_state_drift_state(
    *,
    state_sleep_until: datetime | None,
    cooldown_sleep_until: datetime,
    now: datetime,
) -> str | None:
    if state_sleep_until is None or state_sleep_until <= now:
        return "missing_or_elapsed"
    if state_sleep_until < cooldown_sleep_until:
        return "shorter_than_cooldown"
    if state_sleep_until > cooldown_sleep_until:
        return "longer_than_cooldown"
    return None


def _cooldown_reason_counts(route_states: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in route_states:
        reasons = {
            str(detail["reason"])
            for detail in state.get("cooldowns", [])
            if isinstance(detail, Mapping) and detail.get("reason")
        }
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _cooldown_level_counts(route_states: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in route_states:
        levels = [
            int(detail["level"])
            for detail in state.get("cooldowns", [])
            if isinstance(detail, Mapping) and _optional_int(detail.get("level")) is not None
        ]
        if not levels:
            continue
        level_key = str(max(levels))
        counts[level_key] = counts.get(level_key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _cooldown_state_drift_count(route_states: Iterable[Mapping[str, Any]]) -> int:
    return sum(
        1
        for state in route_states
        for drift in state.get("cooldown_state_drift", [])
        if isinstance(drift, Mapping)
    )


def _cooldown_state_drift_scope_counts(route_states: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in route_states:
        scopes = {
            str(drift["scope"])
            for drift in state.get("cooldown_state_drift", [])
            if isinstance(drift, Mapping) and drift.get("scope")
        }
        for scope in scopes:
            counts[scope] = counts.get(scope, 0) + 1
    return dict(sorted(counts.items()))


def _count_with_reason_prefix(route_states: Iterable[Mapping[str, Any]], prefix: str) -> int:
    return sum(1 for state in route_states if any(str(reason).startswith(prefix) for reason in state["reasons"]))


def _count_exact_reason(route_states: Iterable[Mapping[str, Any]], reason: str) -> int:
    return sum(1 for state in route_states if reason in state["reasons"])


def _quota_reason_counts(route_states: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in route_states:
        reasons = {str(reason) for reason in state.get("reasons", []) if str(reason).startswith("quota:")}
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _route_has_quota_reason(route_state: Mapping[str, Any]) -> bool:
    return any(str(reason).startswith("quota:") for reason in route_state.get("reasons", []))


def _quota_retry_after_seconds(
    *,
    route_states: Sequence[Mapping[str, Any]],
    now: datetime,
) -> int | None:
    route_retry_values: list[int] = []
    for state in route_states:
        retry_after = _route_quota_retry_after_seconds(
            reasons=[str(reason) for reason in state.get("reasons", [])],
            now=now,
        )
        if retry_after is not None:
            route_retry_values.append(retry_after)
    return min(route_retry_values) if route_retry_values else None


def _route_quota_retry_after_seconds(
    *,
    reasons: Sequence[str],
    now: datetime,
) -> int | None:
    quota_reasons = [reason for reason in reasons if reason.startswith("quota:")]
    if not quota_reasons:
        return None
    if any(not reason.startswith("quota:") for reason in reasons):
        return None

    retry_values: list[int] = []
    if any(reason in {"quota:minute_requests", "quota:minute_tokens"} for reason in quota_reasons):
        retry_values.append(_seconds_until_next_minute(now))
    if any(reason in {"quota:day_requests", "quota:day_tokens"} for reason in quota_reasons):
        retry_values.append(_seconds_until_next_day(now))
    return max(retry_values) if retry_values else None


def _route_requires_manual_review(route_state: Mapping[str, Any]) -> bool:
    return any(reason in _MANUAL_REVIEW_REASONS for reason in route_state.get("reasons", []))


def _manual_review_reason_counts(route_states: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in route_states:
        reasons = {str(reason) for reason in state.get("reasons", []) if reason in _MANUAL_REVIEW_REASONS}
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _counts_by_field(rows: Iterable[Mapping[str, Any]], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = _optional_str(row.get(field_name))
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + _count_value(row.get("count"))
    return dict(sorted(counts.items()))


def _provider_reason_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = _optional_str(row.get("provider_reason"))
        if value is None:
            value = _provider_reason_from_payload(row.get("provider_response_json"))
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + _count_value(row.get("count"))
    return dict(sorted(counts.items()))


def _failed_route_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if _optional_str(row.get("status")) != "failed":
            continue
        route_label = _optional_str(row.get("route_label"))
        if route_label is None:
            continue
        counts[route_label] = counts.get(route_label, 0) + _count_value(row.get("count"))
    return dict(sorted(counts.items()))


def _route_attempt_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    rows_by_route: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        route_label = _optional_str(row.get("route_label"))
        if route_label is None:
            continue
        rows_by_route.setdefault(route_label, []).append(row)

    return {
        route_label: {
            "total_count": sum(_count_value(row.get("count")) for row in route_rows),
            "status_counts": _counts_by_field(route_rows, "status"),
            "error_type_counts": _counts_by_field(route_rows, "error_type"),
            "retry_count_counts": _counts_by_field(route_rows, "retry_count"),
            "provider_reason_counts": _provider_reason_counts(route_rows),
        }
        for route_label, route_rows in sorted(rows_by_route.items())
    }


def _provider_reason_from_payload(value: Any) -> str | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Mapping):
        return None
    return _optional_str(value.get("provider_reason"))


def _count_value(value: Any) -> int:
    return max(0, _int_value(value, default=1))


def _retry_after_seconds(values: Iterable[Any], *, now: datetime) -> int | None:
    future_values = [value for value in (_aware(item) for item in values) if value is not None and value > now]
    if not future_values:
        return None
    seconds = ceil((min(future_values) - now).total_seconds())
    return max(1, seconds) if seconds > 0 else None


def _min_retry_after_seconds(*values: int | None) -> int | None:
    positive_values = [value for value in values if value is not None and value > 0]
    return min(positive_values) if positive_values else None


def _seconds_until_next_minute(now: datetime) -> int:
    next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    return max(1, ceil((next_minute - now).total_seconds()))


def _seconds_until_next_day(now: datetime) -> int:
    next_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return max(1, ceil((next_day - now).total_seconds()))


def _retry_after_at_iso(seconds: int | None, *, now: datetime) -> str | None:
    if seconds is None or seconds <= 0:
        return None
    retry_at = now + timedelta(seconds=seconds)
    return retry_at.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_future(value: Any, *, now: datetime) -> bool:
    aware_value = _aware(value)
    return aware_value is not None and aware_value > now


def _aware(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            normalized_value = value.replace("Z", "+00:00")
            parsed_value = datetime.fromisoformat(normalized_value)
        except ValueError:
            return None
        return _aware(parsed_value)
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text_value = str(value).strip()
    return text_value or None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_value(value: Any, *, default: int = 0) -> int:
    item = _optional_int(value)
    return item if item is not None else default


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Gemini Gateway route pool without exposing secrets.")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--soybob-request-id", default=None)
    parser.add_argument("--estimated-tokens", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=_DEFAULT_SAMPLE_LIMIT)
    parser.add_argument(
        "--recent-attempt-window-minutes",
        type=int,
        default=_DEFAULT_RECENT_ATTEMPT_WINDOW_MINUTES,
        help="Include recent route_attempts safe summary for this model; set 0 to disable.",
    )
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="Always exit with 0 after printing the JSON report.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = asyncio.run(
            audit_gemini_gateway_routes(
                model=args.model,
                sample_limit=args.sample_limit,
                soybob_request_id=args.soybob_request_id,
                estimated_tokens=args.estimated_tokens,
                recent_attempt_window_minutes=args.recent_attempt_window_minutes,
            )
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(
            json.dumps(
                gateway_cli_error_payload(exc, fallback_error=_SAFE_AUDIT_ERROR),
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 0 if args.exit_zero else 1

    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if args.exit_zero:
        return 0
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
