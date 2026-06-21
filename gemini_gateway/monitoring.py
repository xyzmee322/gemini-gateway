from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_AVG_TIMING_FIELDS = (
    "avg_provider_total_ms",
    "avg_request_prepare_ms",
    "avg_response_headers_ms",
    "avg_response_body_ms",
    "avg_response_parse_ms",
)
_P95_TIMING_FIELDS = (
    "p95_provider_total_ms",
    "p95_request_prepare_ms",
    "p95_response_headers_ms",
    "p95_response_body_ms",
    "p95_response_parse_ms",
)
_PROXY_P95_TIMING_FIELDS = {
    "p95_provider_total_ms": "max_route_p95_provider_total_ms",
    "p95_request_prepare_ms": "max_route_p95_request_prepare_ms",
    "p95_response_headers_ms": "max_route_p95_response_headers_ms",
    "p95_response_body_ms": "max_route_p95_response_body_ms",
    "p95_response_parse_ms": "max_route_p95_response_parse_ms",
}


@dataclass(frozen=True)
class MonitorWindow:
    """Ограничивает окно мониторинга безопасными значениями."""

    minutes: int | str = 180
    bucket_seconds: int | str = 60
    model: str | None = None
    proxy_label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "minutes", _clamp_int(self.minutes, minimum=5, maximum=10080, default=180))
        object.__setattr__(
            self,
            "bucket_seconds",
            _clamp_int(self.bucket_seconds, minimum=10, maximum=3600, default=60),
        )
        object.__setattr__(self, "model", _optional_non_empty_string(self.model))
        object.__setattr__(self, "proxy_label", _optional_non_empty_string(self.proxy_label))


async def fetch_proxy_summary(
    session_factory: async_sessionmaker[AsyncSession],
    window: MonitorWindow,
) -> dict[str, Any]:
    """Возвращает агрегированную сводку без секретов и сырых payload."""

    normalized_window = MonitorWindow(
        minutes=window.minutes,
        bucket_seconds=window.bucket_seconds,
        model=window.model,
        proxy_label=window.proxy_label,
    )
    async with session_factory() as session:
        result = await session.execute(
            text(
                """
                WITH active_proxy_cooldowns AS (
                    SELECT
                        scope_key::bigint AS proxy_id,
                        COUNT(*)::int AS active_cooldown_count,
                        MAX(cooldown_level)::int AS max_cooldown_level,
                        MIN(sleep_until) AS cooldown_until
                    FROM gemini_gateway.cooldowns
                    WHERE status = 'active'
                      AND scope = 'proxy'
                      AND scope_key ~ '^[0-9]+$'
                      AND sleep_until > now()
                    GROUP BY scope_key::bigint
                )
                SELECT
                    ra.proxy_id,
                    COALESCE(px.label, 'direct') AS proxy_label,
                    ra.route_label,
                    COUNT(*)::int AS total_requests,
                    COUNT(*) FILTER (WHERE ra.status = 'success')::int AS success_count,
                    COUNT(*) FILTER (WHERE ra.status = 'failed')::int AS failed_count,
                    COUNT(*) FILTER (WHERE ra.status = 'skipped_no_route')::int AS skipped_no_route_count,
                    COUNT(*) FILTER (WHERE ra.status = 'leased')::int AS leased_count,
                    COUNT(*) FILTER (WHERE ra.retryable)::int AS retryable_count,
                    COUNT(*) FILTER (WHERE ra.error_type = 'network_timeout')::int AS network_timeout_count,
                    COUNT(*) FILTER (WHERE ra.error_type = 'proxy_failed')::int AS proxy_failed_count,
                    COUNT(*) FILTER (WHERE ra.timeout_kind = 'read_timeout')::int AS read_timeout_count,
                    COUNT(*) FILTER (WHERE ra.timeout_kind = 'connect_timeout')::int AS connect_timeout_count,
                    COUNT(*) FILTER (WHERE ra.timeout_kind = 'write_timeout')::int AS write_timeout_count,
                    COUNT(*) FILTER (WHERE ra.timeout_kind = 'pool_timeout')::int AS pool_timeout_count,
                    COUNT(*) FILTER (WHERE ra.timeout_kind IS NOT NULL)::int AS timeout_count,
                    COUNT(*) FILTER (WHERE ra.timeout_stage = 'request')::int AS request_stage_timeout_count,
                    COUNT(*) FILTER (WHERE ra.timeout_stage = 'response_headers')::int AS response_headers_timeout_count,
                    COUNT(*) FILTER (WHERE ra.timeout_stage = 'response_body')::int AS response_body_timeout_count,
                    COUNT(*) FILTER (WHERE ra.timeout_stage = 'unknown')::int AS unknown_stage_timeout_count,
                    (AVG(ra.latency_ms) FILTER (WHERE ra.latency_ms IS NOT NULL))::int AS avg_latency_ms,
                    percentile_disc(0.95) WITHIN GROUP (ORDER BY ra.latency_ms)
                        FILTER (WHERE ra.latency_ms IS NOT NULL) AS p95_latency_ms,
                    (AVG(ra.provider_total_ms) FILTER (WHERE ra.provider_total_ms IS NOT NULL))::int
                        AS avg_provider_total_ms,
                    percentile_disc(0.95) WITHIN GROUP (ORDER BY ra.provider_total_ms)
                        FILTER (WHERE ra.provider_total_ms IS NOT NULL) AS p95_provider_total_ms,
                    (AVG(ra.request_prepare_ms) FILTER (WHERE ra.request_prepare_ms IS NOT NULL))::int
                        AS avg_request_prepare_ms,
                    percentile_disc(0.95) WITHIN GROUP (ORDER BY ra.request_prepare_ms)
                        FILTER (WHERE ra.request_prepare_ms IS NOT NULL) AS p95_request_prepare_ms,
                    (AVG(ra.response_headers_ms) FILTER (WHERE ra.response_headers_ms IS NOT NULL))::int
                        AS avg_response_headers_ms,
                    percentile_disc(0.95) WITHIN GROUP (ORDER BY ra.response_headers_ms)
                        FILTER (WHERE ra.response_headers_ms IS NOT NULL) AS p95_response_headers_ms,
                    (AVG(ra.response_body_ms) FILTER (WHERE ra.response_body_ms IS NOT NULL))::int
                        AS avg_response_body_ms,
                    percentile_disc(0.95) WITHIN GROUP (ORDER BY ra.response_body_ms)
                        FILTER (WHERE ra.response_body_ms IS NOT NULL) AS p95_response_body_ms,
                    (AVG(ra.response_parse_ms) FILTER (WHERE ra.response_parse_ms IS NOT NULL))::int
                        AS avg_response_parse_ms,
                    percentile_disc(0.95) WITHIN GROUP (ORDER BY ra.response_parse_ms)
                        FILTER (WHERE ra.response_parse_ms IS NOT NULL) AS p95_response_parse_ms,
                    SUM(ra.request_bytes)::bigint AS request_bytes,
                    SUM(ra.response_bytes)::bigint AS response_bytes,
                    SUM(ra.media_count)::bigint AS media_count,
                    SUM(ra.image_count)::bigint AS image_count,
                    px.status AS proxy_status,
                    COALESCE(cooldowns.active_cooldown_count, 0)::int AS active_cooldown_count,
                    cooldowns.max_cooldown_level,
                    cooldowns.cooldown_until
                FROM gemini_gateway.route_attempts ra
                LEFT JOIN gemini_gateway.proxy_endpoints px ON px.id = ra.proxy_id
                LEFT JOIN active_proxy_cooldowns cooldowns ON cooldowns.proxy_id = ra.proxy_id
                WHERE ra.created_at >= now() - (:minutes * INTERVAL '1 minute')
                  AND (CAST(:model AS text) IS NULL OR ra.model = CAST(:model AS text))
                  AND (
                    CAST(:proxy_label AS text) IS NULL
                    OR COALESCE(px.label, 'direct') = CAST(:proxy_label AS text)
                  )
                GROUP BY
                    ra.proxy_id,
                    COALESCE(px.label, 'direct'),
                    ra.route_label,
                    px.status,
                    cooldowns.active_cooldown_count,
                    cooldowns.max_cooldown_level,
                    cooldowns.cooldown_until
                ORDER BY total_requests DESC, proxy_label ASC, ra.route_label ASC
                """
            ),
            _window_params(normalized_window),
        )
        return summarize_proxy_overview_rows(result.mappings().all())


async def fetch_proxy_timeseries(
    session_factory: async_sessionmaker[AsyncSession],
    window: MonitorWindow,
) -> dict[str, Any]:
    """Возвращает временные ряды мониторинга без чувствительных полей."""

    normalized_window = MonitorWindow(
        minutes=window.minutes,
        bucket_seconds=window.bucket_seconds,
        model=window.model,
        proxy_label=window.proxy_label,
    )
    async with session_factory() as session:
        result = await session.execute(
            text(
                """
                WITH active_proxy_cooldowns AS (
                    SELECT
                        scope_key::bigint AS proxy_id,
                        COUNT(*)::int AS active_cooldown_count,
                        MAX(cooldown_level)::int AS max_cooldown_level
                    FROM gemini_gateway.cooldowns
                    WHERE status = 'active'
                      AND scope = 'proxy'
                      AND scope_key ~ '^[0-9]+$'
                      AND sleep_until > now()
                    GROUP BY scope_key::bigint
                )
                SELECT
                    to_timestamp(
                        floor(extract(epoch from ra.created_at) / :bucket_seconds) * :bucket_seconds
                    ) AS bucket_start,
                    ra.proxy_id,
                    COALESCE(px.label, 'direct') AS proxy_label,
                    ra.route_label,
                    COUNT(*)::int AS total_requests,
                    COUNT(*) FILTER (WHERE ra.status = 'success')::int AS success_count,
                    COUNT(*) FILTER (WHERE ra.status = 'failed')::int AS failed_count,
                    COUNT(*) FILTER (WHERE ra.error_type = 'network_timeout')::int AS network_timeout_count,
                    COUNT(*) FILTER (WHERE ra.timeout_kind = 'read_timeout')::int AS read_timeout_count,
                    COUNT(*) FILTER (WHERE ra.timeout_kind IS NOT NULL)::int AS timeout_count,
                    COUNT(*) FILTER (WHERE ra.timeout_stage = 'request')::int AS request_stage_timeout_count,
                    COUNT(*) FILTER (WHERE ra.timeout_stage = 'response_headers')::int AS response_headers_timeout_count,
                    COUNT(*) FILTER (WHERE ra.timeout_stage = 'response_body')::int AS response_body_timeout_count,
                    COUNT(*) FILTER (WHERE ra.timeout_stage = 'unknown')::int AS unknown_stage_timeout_count,
                    (AVG(ra.latency_ms) FILTER (WHERE ra.latency_ms IS NOT NULL))::int AS avg_latency_ms,
                    percentile_disc(0.95) WITHIN GROUP (ORDER BY ra.latency_ms)
                        FILTER (WHERE ra.latency_ms IS NOT NULL) AS p95_latency_ms,
                    (AVG(ra.provider_total_ms) FILTER (WHERE ra.provider_total_ms IS NOT NULL))::int
                        AS avg_provider_total_ms,
                    percentile_disc(0.95) WITHIN GROUP (ORDER BY ra.provider_total_ms)
                        FILTER (WHERE ra.provider_total_ms IS NOT NULL) AS p95_provider_total_ms,
                    (AVG(ra.request_prepare_ms) FILTER (WHERE ra.request_prepare_ms IS NOT NULL))::int
                        AS avg_request_prepare_ms,
                    percentile_disc(0.95) WITHIN GROUP (ORDER BY ra.request_prepare_ms)
                        FILTER (WHERE ra.request_prepare_ms IS NOT NULL) AS p95_request_prepare_ms,
                    (AVG(ra.response_headers_ms) FILTER (WHERE ra.response_headers_ms IS NOT NULL))::int
                        AS avg_response_headers_ms,
                    percentile_disc(0.95) WITHIN GROUP (ORDER BY ra.response_headers_ms)
                        FILTER (WHERE ra.response_headers_ms IS NOT NULL) AS p95_response_headers_ms,
                    (AVG(ra.response_body_ms) FILTER (WHERE ra.response_body_ms IS NOT NULL))::int
                        AS avg_response_body_ms,
                    percentile_disc(0.95) WITHIN GROUP (ORDER BY ra.response_body_ms)
                        FILTER (WHERE ra.response_body_ms IS NOT NULL) AS p95_response_body_ms,
                    (AVG(ra.response_parse_ms) FILTER (WHERE ra.response_parse_ms IS NOT NULL))::int
                        AS avg_response_parse_ms,
                    percentile_disc(0.95) WITHIN GROUP (ORDER BY ra.response_parse_ms)
                        FILTER (WHERE ra.response_parse_ms IS NOT NULL) AS p95_response_parse_ms,
                    SUM(ra.request_bytes)::bigint AS request_bytes,
                    SUM(ra.response_bytes)::bigint AS response_bytes,
                    COALESCE(cooldowns.active_cooldown_count, 0)::int AS active_cooldown_count,
                    cooldowns.max_cooldown_level
                FROM gemini_gateway.route_attempts ra
                LEFT JOIN gemini_gateway.proxy_endpoints px ON px.id = ra.proxy_id
                LEFT JOIN active_proxy_cooldowns cooldowns ON cooldowns.proxy_id = ra.proxy_id
                WHERE ra.created_at >= now() - (:minutes * INTERVAL '1 minute')
                  AND (CAST(:model AS text) IS NULL OR ra.model = CAST(:model AS text))
                  AND (
                    CAST(:proxy_label AS text) IS NULL
                    OR COALESCE(px.label, 'direct') = CAST(:proxy_label AS text)
                  )
                GROUP BY
                    bucket_start,
                    ra.proxy_id,
                    COALESCE(px.label, 'direct'),
                    ra.route_label,
                    cooldowns.active_cooldown_count,
                    cooldowns.max_cooldown_level
                ORDER BY bucket_start ASC, proxy_label ASC, ra.route_label ASC
                """
            ),
            _window_params(normalized_window),
        )
        return summarize_proxy_timeseries_rows(
            result.mappings().all(),
            bucket_seconds=normalized_window.bucket_seconds,
        )


def summarize_proxy_overview_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Агрегирует строки БД в безопасную сводку по прокси."""

    proxies: dict[tuple[Any, str], dict[str, Any]] = {}
    total_requests = 0
    success_count = 0
    failed_count = 0

    for row in rows:
        proxy_label = _safe_label(row.get("proxy_label"), fallback="direct")
        proxy_key = (row.get("proxy_id"), proxy_label)
        proxy = proxies.setdefault(proxy_key, _empty_proxy(proxy_label))
        route = _route_metrics(row)

        _add_counts(proxy, route)
        _update_proxy_state(proxy, row)
        proxy["routes"].append(route)
        proxy["max_route_p95_latency_ms"] = _max_optional_int(
            proxy.get("max_route_p95_latency_ms"),
            route.get("p95_latency_ms"),
        )
        proxy["avg_latency_ms"] = _weighted_average(
            current_value=proxy.get("avg_latency_ms"),
            current_weight=proxy["total_requests"] - route["total_requests"],
            next_value=route.get("avg_latency_ms"),
            next_weight=route["total_requests"],
        )
        _update_proxy_timing(proxy, route, previous_request_count=proxy["total_requests"] - route["total_requests"])
        total_requests += route["total_requests"]
        success_count += route["success_count"]
        failed_count += route["failed_count"]

    proxy_items = sorted(
        (_finalize_proxy(proxy) for proxy in proxies.values()),
        key=lambda item: (-item["total_requests"], item["proxy_label"]),
    )
    return {
        "total_requests": total_requests,
        "success_count": success_count,
        "failed_count": failed_count,
        "failure_rate": _failure_rate(failed_count, total_requests),
        "proxy_count": len(proxy_items),
        "proxies": proxy_items,
    }


def summarize_proxy_timeseries_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    bucket_seconds: int,
) -> dict[str, Any]:
    """Агрегирует bucket-строки в безопасный временной ряд."""

    normalized_bucket_seconds = _clamp_int(bucket_seconds, minimum=10, maximum=3600, default=60)
    series: list[dict[str, Any]] = []
    total_requests = 0
    for row in rows:
        route = _route_metrics(row)
        total_requests += route["total_requests"]
        series.append(
            {
                "bucket_start": _serialize_time(row.get("bucket_start")),
                "proxy_label": _safe_label(row.get("proxy_label"), fallback="direct"),
                "route_label": route["route_label"],
                "total_requests": route["total_requests"],
                "success_count": route["success_count"],
                "failed_count": route["failed_count"],
                "failure_rate": route["failure_rate"],
                "network_timeout_count": route["network_timeout_count"],
                "read_timeout_count": route["read_timeout_count"],
                "timeout_count": route["timeout_count"],
                "request_stage_timeout_count": route["request_stage_timeout_count"],
                "response_headers_timeout_count": route["response_headers_timeout_count"],
                "response_body_timeout_count": route["response_body_timeout_count"],
                "unknown_stage_timeout_count": route["unknown_stage_timeout_count"],
                "avg_latency_ms": route["avg_latency_ms"],
                "p95_latency_ms": route["p95_latency_ms"],
                **_timing_metrics(route),
                "request_bytes": route["request_bytes"],
                "response_bytes": route["response_bytes"],
            }
        )
    return {
        "bucket_seconds": normalized_bucket_seconds,
        "total_requests": total_requests,
        "series": series,
    }


def _empty_proxy(proxy_label: str) -> dict[str, Any]:
    return {
        "proxy_label": proxy_label,
        "proxy_status": None,
        "total_requests": 0,
        "success_count": 0,
        "failed_count": 0,
        "skipped_no_route_count": 0,
        "leased_count": 0,
        "retryable_count": 0,
        "network_timeout_count": 0,
        "proxy_failed_count": 0,
        "read_timeout_count": 0,
        "connect_timeout_count": 0,
        "write_timeout_count": 0,
        "pool_timeout_count": 0,
        "timeout_count": 0,
        "request_stage_timeout_count": 0,
        "response_headers_timeout_count": 0,
        "response_body_timeout_count": 0,
        "unknown_stage_timeout_count": 0,
        "request_bytes": 0,
        "response_bytes": 0,
        "media_count": 0,
        "image_count": 0,
        "avg_latency_ms": None,
        "max_route_p95_latency_ms": None,
        "avg_provider_total_ms": None,
        "avg_request_prepare_ms": None,
        "avg_response_headers_ms": None,
        "avg_response_body_ms": None,
        "avg_response_parse_ms": None,
        "max_route_p95_provider_total_ms": None,
        "max_route_p95_request_prepare_ms": None,
        "max_route_p95_response_headers_ms": None,
        "max_route_p95_response_body_ms": None,
        "max_route_p95_response_parse_ms": None,
        "active_cooldown_count": 0,
        "max_cooldown_level": None,
        "cooldown_until": None,
        "routes": [],
    }


def _route_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    total_requests = _non_negative_int(row.get("total_requests"))
    failed_count = _non_negative_int(row.get("failed_count"))
    return {
        "route_label": _optional_non_empty_string(row.get("route_label")) or "unknown",
        "total_requests": total_requests,
        "success_count": _non_negative_int(row.get("success_count")),
        "failed_count": failed_count,
        "skipped_no_route_count": _non_negative_int(row.get("skipped_no_route_count")),
        "leased_count": _non_negative_int(row.get("leased_count")),
        "retryable_count": _non_negative_int(row.get("retryable_count")),
        "network_timeout_count": _non_negative_int(row.get("network_timeout_count")),
        "proxy_failed_count": _non_negative_int(row.get("proxy_failed_count")),
        "read_timeout_count": _non_negative_int(row.get("read_timeout_count")),
        "connect_timeout_count": _non_negative_int(row.get("connect_timeout_count")),
        "write_timeout_count": _non_negative_int(row.get("write_timeout_count")),
        "pool_timeout_count": _non_negative_int(row.get("pool_timeout_count")),
        "timeout_count": _non_negative_int(row.get("timeout_count")),
        "request_stage_timeout_count": _non_negative_int(row.get("request_stage_timeout_count")),
        "response_headers_timeout_count": _non_negative_int(row.get("response_headers_timeout_count")),
        "response_body_timeout_count": _non_negative_int(row.get("response_body_timeout_count")),
        "unknown_stage_timeout_count": _non_negative_int(row.get("unknown_stage_timeout_count")),
        "request_bytes": _non_negative_int(row.get("request_bytes")),
        "response_bytes": _non_negative_int(row.get("response_bytes")),
        "media_count": _non_negative_int(row.get("media_count")),
        "image_count": _non_negative_int(row.get("image_count")),
        "avg_latency_ms": _optional_non_negative_int(row.get("avg_latency_ms")),
        "p95_latency_ms": _optional_non_negative_int(row.get("p95_latency_ms")),
        "avg_provider_total_ms": _optional_non_negative_int(row.get("avg_provider_total_ms")),
        "p95_provider_total_ms": _optional_non_negative_int(row.get("p95_provider_total_ms")),
        "avg_request_prepare_ms": _optional_non_negative_int(row.get("avg_request_prepare_ms")),
        "p95_request_prepare_ms": _optional_non_negative_int(row.get("p95_request_prepare_ms")),
        "avg_response_headers_ms": _optional_non_negative_int(row.get("avg_response_headers_ms")),
        "p95_response_headers_ms": _optional_non_negative_int(row.get("p95_response_headers_ms")),
        "avg_response_body_ms": _optional_non_negative_int(row.get("avg_response_body_ms")),
        "p95_response_body_ms": _optional_non_negative_int(row.get("p95_response_body_ms")),
        "avg_response_parse_ms": _optional_non_negative_int(row.get("avg_response_parse_ms")),
        "p95_response_parse_ms": _optional_non_negative_int(row.get("p95_response_parse_ms")),
        "failure_rate": _failure_rate(failed_count, total_requests),
    }


def _add_counts(proxy: dict[str, Any], route: Mapping[str, Any]) -> None:
    count_keys = (
        "total_requests",
        "success_count",
        "failed_count",
        "skipped_no_route_count",
        "leased_count",
        "retryable_count",
        "network_timeout_count",
        "proxy_failed_count",
        "read_timeout_count",
        "connect_timeout_count",
        "write_timeout_count",
        "pool_timeout_count",
        "timeout_count",
        "request_stage_timeout_count",
        "response_headers_timeout_count",
        "response_body_timeout_count",
        "unknown_stage_timeout_count",
        "request_bytes",
        "response_bytes",
        "media_count",
        "image_count",
    )
    for key in count_keys:
        proxy[key] += _non_negative_int(route.get(key))


def _update_proxy_state(proxy: dict[str, Any], row: Mapping[str, Any]) -> None:
    proxy_status = _optional_non_empty_string(row.get("proxy_status"))
    if proxy_status is not None:
        proxy["proxy_status"] = proxy_status
    proxy["active_cooldown_count"] = max(
        proxy["active_cooldown_count"],
        _non_negative_int(row.get("active_cooldown_count")),
    )
    proxy["max_cooldown_level"] = _max_optional_int(
        proxy.get("max_cooldown_level"),
        row.get("max_cooldown_level"),
    )
    cooldown_until = _serialize_time(row.get("cooldown_until"))
    if cooldown_until is not None:
        proxy["cooldown_until"] = cooldown_until


def _update_proxy_timing(proxy: dict[str, Any], route: Mapping[str, Any], *, previous_request_count: int) -> None:
    route_requests = _non_negative_int(route.get("total_requests"))
    for field_name in _AVG_TIMING_FIELDS:
        proxy[field_name] = _weighted_average(
            current_value=proxy.get(field_name),
            current_weight=previous_request_count,
            next_value=route.get(field_name),
            next_weight=route_requests,
        )
    for route_field_name, proxy_field_name in _PROXY_P95_TIMING_FIELDS.items():
        proxy[proxy_field_name] = _max_optional_int(proxy.get(proxy_field_name), route.get(route_field_name))


def _timing_metrics(row: Mapping[str, Any]) -> dict[str, int | None]:
    return {
        field_name: _optional_non_negative_int(row.get(field_name))
        for field_name in (*_AVG_TIMING_FIELDS, *_P95_TIMING_FIELDS)
    }


def _finalize_proxy(proxy: dict[str, Any]) -> dict[str, Any]:
    proxy["failure_rate"] = _failure_rate(proxy["failed_count"], proxy["total_requests"])
    proxy["routes"] = sorted(proxy["routes"], key=lambda item: (-item["total_requests"], item["route_label"]))
    return proxy


def _weighted_average(
    *,
    current_value: Any,
    current_weight: int,
    next_value: Any,
    next_weight: int,
) -> int | None:
    current_int = _optional_non_negative_int(current_value)
    next_int = _optional_non_negative_int(next_value)
    if current_int is None:
        return next_int
    if next_int is None or next_weight <= 0:
        return current_int
    total_weight = max(0, current_weight) + next_weight
    if total_weight <= 0:
        return next_int
    return int(round(((current_int * max(0, current_weight)) + (next_int * next_weight)) / total_weight))


def _failure_rate(failed_count: int, total_requests: int) -> float:
    if total_requests <= 0:
        return 0.0
    return round(failed_count / total_requests, 3)


def _max_optional_int(current_value: Any, next_value: Any) -> int | None:
    current_int = _optional_non_negative_int(current_value)
    next_int = _optional_non_negative_int(next_value)
    if current_int is None:
        return next_int
    if next_int is None:
        return current_int
    return max(current_int, next_int)


def _safe_label(value: Any, *, fallback: str) -> str:
    return _optional_non_empty_string(value) or fallback


def _window_params(window: MonitorWindow) -> dict[str, Any]:
    return {
        "minutes": window.minutes,
        "bucket_seconds": window.bucket_seconds,
        "model": window.model,
        "proxy_label": window.proxy_label,
    }


def _clamp_int(value: Any, *, minimum: int, maximum: int, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, parsed))


def _non_negative_int(value: Any) -> int:
    parsed = _optional_non_negative_int(value)
    return parsed if parsed is not None else 0


def _optional_non_negative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        value = int(value)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _serialize_time(value: Any) -> str | None:
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return aware.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
