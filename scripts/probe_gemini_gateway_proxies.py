from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import TextClause

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gemini_gateway.security import (  # noqa: E402
    ProxySecretParts,
    SecretDecryptionError,
    SecretVault,
    build_proxy_url,
)
from scripts.gemini_gateway_cli_errors import gateway_cli_error_payload  # noqa: E402

_DEFAULT_TARGET_URL = "https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta"
_DEFAULT_TIMEOUT_SECONDS = 20.0
_DEFAULT_CONCURRENCY = 1
_DEFAULT_LIMIT = 50
_SAFE_PROBE_ERROR = "Не удалось проверить proxy routes. Проверьте Postgres, ключ шифрования и сеть."


@dataclass(frozen=True)
class ProxyProbeTarget:
    """Маршрут для сетевой проверки без раскрытия proxy URL наружу."""

    route_label: str | None
    key_label: str | None
    proxy_label: str | None
    key_status: str | None
    binding_status: str | None
    proxy_status: str | None
    proxy_url: str | None


async def probe_proxy_targets(
    targets: Iterable[ProxyProbeTarget],
    *,
    target_url: str = _DEFAULT_TARGET_URL,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    concurrency: int = _DEFAULT_CONCURRENCY,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Проверяет proxy routes и возвращает безопасный JSON-ready отчет."""

    normalized_targets = list(targets)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(target: ProxyProbeTarget) -> dict[str, Any]:
        async with semaphore:
            return await probe_proxy_target(
                target,
                target_url=target_url,
                timeout_seconds=timeout_seconds,
                transport=transport,
            )

    results = await asyncio.gather(*(run_one(target) for target in normalized_targets))
    reason_counts = Counter(result["reason"] for result in results if result.get("reason"))
    timeout_reason_counts = Counter(result["timeout_reason"] for result in results if result.get("timeout_reason"))
    ok_count = sum(1 for result in results if result["status"] == "ok")
    error_count = len(results) - ok_count
    return {
        "status": "ok" if error_count == 0 else "error",
        "target_host": _safe_target_host(target_url),
        "total": len(results),
        "ok_count": ok_count,
        "error_count": error_count,
        "reason_counts": dict(sorted(reason_counts.items())),
        "timeout_reason_counts": dict(sorted(timeout_reason_counts.items())),
        "results": results,
    }


async def probe_proxy_target(
    target: ProxyProbeTarget,
    *,
    target_url: str,
    timeout_seconds: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Проверяет один proxy route, не возвращая proxy URL или credentials."""

    event = _base_probe_result(target)
    if not target.proxy_url:
        event.update(status="error", reason="secret_unavailable", latency_ms=0, http_status_family=None)
        return event

    started_at = perf_counter()
    try:
        client_kwargs: dict[str, Any] = {
            "timeout": timeout_seconds,
            "follow_redirects": False,
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        else:
            client_kwargs["proxy"] = target.proxy_url
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.get(target_url)
        event.update(
            status="ok",
            reason=None,
            latency_ms=_elapsed_ms(started_at),
            http_status_family=_status_family(response.status_code),
        )
    except Exception as exc:
        timeout_reason = _safe_timeout_reason(exc)
        event.update(
            status="error",
            reason=_safe_http_error_reason(exc),
            timeout_reason=timeout_reason,
            latency_ms=_elapsed_ms(started_at),
            http_status_family=None,
        )
    return event


async def fetch_proxy_probe_targets(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    vault: SecretVault,
    model: str | None = None,
    only_active_routes: bool = True,
    route_labels: Sequence[str] | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> list[ProxyProbeTarget]:
    """Читает proxy routes для диагностики без возврата секретов из функции."""

    normalized_model = _optional_str(model)
    normalized_route_labels = _normalize_route_labels(route_labels)
    query = _build_fetch_proxy_probe_targets_query(
        model=normalized_model,
        only_active_routes=only_active_routes,
        route_labels=normalized_route_labels,
    )
    params: dict[str, Any] = {"limit": max(1, limit)}
    if normalized_model is not None:
        params["model"] = normalized_model
    if normalized_route_labels:
        params["route_labels"] = normalized_route_labels

    async with session_factory() as session:
        rows = (await session.execute(query, params)).mappings().all()

    return [_target_from_row(dict(row), vault=vault) for row in rows]


def _build_fetch_proxy_probe_targets_query(
    *,
    model: str | None,
    only_active_routes: bool,
    route_labels: Sequence[str] | None,
) -> TextClause:
    """Собирает SQL чтения probe targets с опциональным фильтром model pool."""

    has_model_filter = _optional_str(model) is not None
    has_route_label_filter = bool(_normalize_route_labels(route_labels))
    model_join = (
        """
        JOIN gemini_gateway.google_projects p ON p.id = k.project_id
        JOIN gemini_gateway.model_limits ml ON ml.project_id = p.id AND ml.model = :model
        """
        if has_model_filter
        else ""
    )
    active_conditions = [
        "k.status = 'active'",
        "b.status = 'active'",
        "px.status = 'active'",
    ]
    if has_model_filter:
        active_conditions.extend(
            [
                "p.status = 'active'",
                "ml.status = 'active'",
            ]
    )
    status_filter = "\n".join(f"      AND {condition}" for condition in active_conditions)
    route_label_filter = "\n      AND b.label = ANY(CAST(:route_labels AS text[]))" if has_route_label_filter else ""
    return text(
        f"""
        SELECT
            b.label AS route_label,
            k.label AS key_label,
            px.label AS proxy_label,
            k.status AS key_status,
            b.status AS binding_status,
            px.status AS proxy_status,
            px.scheme,
            px.host,
            px.port,
            px.encrypted_username,
            px.encrypted_password
        FROM gemini_gateway.key_proxy_bindings b
        JOIN gemini_gateway.api_keys k ON k.id = b.api_key_id
        JOIN gemini_gateway.proxy_endpoints px ON px.id = b.proxy_id
        {model_join}
        WHERE b.transport_mode = 'proxy'
        {" " + status_filter if only_active_routes else ""}
        {route_label_filter}
        ORDER BY b.label
        LIMIT :limit
        """
    )


async def probe_gemini_gateway_proxies(
    *,
    model: str | None = None,
    target_url: str = _DEFAULT_TARGET_URL,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    concurrency: int = _DEFAULT_CONCURRENCY,
    limit: int = _DEFAULT_LIMIT,
    only_active_routes: bool = True,
    route_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    from gemini_gateway.config import get_gateway_settings
    from gemini_gateway.db.session import create_gateway_engine

    settings = get_gateway_settings()
    engine = create_gateway_engine(settings)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    vault = SecretVault(encryption_key=settings.encryption_key, hmac_key=settings.hmac_key)
    try:
        targets = await fetch_proxy_probe_targets(
            session_factory,
            vault=vault,
            model=model,
            only_active_routes=only_active_routes,
            route_labels=route_labels,
            limit=limit,
        )
        report = await probe_proxy_targets(
            targets,
            target_url=target_url,
            timeout_seconds=timeout_seconds,
            concurrency=concurrency,
        )
        normalized_model = _optional_str(model)
        if normalized_model is not None:
            report["model"] = normalized_model
        normalized_route_labels = _normalize_route_labels(route_labels)
        if normalized_route_labels:
            report["route_labels"] = normalized_route_labels
        return report
    finally:
        await engine.dispose()


def _target_from_row(row: dict[str, Any], *, vault: SecretVault) -> ProxyProbeTarget:
    proxy_url: str | None
    try:
        username = vault.decrypt(row.get("encrypted_username"))
        password = vault.decrypt(row.get("encrypted_password"))
        proxy_url = build_proxy_url(
            ProxySecretParts(
                scheme=str(row["scheme"]),
                host=str(row["host"]),
                port=int(row["port"]),
                username=username,
                password=password,
            )
        )
    except (KeyError, TypeError, ValueError, SecretDecryptionError):
        proxy_url = None

    return ProxyProbeTarget(
        route_label=_optional_str(row.get("route_label")),
        key_label=_optional_str(row.get("key_label")),
        proxy_label=_optional_str(row.get("proxy_label")),
        key_status=_optional_str(row.get("key_status")),
        binding_status=_optional_str(row.get("binding_status")),
        proxy_status=_optional_str(row.get("proxy_status")),
        proxy_url=proxy_url,
    )


def _base_probe_result(target: ProxyProbeTarget) -> dict[str, Any]:
    return {
        "route_label": target.route_label,
        "key_label": target.key_label,
        "proxy_label": target.proxy_label,
        "key_status": target.key_status,
        "binding_status": target.binding_status,
        "proxy_status": target.proxy_status,
    }


def _safe_http_error_reason(error: Exception) -> str:
    if isinstance(error, httpx.TimeoutException):
        return "network_timeout"
    if isinstance(error, (httpx.ProxyError, httpx.ConnectError)):
        return "proxy_failed"
    if isinstance(error, httpx.HTTPError):
        return "request_failed"
    return "request_failed"


def _safe_timeout_reason(error: Exception) -> str | None:
    if isinstance(error, httpx.ConnectTimeout):
        return "connect_timeout"
    if isinstance(error, httpx.ReadTimeout):
        return "read_timeout"
    if isinstance(error, httpx.WriteTimeout):
        return "write_timeout"
    if isinstance(error, httpx.PoolTimeout):
        return "pool_timeout"
    if isinstance(error, httpx.TimeoutException):
        return "timeout"
    return None


def _status_family(status_code: int) -> str:
    return f"{max(0, status_code) // 100}xx"


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((perf_counter() - started_at) * 1000))


def _safe_target_host(target_url: str) -> str | None:
    try:
        parsed = httpx.URL(target_url)
    except httpx.InvalidURL:
        return None
    return parsed.host


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_route_labels(route_labels: Sequence[str] | None) -> list[str]:
    if not route_labels:
        return []
    return sorted({item for value in route_labels if (item := _optional_str(value))})


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Gemini Gateway proxy routes without exposing proxy secrets.")
    parser.add_argument("--model", default=None, help="Probe only routes that belong to this active model pool.")
    parser.add_argument(
        "--route-label",
        dest="route_labels",
        action="append",
        default=None,
        help="Probe only the selected route label; repeat the option for multiple routes.",
    )
    parser.add_argument("--target-url", default=_DEFAULT_TARGET_URL)
    parser.add_argument("--timeout-seconds", type=float, default=_DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--concurrency", type=int, default=_DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=_DEFAULT_LIMIT)
    parser.add_argument(
        "--include-inactive-routes",
        action="store_true",
        help="Probe routes even when api key, binding or proxy status is not active.",
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
            probe_gemini_gateway_proxies(
                model=args.model,
                target_url=args.target_url,
                timeout_seconds=max(1.0, args.timeout_seconds),
                concurrency=max(1, args.concurrency),
                limit=max(1, args.limit),
                only_active_routes=not args.include_inactive_routes,
                route_labels=args.route_labels,
            )
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(
            json.dumps(
                gateway_cli_error_payload(exc, fallback_error=_SAFE_PROBE_ERROR),
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
