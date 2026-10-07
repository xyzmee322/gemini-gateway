from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from math import ceil
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.string_parsing import parse_optional_string as _optional_string
from gemini_gateway.contracts import (
    GatewayChatRequest,
    GatewayEmbeddingRequest,
    GatewayErrorReason,
    GatewayRouteRequest,
    GatewayTTSRequest,
)
from gemini_gateway.errors import GatewayError, public_message_for_reason, public_provider_reason
from gemini_gateway.monitoring import (
    MonitorWindow,
    fetch_proxy_summary,
    fetch_proxy_timeseries,
    summarize_proxy_overview_rows,
    summarize_proxy_timeseries_rows,
)
from gemini_gateway.service import create_default_service

_LOGGER = logging.getLogger(__name__)
_MONITOR_ERROR_MESSAGE = "Не удалось загрузить мониторинг, попробуйте позже"
_SAFE_MONITOR_KEYS = frozenset(
    {
        "avg_response_headers_ms",
        "avg_response_body_ms",
        "max_route_p95_provider_total_ms",
        "max_route_p95_request_prepare_ms",
        "max_route_p95_response_headers_ms",
        "max_route_p95_response_body_ms",
        "max_route_p95_response_parse_ms",
        "p95_response_headers_ms",
        "p95_response_body_ms",
        "request_bytes",
        "response_bytes",
        "response_headers_timeout_count",
        "response_body_timeout_count",
    }
)
_FORBIDDEN_MONITOR_TOKENS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "base64",
        "credential",
        "cookie",
        "encrypted",
        "fingerprint",
        "header",
        "host",
        "password",
        "port",
        "prompt",
        "provider_payload",
        "provider_response_json",
        "proxy_fingerprint",
        "proxy_url",
        "raw_provider_payload",
        "raw_response",
        "request_body",
        "response_body",
        "response_text",
        "secret",
        "stack",
        "token",
        "traceback",
        "username",
    }
)
_MONITOR_REDACTED_VALUE = "[redacted]"
_SENSITIVE_MONITOR_VALUE_PATTERNS = (
    re.compile(r"\b(?:https?|socks[45]?)://", re.IGNORECASE),
    re.compile(r"\b(?:bearer|basic)\s+[a-z0-9._~+/=-]{8,}", re.IGNORECASE),
    re.compile(r"\bAIza[0-9A-Za-z_-]{10,}\b"),
    re.compile(r"\bsk-[0-9A-Za-z_-]{10,}\b", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|password|secret|token)\s*[:=]", re.IGNORECASE),
    re.compile(r"^[A-Za-z0-9+/]{80,}={0,2}$"),
)


ReadinessCheck = Callable[[], Awaitable[dict[str, Any]] | dict[str, Any]]
MonitorFetcher = Callable[[MonitorWindow], Awaitable[dict[str, Any]] | dict[str, Any]]


def create_app(
    *,
    auth_token: str,
    completion_service: Any | None = None,
    readiness_check: ReadinessCheck | None = None,
    monitoring_session_factory: async_sessionmaker[AsyncSession] | None = None,
    monitoring_summary_fetcher: MonitorFetcher | None = None,
    monitoring_timeseries_fetcher: MonitorFetcher | None = None,
    service_name: str = "gemini-gateway",
    environment: str = "development",
) -> FastAPI:
    app = FastAPI(title="Gemini Gateway", docs_url=None, redoc_url=None)
    event_environment = environment or "development"
    event_service_name = service_name or "gemini-gateway"
    service = completion_service or create_default_service(
        service_name=event_service_name,
        environment=event_environment,
    )

    @app.get("/health/live")
    async def health_live() -> JSONResponse:
        return JSONResponse(status_code=200, content={"status": "ok"})

    @app.get("/health")
    async def health() -> JSONResponse:
        try:
            readiness = await _resolve_readiness(readiness_check)
        except Exception as exc:
            _log_gateway_api_error(
                event="gemini_gateway_health_error",
                service_name=event_service_name,
                environment=event_environment,
                request_id=None,
                error=exc,
                endpoint="health",
                reason="readiness_check_failed",
                response_reason="health_unready",
                retryable=True,
                failed_stage="readiness_check",
            )
            return JSONResponse(status_code=503, content={"status": "unready", "checks": {}})

        if not readiness.get("ok", False):
            return JSONResponse(
                status_code=503,
                content={"status": "unready", "checks": readiness.get("checks", {})},
            )
        return JSONResponse(status_code=200, content={"status": "ok", "checks": readiness.get("checks", {})})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
        if not _is_authorized(authorization=authorization, auth_token=auth_token):
            return _error_response(
                request_id=None,
                status_code=401,
                reason="unauthorized",
                retryable=False,
            )

        body = await _safe_body(request)
        request_id = _request_id_from_body(body)

        try:
            gateway_request = _parse_gateway_request(body)
        except ValidationError:
            return _error_response(
                request_id=request_id,
                status_code=400,
                reason="bad_request",
                retryable=False,
            )

        try:
            response = await service.complete(gateway_request)
        except GatewayError as error:
            return _error_response(
                request_id=error.request_id or gateway_request.request_id,
                status_code=getattr(error, "status_code", 503),
                reason=error.reason,
                retryable=error.retryable,
                public_message=error.public_message,
                provider_reason=public_provider_reason(error.provider_message_safe),
                provider_status_code=error.provider_status_code,
                retry_after_seconds=_gateway_error_retry_after_seconds(error),
                route_context=_gateway_error_route_context(error),
                diagnostics=_gateway_error_diagnostics(error),
            )
        except Exception as exc:
            _log_gateway_api_error(
                event="gemini_gateway_api_error",
                service_name=event_service_name,
                environment=event_environment,
                request_id=gateway_request.request_id,
                error=exc,
                endpoint="chat_completions",
                gateway_request=gateway_request,
                reason="unhandled_exception",
                response_reason="request_failed",
                retryable=True,
                failed_stage="chat_completion_handler",
            )
            return _error_response(
                request_id=gateway_request.request_id,
                status_code=500,
                reason="request_failed",
                retryable=True,
            )

        return JSONResponse(status_code=200, content=_dump(response))

    @app.post("/v1/embeddings")
    async def embeddings(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
        if not _is_authorized(authorization=authorization, auth_token=auth_token):
            return _error_response(
                request_id=None,
                status_code=401,
                reason="unauthorized",
                retryable=False,
            )

        body = await _safe_body(request)
        request_id = _request_id_from_body(body)

        try:
            gateway_request = _parse_embedding_request(body)
        except ValidationError:
            return _error_response(
                request_id=request_id,
                status_code=400,
                reason="bad_request",
                retryable=False,
            )

        try:
            response = await service.embed(gateway_request)
        except GatewayError as error:
            return _error_response(
                request_id=error.request_id or gateway_request.request_id,
                status_code=getattr(error, "status_code", 503),
                reason=error.reason,
                retryable=error.retryable,
                public_message=error.public_message,
                provider_reason=public_provider_reason(error.provider_message_safe),
                provider_status_code=error.provider_status_code,
                retry_after_seconds=_gateway_error_retry_after_seconds(error),
                route_context=_gateway_error_route_context(error),
                diagnostics=_gateway_error_diagnostics(error),
            )
        except Exception as exc:
            _log_gateway_api_error(
                event="gemini_gateway_embeddings_api_error",
                service_name=event_service_name,
                environment=event_environment,
                request_id=gateway_request.request_id,
                error=exc,
                endpoint="embeddings",
                gateway_request=gateway_request,
                reason="unhandled_exception",
                response_reason="request_failed",
                retryable=True,
                failed_stage="embeddings_handler",
            )
            return _error_response(
                request_id=gateway_request.request_id,
                status_code=500,
                reason="request_failed",
                retryable=True,
            )

        return JSONResponse(status_code=200, content=_dump(response))

    @app.post("/v1/audio/speech")
    async def audio_speech(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
        if not _is_authorized(authorization=authorization, auth_token=auth_token):
            return _error_response(
                request_id=None,
                status_code=401,
                reason="unauthorized",
                retryable=False,
            )

        body = await _safe_body(request)
        request_id = _request_id_from_body(body)

        try:
            gateway_request = _parse_tts_request(body)
        except ValidationError:
            return _error_response(
                request_id=request_id,
                status_code=400,
                reason="bad_request",
                retryable=False,
            )

        try:
            response = await service.synthesize_speech(gateway_request)
        except GatewayError as error:
            return _error_response(
                request_id=error.request_id or gateway_request.request_id,
                status_code=getattr(error, "status_code", 503),
                reason=error.reason,
                retryable=error.retryable,
                public_message=error.public_message,
                provider_reason=public_provider_reason(error.provider_message_safe),
                provider_status_code=error.provider_status_code,
                retry_after_seconds=_gateway_error_retry_after_seconds(error),
                route_context=_gateway_error_route_context(error),
                diagnostics=_gateway_error_diagnostics(error),
            )
        except Exception as exc:
            _log_gateway_api_error(
                event="gemini_gateway_tts_api_error",
                service_name=event_service_name,
                environment=event_environment,
                request_id=gateway_request.request_id,
                error=exc,
                endpoint="audio_speech",
                gateway_request=gateway_request,
                reason="unhandled_exception",
                response_reason="request_failed",
                retryable=True,
                failed_stage="tts_handler",
            )
            return _error_response(
                request_id=gateway_request.request_id,
                status_code=500,
                reason="request_failed",
                retryable=True,
            )

        return JSONResponse(status_code=200, content=_dump(response))

    @app.get("/admin/monitor", response_class=HTMLResponse)
    async def monitor_dashboard() -> HTMLResponse:
        return HTMLResponse(status_code=200, content=_monitor_dashboard_html())

    @app.get("/admin/monitor/api/summary")
    async def monitor_summary(
        authorization: str | None = Header(default=None),
        minutes: str = "180",
        model: str | None = None,
        proxy_label: str | None = None,
    ) -> JSONResponse:
        if not _is_authorized(authorization=authorization, auth_token=auth_token):
            return JSONResponse(status_code=401, content={"error": "Недостаточно прав для выполнения запроса"})

        window = MonitorWindow(minutes=minutes, model=model, proxy_label=proxy_label)
        try:
            payload = await _resolve_monitor_summary(
                session_factory=monitoring_session_factory,
                fetcher=monitoring_summary_fetcher,
                window=window,
            )
        except Exception as exc:
            _log_monitor_error(
                service_name=event_service_name,
                environment=event_environment,
                endpoint="monitor_summary",
                error=exc,
            )
            return JSONResponse(status_code=500, content={"error": _MONITOR_ERROR_MESSAGE})
        return JSONResponse(status_code=200, content=payload)

    @app.get("/admin/monitor/api/timeseries")
    async def monitor_timeseries(
        authorization: str | None = Header(default=None),
        minutes: str = "180",
        bucket_seconds: str = "60",
        model: str | None = None,
        proxy_label: str | None = None,
    ) -> JSONResponse:
        if not _is_authorized(authorization=authorization, auth_token=auth_token):
            return JSONResponse(status_code=401, content={"error": "Недостаточно прав для выполнения запроса"})

        window = MonitorWindow(
            minutes=minutes,
            bucket_seconds=bucket_seconds,
            model=model,
            proxy_label=proxy_label,
        )
        try:
            payload = await _resolve_monitor_timeseries(
                session_factory=monitoring_session_factory,
                fetcher=monitoring_timeseries_fetcher,
                window=window,
            )
        except Exception as exc:
            _log_monitor_error(
                service_name=event_service_name,
                environment=event_environment,
                endpoint="monitor_timeseries",
                error=exc,
            )
            return JSONResponse(status_code=500, content={"error": _MONITOR_ERROR_MESSAGE})
        return JSONResponse(status_code=200, content=payload)

    return app


def _monitor_dashboard_html() -> str:
    return """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gemini Proxy Monitor</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f6;
      --panel: #ffffff;
      --ink: #182322;
      --muted: #65706d;
      --line: #d7dedb;
      --accent: #176a64;
      --accent-2: #b35332;
      --ok: #197347;
      --bad: #b93535;
      --warn: #9a6614;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Trebuchet MS", Verdana, sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }
    main { max-width: 1320px; margin: 0 auto; padding: 24px; }
    header {
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-end;
      border-bottom: 2px solid var(--ink);
      padding-bottom: 14px;
      margin-bottom: 18px;
    }
    h1 { margin: 0; font-size: 30px; letter-spacing: 0; }
    .endpoints { color: var(--muted); font-family: "Courier New", monospace; font-size: 12px; text-align: right; }
    form {
      display: grid;
      grid-template-columns: minmax(220px, 1.3fr) repeat(5, minmax(110px, 1fr)) minmax(96px, .6fr) auto;
      gap: 10px;
      align-items: end;
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 14px;
      margin-bottom: 16px;
    }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; }
    .checkbox {
      display: flex;
      align-items: center;
      min-height: 38px;
      gap: 8px;
      color: var(--ink);
      border: 1px solid var(--line);
      background: #fff;
      padding: 0 10px;
    }
    input {
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 9px 10px;
      min-height: 38px;
      font: inherit;
    }
    input[type="checkbox"] { width: 16px; min-height: 16px; }
    button {
      border: 0;
      background: var(--accent);
      color: #fff;
      min-height: 38px;
      padding: 0 16px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled { opacity: .55; cursor: wait; }
    .message {
      min-height: 22px;
      color: var(--accent-2);
      font-weight: 700;
      margin: 8px 0 12px;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 12px;
      min-height: 86px;
    }
    .card span { color: var(--muted); display: block; font-size: 12px; }
    .card strong { display: block; font-size: 24px; margin-top: 6px; }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      margin: 16px 0;
      padding: 14px;
      overflow: auto;
    }
    h2 { margin: 0 0 12px; font-size: 18px; }
    table { width: 100%; border-collapse: collapse; min-width: 980px; }
    th, td { border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 12px; font-weight: 700; }
    .status-ok { color: var(--ok); font-weight: 700; }
    .status-bad { color: var(--bad); font-weight: 700; }
    .status-warn { color: var(--warn); font-weight: 700; }
    .stage-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(150px, 1fr));
      gap: 10px;
      min-width: 760px;
    }
    .stage {
      border: 1px solid var(--line);
      padding: 10px;
      background: #fff;
    }
    .stage span { color: var(--muted); display: block; font-size: 12px; }
    .stage strong { display: block; margin-top: 5px; font-size: 18px; }
    canvas {
      display: block;
      width: 100%;
      height: 260px;
      border: 1px solid var(--line);
      background: #fff;
    }
    .empty { color: var(--muted); padding: 10px 0; }
    @media (max-width: 900px) {
      main { padding: 14px; }
      header { display: block; }
      .endpoints { text-align: left; margin-top: 8px; }
      form { grid-template-columns: 1fr; }
      .cards { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Gemini Proxy Monitor</h1>
      <div class="empty">Введите токен мониторинга, чтобы загрузить защищенные метрики.</div>
    </div>
    <div class="endpoints">
      <div>/admin/monitor/api/summary</div>
      <div>/admin/monitor/api/timeseries</div>
    </div>
  </header>

  <form id="monitor-form">
    <label>token
      <input name="token" type="password" autocomplete="off" placeholder="Введите токен мониторинга">
    </label>
    <label>minutes
      <input name="minutes" type="number" min="1" max="10080" step="1" value="180">
    </label>
    <label>bucket_seconds
      <input name="bucket_seconds" type="number" min="10" max="3600" step="10" value="60">
    </label>
    <label>model
      <input name="model" type="text" autocomplete="off" placeholder="all">
    </label>
    <label>proxy_label
      <input name="proxy_label" type="text" autocomplete="off" placeholder="all">
    </label>
    <label>refresh_seconds
      <input name="refresh_seconds" type="number" min="5" max="300" step="5" value="15">
    </label>
    <label class="checkbox">
      <input name="auto_refresh" type="checkbox" checked>
      auto
    </label>
    <button type="submit">Обновить</button>
  </form>

  <div id="message" class="message" role="status"></div>
  <div id="cards" class="cards"></div>

  <section>
    <h2>Proxy summary</h2>
    <div id="proxy-table"></div>
  </section>

  <section>
    <h2>Latency stages</h2>
    <div id="stage-panel" class="stage-grid"></div>
  </section>

  <section>
    <h2>Timeseries</h2>
    <canvas id="timeseries-chart" width="1100" height="260"></canvas>
  </section>
</main>
<script>
(() => {
  "use strict";

  const fallbackMessage = "Не удалось загрузить мониторинг, попробуйте позже";
  const storageKey = "gemini_proxy_monitor_auth";
  const refreshStorageKey = "gemini_proxy_monitor_refresh_seconds";
  const autoRefreshStorageKey = "gemini_proxy_monitor_auto_refresh";
  const form = document.getElementById("monitor-form");
  const message = document.getElementById("message");
  const cards = document.getElementById("cards");
  const proxyTable = document.getElementById("proxy-table");
  const stagePanel = document.getElementById("stage-panel");
  const chart = document.getElementById("timeseries-chart");
  const submitButton = form.querySelector("button");
  let refreshTimer = null;
  let activeLoad = null;

  class MonitorApiError extends Error {}

  form.elements.token.value = localStorage.getItem(storageKey) || "";
  form.elements.refresh_seconds.value = localStorage.getItem(refreshStorageKey) || "15";
  form.elements.auto_refresh.checked = localStorage.getItem(autoRefreshStorageKey) !== "false";
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    loadMonitoring();
  });
  form.elements.auto_refresh.addEventListener("change", scheduleAutoRefresh);
  form.elements.refresh_seconds.addEventListener("change", scheduleAutoRefresh);

  function buildQuery() {
    const params = new URLSearchParams();
    appendParam(params, "minutes", form.elements.minutes.value);
    appendParam(params, "bucket_seconds", form.elements.bucket_seconds.value);
    appendParam(params, "model", form.elements.model.value);
    appendParam(params, "proxy_label", form.elements.proxy_label.value);
    return params;
  }

  function appendParam(params, key, value) {
    const normalized = String(value || "").trim();
    if (normalized) {
      params.set(key, normalized);
    }
  }

  async function loadMonitoring() {
    if (activeLoad) {
      return activeLoad;
    }
    activeLoad = loadMonitoringOnce();
    try {
      return await activeLoad;
    } finally {
      activeLoad = null;
    }
  }

  async function loadMonitoringOnce() {
    const token = String(form.elements.token.value || "").trim();
    if (!token) {
      showError("Введите токен мониторинга");
      return;
    }

    localStorage.setItem(storageKey, token);
    localStorage.setItem(refreshStorageKey, String(refreshSeconds()));
    localStorage.setItem(autoRefreshStorageKey, form.elements.auto_refresh.checked ? "true" : "false");
    setLoading(true);
    showError("");

    try {
      const params = buildQuery();
      const summaryParams = new URLSearchParams(params);
      summaryParams.delete("bucket_seconds");
      const [summary, timeseries] = await Promise.all([
        fetchJson("/admin/monitor/api/summary?" + summaryParams.toString(), token),
        fetchJson("/admin/monitor/api/timeseries?" + params.toString(), token),
      ]);
      renderSummary(summary || {});
      renderTimeseries(timeseries || {});
    } catch (error) {
      renderSummary({});
      renderTimeseries({});
      showError(publicError(error));
    } finally {
      setLoading(false);
      scheduleAutoRefresh();
    }
  }

  function scheduleAutoRefresh() {
    if (refreshTimer) {
      clearInterval(refreshTimer);
      refreshTimer = null;
    }
    localStorage.setItem(refreshStorageKey, String(refreshSeconds()));
    localStorage.setItem(autoRefreshStorageKey, form.elements.auto_refresh.checked ? "true" : "false");
    if (!form.elements.auto_refresh.checked) {
      return;
    }
    refreshTimer = setInterval(() => {
      if (String(form.elements.token.value || "").trim()) {
        loadMonitoring();
      }
    }, refreshSeconds() * 1000);
  }

  function refreshSeconds() {
    const value = Number(form.elements.refresh_seconds.value);
    if (!Number.isFinite(value)) {
      return 15;
    }
    return Math.min(300, Math.max(5, Math.round(value)));
  }

  async function fetchJson(url, token) {
    const response = await fetch(url, {
      headers: { "Authorization": "Bearer " + token, "Accept": "application/json" },
    });
    let payload = {};
    try {
      payload = await response.json();
    } catch {
      payload = {};
    }
    if (!response.ok) {
      throw new MonitorApiError(safeText(payload.error));
    }
    return payload;
  }

  function renderSummary(summary) {
    const proxies = Array.isArray(summary.proxies) ? summary.proxies.filter(isPlainRecord) : [];
    cards.innerHTML = [
      card("requests", summary.total_requests),
      card("proxies", summary.proxy_count || proxies.length),
      card("failed", sum(proxies, "failed_count")),
      card("read_timeout", sum(proxies, "read_timeout_count")),
    ].join("");
    renderProxyTable(proxies);
    renderStages(proxies);
  }

  function renderProxyTable(proxies) {
    if (!proxies.length) {
      proxyTable.innerHTML = '<div class="empty">Нет данных за выбранный период.</div>';
      return;
    }
    const rows = proxies.map((proxy) => `
      <tr>
        <td>${escapeHtml(proxy.proxy_label || "direct")}</td>
        <td class="${statusClass(proxy)}">${escapeHtml(statusText(proxy))}</td>
        <td>${formatNumber(proxy.total_requests)}</td>
        <td>${formatNumber(proxy.failed_count)} / ${formatPercent(proxy.failure_rate)}</td>
        <td>${formatMs(proxy.max_route_p95_latency_ms)}</td>
        <td>${formatNumber(proxy.read_timeout_count)}</td>
        <td>${formatNumber(proxy.media_count)}</td>
        <td>${cooldownText(proxy)}</td>
      </tr>
    `).join("");
    proxyTable.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>proxy_label</th>
            <th>статус</th>
            <th>total_requests</th>
            <th>failed_count / failure rate</th>
            <th>max_route_p95_latency_ms</th>
            <th>read_timeout_count</th>
            <th>media_count</th>
            <th>cooldown</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  function renderStages(proxies) {
    const totals = averageStages(proxies);
    stagePanel.innerHTML = [
      stage("request_prepare_ms", totals.request_prepare_ms),
      stage("response_headers_ms", totals.response_headers_ms),
      stage("response_body_ms", totals.response_body_ms),
      stage("response_parse_ms", totals.response_parse_ms),
      stage("provider_total_ms", totals.provider_total_ms),
    ].join("");
  }

  function renderTimeseries(payload) {
    const points = Array.isArray(payload.series) ? payload.series.filter(isPlainRecord) : [];
    drawChart(points);
  }

  function drawChart(points) {
    const context = chart.getContext("2d");
    const width = chart.width;
    const height = chart.height;
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, width, height);
    context.strokeStyle = "#d7dedb";
    context.lineWidth = 1;
    for (let index = 0; index < 5; index += 1) {
      const y = 28 + index * 48;
      context.beginPath();
      context.moveTo(36, y);
      context.lineTo(width - 18, y);
      context.stroke();
    }
    if (!points.length) {
      context.fillStyle = "#6d7478";
      context.fillText("Нет данных за выбранный период.", 42, 132);
      return;
    }
    const values = points.map((point) => numberValue(point.p95_provider_total_ms || point.p95_latency_ms));
    const maxValue = Math.max(...values, 1);
    const plotWidth = width - 70;
    const plotHeight = height - 58;
    context.strokeStyle = "#12665f";
    context.lineWidth = 2;
    context.beginPath();
    points.forEach((point, index) => {
      const x = 42 + (points.length === 1 ? 0 : (plotWidth * index) / (points.length - 1));
      const y = 18 + plotHeight - (numberValue(point.p95_provider_total_ms || point.p95_latency_ms) / maxValue) * plotHeight;
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
    context.fillStyle = "#1d2528";
    context.fillText("p95 provider_total_ms", 42, 18);
    context.fillText(formatMs(maxValue), width - 110, 18);
  }

  function averageStages(proxies) {
    return {
      request_prepare_ms: avg(proxies, "max_route_p95_request_prepare_ms"),
      response_headers_ms: avg(proxies, "max_route_p95_response_headers_ms"),
      response_body_ms: avg(proxies, "max_route_p95_response_body_ms"),
      response_parse_ms: avg(proxies, "max_route_p95_response_parse_ms"),
      provider_total_ms: avg(proxies, "max_route_p95_provider_total_ms"),
    };
  }

  function card(label, value) {
    return `<div class="card"><span>${label}</span><strong>${formatNumber(value)}</strong></div>`;
  }

  function stage(label, value) {
    return `<div class="stage"><span>${label}</span><strong>${formatMs(value)}</strong></div>`;
  }

  function statusText(proxy) {
    if (proxy.proxy_status) {
      return String(proxy.proxy_status);
    }
    if (numberValue(proxy.active_cooldown_count) > 0 || proxy.cooldown_until) {
      return "cooldown";
    }
    if (numberValue(proxy.failed_count) > 0 && numberValue(proxy.success_count) === 0) {
      return "ошибка";
    }
    return "ok";
  }

  function statusClass(proxy) {
    const status = statusText(proxy).toLowerCase();
    if (status.includes("cooldown")) {
      return "status-warn";
    }
    if (status.includes("fail") || status.includes("ошиб") || status.includes("down")) {
      return "status-bad";
    }
    return "status-ok";
  }

  function cooldownText(proxy) {
    if (proxy.cooldown_until) {
      return escapeHtml(String(proxy.cooldown_until));
    }
    const count = numberValue(proxy.active_cooldown_count);
    return count > 0 ? String(count) : "нет";
  }

  function sum(items, key) {
    return items.reduce((total, item) => total + numberValue(item[key]), 0);
  }

  function avg(items, key) {
    const values = items.map((item) => numberValue(item[key])).filter((value) => value > 0);
    if (!values.length) {
      return null;
    }
    return values.reduce((total, value) => total + value, 0) / values.length;
  }

  function numberValue(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : 0;
  }

  function formatNumber(value) {
    return String(Math.round(numberValue(value)));
  }

  function formatMs(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "n/a";
    }
    return Math.round(numberValue(value)) + " ms";
  }

  function formatPercent(value) {
    return Math.round(numberValue(value) * 1000) / 10 + "%";
  }

  function safeText(value) {
    const text = String(value || "").trim();
    return text || fallbackMessage;
  }

  function publicError(error) {
    if (error instanceof MonitorApiError) {
      const text = safeText(error.message);
      return text.length > 180 ? fallbackMessage : text;
    }
    return fallbackMessage;
  }

  function showError(text) {
    message.textContent = text || "";
  }

  function setLoading(isLoading) {
    submitButton.disabled = isLoading;
    submitButton.textContent = isLoading ? "Загрузка" : "Обновить";
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function isPlainRecord(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  renderSummary({});
  renderTimeseries({});
  scheduleAutoRefresh();
  if (form.elements.token.value) {
    loadMonitoring();
  }
})();
</script>
</body>
</html>"""


async def _resolve_monitor_summary(
    *,
    session_factory: async_sessionmaker[AsyncSession] | None,
    fetcher: MonitorFetcher | None,
    window: MonitorWindow,
) -> dict[str, Any]:
    if fetcher is not None:
        return _sanitize_monitor_payload(await _maybe_await_monitor(fetcher(window)))
    if session_factory is None:
        return summarize_proxy_overview_rows([])
    return _sanitize_monitor_payload(await fetch_proxy_summary(session_factory, window))


async def _resolve_monitor_timeseries(
    *,
    session_factory: async_sessionmaker[AsyncSession] | None,
    fetcher: MonitorFetcher | None,
    window: MonitorWindow,
) -> dict[str, Any]:
    if fetcher is not None:
        return _sanitize_monitor_payload(await _maybe_await_monitor(fetcher(window)))
    if session_factory is None:
        return summarize_proxy_timeseries_rows([], bucket_seconds=window.bucket_seconds)
    return _sanitize_monitor_payload(await fetch_proxy_timeseries(session_factory, window))


async def _maybe_await_monitor(value: Awaitable[dict[str, Any]] | dict[str, Any]) -> dict[str, Any]:
    if hasattr(value, "__await__"):
        value = await value
    return value if isinstance(value, dict) else {}


def _sanitize_monitor_payload(payload: Any) -> dict[str, Any]:
    sanitized = _sanitize_monitor_value(payload)
    return sanitized if isinstance(sanitized, dict) else {}


def _sanitize_monitor_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_monitor_value(item)
            for key, item in value.items()
            if isinstance(key, str) and not _is_forbidden_monitor_key(key)
        }
    if isinstance(value, list):
        return [_sanitize_monitor_value(item) for item in value]
    if isinstance(value, str):
        return _sanitize_monitor_string(value)
    return value


def _sanitize_monitor_string(value: str) -> str:
    stripped_value = value.strip()
    if not stripped_value:
        return value
    if any(pattern.search(stripped_value) for pattern in _SENSITIVE_MONITOR_VALUE_PATTERNS):
        return _MONITOR_REDACTED_VALUE
    return value


def _is_forbidden_monitor_key(key: str) -> bool:
    normalized_key = _normalize_monitor_key(key)
    if normalized_key in _SAFE_MONITOR_KEYS:
        return False
    return any(token in normalized_key or normalized_key.endswith(token) for token in _FORBIDDEN_MONITOR_TOKENS)


def _normalize_monitor_key(key: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in key.lower()).strip("_")


def _log_monitor_error(
    *,
    service_name: str,
    environment: str,
    endpoint: str,
    error: Exception,
) -> None:
    _LOGGER.error(
        "gemini_gateway_monitor_error",
        extra={
            "event": "gemini_gateway_monitor_error",
            "service": service_name,
            "environment": environment,
            "status": "error",
            "reason": "monitor_fetch_failed",
            "response_reason": "monitor_unavailable",
            "retryable": True,
            "error_message": _MONITOR_ERROR_MESSAGE,
            "failed_stage": "monitor_fetch",
            "request_id": None,
            "endpoint": endpoint,
            "source_service": None,
            "model": None,
            "chat_id": None,
            "telegram_message_id": None,
            "error_type": type(error).__name__,
        },
    )


def _log_gateway_api_error(
    *,
    event: str,
    service_name: str,
    environment: str,
    request_id: str | None,
    error: Exception,
    endpoint: str,
    reason: str,
    response_reason: str,
    retryable: bool,
    failed_stage: str,
    gateway_request: GatewayRouteRequest | None = None,
) -> None:
    _LOGGER.error(
        event,
        extra={
            "event": event,
            "service": service_name,
            "environment": environment,
            "status": "error",
            "reason": reason,
            "response_reason": response_reason,
            "retryable": retryable,
            "error_message": public_message_for_reason(response_reason),
            "failed_stage": failed_stage,
            "request_id": request_id,
            "endpoint": endpoint,
            "source_service": gateway_request.source_service if gateway_request else None,
            "model": gateway_request.model if gateway_request else None,
            "chat_id": gateway_request.chat_id if gateway_request else None,
            "telegram_message_id": gateway_request.telegram_message_id if gateway_request else None,
            "error_type": type(error).__name__,
        },
    )


async def _resolve_readiness(readiness_check: ReadinessCheck | None) -> dict[str, Any]:
    if readiness_check is None:
        return {"ok": True, "checks": {}}
    result = readiness_check()
    if hasattr(result, "__await__"):
        result = await result
    return result if isinstance(result, dict) else {"ok": False, "checks": {}}


def _is_authorized(*, authorization: str | None, auth_token: str) -> bool:
    return bool(auth_token) and authorization == f"Bearer {auth_token}"


async def _safe_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def _request_id_from_body(body: dict[str, Any]) -> str | None:
    request_id = body.get("request_id")
    return str(request_id) if request_id is not None else None


def _parse_gateway_request(body: dict[str, Any]) -> GatewayChatRequest:
    if hasattr(GatewayChatRequest, "model_validate"):
        return GatewayChatRequest.model_validate(body)
    return GatewayChatRequest(**body)


def _parse_tts_request(body: dict[str, Any]) -> GatewayTTSRequest:
    if hasattr(GatewayTTSRequest, "model_validate"):
        return GatewayTTSRequest.model_validate(body)
    return GatewayTTSRequest(**body)


def _parse_embedding_request(body: dict[str, Any]) -> GatewayEmbeddingRequest:
    if hasattr(GatewayEmbeddingRequest, "model_validate"):
        return GatewayEmbeddingRequest.model_validate(body)
    return GatewayEmbeddingRequest(**body)


def _error_response(
    *,
    request_id: str | None,
    status_code: int,
    reason: str,
    retryable: bool,
    public_message: str | None = None,
    provider_reason: str | None = None,
    provider_status_code: int | None = None,
    retry_after_seconds: int | None = None,
    route_context: dict[str, str | None] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> JSONResponse:
    response_headers: dict[str, str] = {}
    content = {
        "request_id": request_id,
        "error": public_message or public_message_for_reason(reason),
        "reason": reason,
        "retryable": retryable,
    }
    if provider_reason:
        content["provider_reason"] = provider_reason
    if provider_status_code is not None:
        content["provider_status_code"] = provider_status_code
    if retry_after_seconds is not None and retry_after_seconds > 0:
        content["retry_after_seconds"] = retry_after_seconds
        response_headers["Retry-After"] = str(retry_after_seconds)
    if diagnostics:
        content.update({key: value for key, value in diagnostics.items() if value is not None})
    if route_context:
        content.update({key: value for key, value in route_context.items() if value is not None})
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=response_headers,
    )


def _gateway_error_route_context(error: GatewayError) -> dict[str, str | None]:
    return {
        "route_label": _optional_string(getattr(error, "route_label", None)),
        "project_label": _optional_string(getattr(error, "project_label", None)),
        "key_label": _optional_string(getattr(error, "key_label", None)),
        "proxy_label": _optional_string(getattr(error, "proxy_label", None)),
        "transport_mode": _optional_string(getattr(error, "transport_mode", None)),
    }


def _gateway_error_diagnostics(error: GatewayError) -> dict[str, Any]:
    return {
        "provider_called": False if getattr(error, "provider_called", None) is False else None,
        "error_code": _optional_string(getattr(error, "error_code", None)),
        "quota_scope": _optional_string(getattr(error, "quota_scope", None)),
        "quota_reset_at": _serialize_error_time(getattr(error, "quota_reset_at", None)),
        "eligible_routes_count": _optional_non_negative_int(getattr(error, "eligible_routes_count", None)),
        "exhausted_routes_count": _optional_non_negative_int(getattr(error, "exhausted_routes_count", None)),
        "disabled_routes_count": _optional_non_negative_int(getattr(error, "disabled_routes_count", None)),
        "cooldown_scope": _optional_string(getattr(error, "cooldown_scope", None)),
        "cooldown_level": _optional_non_negative_int(getattr(error, "cooldown_level", None)),
        "sleep_until": _serialize_error_time(getattr(error, "sleep_until", None)),
    }


def _gateway_error_retry_after_seconds(error: GatewayError) -> int | None:
    retry_after_seconds = getattr(error, "retry_after_seconds", None)
    if isinstance(retry_after_seconds, int) and retry_after_seconds > 0:
        return retry_after_seconds

    sleep_until = getattr(error, "sleep_until", None)
    if not isinstance(sleep_until, datetime):
        return None
    if sleep_until.tzinfo is None:
        sleep_until = sleep_until.replace(tzinfo=UTC)
    seconds = ceil((sleep_until - datetime.now(tz=UTC)).total_seconds())
    return max(1, seconds) if seconds > 0 else None


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return value
    return {key: item for key, item in vars(value).items() if item is not None}


def _optional_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _serialize_error_time(value: Any) -> str | None:
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return aware.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
