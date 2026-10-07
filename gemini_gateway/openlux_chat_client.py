from __future__ import annotations

import asyncio
import json
import logging
import math
from copy import deepcopy
from time import monotonic, perf_counter
from typing import Any

import httpx

from gemini_gateway.contracts import GatewayChatRequest, GatewayChatResponse
from gemini_gateway.errors import GatewayError
from gemini_gateway.gemini_client import (
    _is_content_filter_finish_reason,
    _merge_extra_headers,
    _provider_payload,
    _raise_for_embedded_provider_error,
    _timeout_error_kind,
)
from gemini_gateway.gemini_schema import adapt_gemini_json_schema
from gemini_gateway.provider_http_errors import (
    build_gateway_error_from_response,
    extract_provider_message,
    parse_retry_after,
)
from gemini_gateway.provider_observability import classify_chat_payload
from gemini_gateway.openlux_pricing import (
    OpenLuxPricingSnapshot,
    calculate_openlux_cost,
)

_DEFAULT_COOLDOWN_SECONDS = 30
_PROVIDER_REQUEST_ID_HEADERS = ("x-api-request-id", "x-request-id", "request-id")
_LOGGER = logging.getLogger(__name__)


class OpenLuxChatClient:
    """OpenAI-compatible streaming client для прямого OpenLux fallback."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.openlux.ai/v1",
        timeout: float = 60.0,
        max_stream_bytes: int = 262_144,
        cooldown_seconds: int = _DEFAULT_COOLDOWN_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
        pricing_catalog: Any | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_stream_bytes = max(1, int(max_stream_bytes))
        self._cooldown_seconds = max(1, int(cooldown_seconds))
        self._transport = transport
        self._pricing_catalog = pricing_catalog
        self._cooldown_until = 0.0
        self._cooldown_lock = asyncio.Lock()

    async def complete(
        self,
        *,
        request: GatewayChatRequest,
        api_key: str,
        model: str,
    ) -> GatewayChatResponse:
        await self._raise_if_cooling_down(request.request_id)
        pricing_task = (
            asyncio.create_task(self._pricing_snapshot(model))
            if self._pricing_catalog is not None
            else None
        )
        payload = _openlux_payload(request=request, model=model)
        headers = _merge_extra_headers(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            request.extra_headers,
        )
        started_at = perf_counter()
        request_bytes = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        payload_summary = classify_chat_payload(payload)

        client_kwargs: dict[str, Any] = {"timeout": self._timeout, "trust_env": False}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                headers_started_at = perf_counter()
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=request.timeout_seconds,
                ) as response:
                    response_headers_ms = _elapsed_ms(headers_started_at)
                    if response.status_code >= 400:
                        await response.aread()
                        error = _openlux_http_error(response=response, request_id=request.request_id)
                        await self._open_cooldown_for_error(error)
                        raise error

                    body_started_at = perf_counter()
                    raw_response, response_bytes = await _consume_sse_response(
                        response=response,
                        request_id=request.request_id,
                        max_stream_bytes=self._max_stream_bytes,
                    )
                    body_ms = _elapsed_ms(body_started_at)
                    provider_request_id = _provider_request_id(response.headers)
        except GatewayError:
            raise
        except httpx.TimeoutException as exc:
            error = GatewayError(
                reason="network_timeout",
                retryable=True,
                provider_message_safe=_timeout_error_kind(exc),
                request_id=request.request_id,
                provider_called=True,
            )
            await self._open_cooldown_for_error(error)
            raise error from exc
        except httpx.HTTPError as exc:
            error = GatewayError(
                reason="provider_unavailable",
                retryable=True,
                provider_message_safe="transport_error",
                request_id=request.request_id,
                provider_called=True,
            )
            await self._open_cooldown_for_error(error)
            raise error from exc

        provider_timing = {
            "operation_type": payload_summary.operation_type,
            "payload_kind": payload_summary.payload_kind,
            "request_bytes": request_bytes,
            "response_bytes": response_bytes,
            "media_count": payload_summary.media_count,
            "image_count": payload_summary.image_count,
            "provider_total_ms": _elapsed_ms(started_at),
            "request_prepare_ms": 0,
            "response_headers_ms": response_headers_ms,
            "response_body_ms": body_ms,
            "response_parse_ms": 0,
            "timeout_kind": None,
            "timeout_stage": None,
        }
        pricing_snapshot = await pricing_task if pricing_task is not None else None
        return _to_gateway_response(
            request=request,
            raw_response=raw_response,
            provider_request_id=provider_request_id,
            provider_timing=provider_timing,
            pricing_snapshot=pricing_snapshot,
        )

    async def _pricing_snapshot(self, model: str) -> OpenLuxPricingSnapshot | None:
        if self._pricing_catalog is None:
            return None
        try:
            return await self._pricing_catalog.get_snapshot(model)
        except Exception as error:
            _LOGGER.warning(
                "openlux_pricing_lookup_failed",
                extra={"error_type": type(error).__name__, "model": model},
            )
            return None

    async def _raise_if_cooling_down(self, request_id: str) -> None:
        async with self._cooldown_lock:
            remaining = self._cooldown_until - monotonic()
        if remaining <= 0:
            return
        raise GatewayError(
            reason="cooldown_active",
            retryable=True,
            retry_after_seconds=max(1, math.ceil(remaining)),
            provider_message_safe="openlux_circuit_open",
            request_id=request_id,
            provider_called=False,
        )

    async def _open_cooldown_for_error(self, error: GatewayError) -> None:
        if not error.retryable or error.reason == "invalid_response":
            return
        cooldown_seconds = max(error.retry_after_seconds or 0, self._cooldown_seconds)
        async with self._cooldown_lock:
            self._cooldown_until = max(self._cooldown_until, monotonic() + cooldown_seconds)


def _openlux_payload(*, request: GatewayChatRequest, model: str) -> dict[str, Any]:
    payload = _provider_payload(request)
    response_format = payload.get("response_format")
    if isinstance(response_format, dict):
        payload["response_format"] = adapt_gemini_json_schema(response_format)
    payload["model"] = model
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    return payload


def _openlux_http_error(*, response: httpx.Response, request_id: str) -> GatewayError:
    provider_message = extract_provider_message(response)
    normalized_message = (provider_message or "").lower()
    if response.status_code == 400:
        return GatewayError(
            reason="invalid_response",
            retryable=True,
            provider_status_code=response.status_code,
            provider_message_safe=provider_message,
            request_id=request_id,
            provider_called=True,
        )
    if response.status_code == 402 or (
        response.status_code == 403
        and any(marker in normalized_message for marker in ("quota", "balance", "insufficient"))
    ):
        return GatewayError(
            reason="quota_exhausted",
            retryable=True,
            provider_status_code=response.status_code,
            provider_message_safe=provider_message,
            retry_after_seconds=parse_retry_after(response.headers.get("Retry-After")),
            request_id=request_id,
            provider_called=True,
        )
    return build_gateway_error_from_response(
        response=response,
        request_id=request_id,
        supports_content_filter=True,
    )


async def _consume_sse_response(
    *,
    response: httpx.Response,
    request_id: str,
    max_stream_bytes: int,
) -> tuple[dict[str, Any], int]:
    accumulator = _OpenAIStreamAccumulator(request_id=request_id)
    response_bytes = 0
    async for line in response.aiter_lines():
        response_bytes += len(line.encode("utf-8")) + 1
        if response_bytes > max_stream_bytes:
            raise GatewayError(
                reason="invalid_response",
                retryable=False,
                provider_message_safe="OpenLux stream exceeded the configured output limit",
                request_id=request_id,
                provider_called=True,
            )
        data = _sse_data(line)
        if data is None:
            continue
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GatewayError(
                reason="invalid_response",
                retryable=False,
                provider_message_safe="OpenLux stream contained invalid JSON",
                request_id=request_id,
                provider_called=True,
            ) from exc
        if not isinstance(event, dict):
            raise GatewayError(
                reason="invalid_response",
                retryable=False,
                provider_message_safe="OpenLux stream event must be a JSON object",
                request_id=request_id,
                provider_called=True,
            )
        _raise_for_embedded_provider_error(raw_response=event, request_id=request_id)
        accumulator.add(event)
    return accumulator.build(), response_bytes


def _sse_data(line: str) -> str | None:
    normalized = line.strip()
    if not normalized or normalized.startswith(":"):
        return None
    if not normalized.startswith("data:"):
        return None
    return normalized.removeprefix("data:").strip()


class _OpenAIStreamAccumulator:
    def __init__(self, *, request_id: str) -> None:
        self._request_id = request_id
        self._id: str | None = None
        self._model: str | None = None
        self._object: str | None = None
        self._created: Any = None
        self._usage: dict[str, Any] = {}
        self._choices: dict[int, dict[str, Any]] = {}

    def add(self, event: dict[str, Any]) -> None:
        self._id = _first_string(self._id, event.get("id"))
        self._model = _first_string(self._model, event.get("model"))
        self._object = _first_string(self._object, event.get("object"))
        if self._created is None and event.get("created") is not None:
            self._created = event["created"]
        if isinstance(event.get("usage"), dict):
            self._usage = deepcopy(event["usage"])

        choices = event.get("choices")
        if not isinstance(choices, list):
            return
        for raw_choice in choices:
            if not isinstance(raw_choice, dict):
                continue
            index = _choice_index(raw_choice.get("index"))
            choice = self._choices.setdefault(
                index,
                {
                    "index": index,
                    "finish_reason": None,
                    "role": "assistant",
                    "content_parts": [],
                    "tool_calls": {},
                },
            )
            finish_reason = raw_choice.get("finish_reason")
            if isinstance(finish_reason, str):
                choice["finish_reason"] = finish_reason
            delta = raw_choice.get("delta")
            if not isinstance(delta, dict):
                continue
            if isinstance(delta.get("role"), str):
                choice["role"] = delta["role"]
            if isinstance(delta.get("content"), str):
                choice["content_parts"].append(delta["content"])
            self._add_tool_calls(choice, delta.get("tool_calls"))

    def _add_tool_calls(self, choice: dict[str, Any], raw_tool_calls: Any) -> None:
        if not isinstance(raw_tool_calls, list):
            return
        tool_calls: dict[int, dict[str, str]] = choice["tool_calls"]
        for implicit_index, raw_tool_call in enumerate(raw_tool_calls):
            if not isinstance(raw_tool_call, dict):
                continue
            raw_id = raw_tool_call.get("id") if isinstance(raw_tool_call.get("id"), str) else None
            index = _tool_call_index(
                raw_tool_call.get("index"),
                implicit_index=implicit_index,
                raw_id=raw_id,
                tool_calls=tool_calls,
            )
            tool_call = tool_calls.setdefault(
                index,
                {"id": "", "type": "function", "name": "", "arguments": ""},
            )
            if raw_id is not None and raw_id != tool_call["id"]:
                tool_call["id"] += raw_id
            if isinstance(raw_tool_call.get("type"), str):
                tool_call["type"] = raw_tool_call["type"]
            function = raw_tool_call.get("function")
            if not isinstance(function, dict):
                continue
            if isinstance(function.get("name"), str):
                tool_call["name"] += function["name"]
            if isinstance(function.get("arguments"), str):
                tool_call["arguments"] += function["arguments"]

    def build(self) -> dict[str, Any]:
        if not self._choices:
            raise GatewayError(
                reason="invalid_response",
                retryable=False,
                provider_message_safe="OpenLux stream did not contain choices",
                request_id=self._request_id,
                provider_called=True,
            )
        choices = [self._build_choice(self._choices[index]) for index in sorted(self._choices)]
        return {
            "id": self._id,
            "object": self._object or "chat.completion",
            "created": self._created,
            "model": self._model,
            "choices": choices,
            "usage": self._usage,
        }

    @staticmethod
    def _build_choice(choice: dict[str, Any]) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": choice["role"],
            "content": "".join(choice["content_parts"]) or None,
        }
        tool_calls = choice["tool_calls"]
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_calls[index]["id"],
                    "type": tool_calls[index]["type"],
                    "function": {
                        "name": tool_calls[index]["name"],
                        "arguments": tool_calls[index]["arguments"],
                    },
                }
                for index in sorted(tool_calls)
            ]
        finish_reason = "tool_calls" if tool_calls else choice["finish_reason"]
        return {
            "index": choice["index"],
            "finish_reason": finish_reason,
            "message": message,
        }


def _to_gateway_response(
    *,
    request: GatewayChatRequest,
    raw_response: dict[str, Any],
    provider_request_id: str | None,
    provider_timing: dict[str, Any],
    pricing_snapshot: OpenLuxPricingSnapshot | None,
) -> GatewayChatResponse:
    choices = raw_response["choices"]
    finish_reason = choices[0].get("finish_reason")
    if _is_content_filter_finish_reason(finish_reason):
        raise GatewayError(
            reason="content_filtered",
            retryable=False,
            provider_message_safe=str(finish_reason),
            request_id=request.request_id,
            provider_called=True,
        )
    usage = deepcopy(raw_response.get("usage") or {})
    pricing_snapshot_json = None
    if pricing_snapshot is not None:
        prompt_tokens = _non_negative_token_count(usage.get("prompt_tokens"))
        completion_tokens = _non_negative_token_count(usage.get("completion_tokens"))
        cost = calculate_openlux_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            snapshot=pricing_snapshot,
        )
        cost_value = float(cost)
        usage["cost"] = cost_value
        cost_details = usage.get("cost_details") if isinstance(usage.get("cost_details"), dict) else {}
        usage["cost_details"] = {**cost_details, "upstream_inference_cost": cost_value}
        pricing_snapshot_json = pricing_snapshot.as_json()
    return GatewayChatResponse(
        request_id=request.request_id,
        generation_id=raw_response.get("id"),
        provider_request_id=provider_request_id,
        model=raw_response.get("model") or request.model,
        choices=choices,
        usage=usage,
        pricing_snapshot_json=pricing_snapshot_json,
        finish_reason=finish_reason,
        raw_response=raw_response,
        provider_specific_fields={},
        provider_timing=provider_timing,
    )


def _provider_request_id(headers: httpx.Headers) -> str | None:
    for header_name in _PROVIDER_REQUEST_ID_HEADERS:
        value = headers.get(header_name)
        if value and value.strip():
            return value.strip()[:160]
    return None


def _choice_index(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _tool_call_index(
    value: Any,
    *,
    implicit_index: int,
    raw_id: str | None,
    tool_calls: dict[int, dict[str, str]],
) -> int:
    declared_index = implicit_index
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        declared_index = value
    if raw_id:
        for index, tool_call in tool_calls.items():
            if tool_call["id"] == raw_id:
                return index
        declared_tool_call = tool_calls.get(declared_index)
        if declared_tool_call is not None and declared_tool_call["id"]:
            return max(tool_calls, default=-1) + 1
    return declared_index


def _first_string(current: str | None, candidate: Any) -> str | None:
    if current is not None:
        return current
    if isinstance(candidate, str) and candidate:
        return candidate
    return None


def _non_negative_token_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))
