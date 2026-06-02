from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

import httpx

from gemini_gateway.contracts import (
    GatewayChatRequest,
    GatewayChatResponse,
    GatewayErrorReason,
    sanitize_gateway_choices,
    sanitize_gateway_provider_specific_fields,
)
from gemini_gateway.errors import GatewayError, reason_from_gemini_status
from gemini_gateway.http_client_pool import GatewayHttpClientPool
from gemini_gateway.provider_http_errors import build_gateway_error_from_response

_SUPPORTED_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "none"}
_REASONING_EFFORT_ALIASES = {
    "max": "high",
    "maximum": "high",
    "xhigh": "high",
}
_PROTECTED_PROVIDER_PAYLOAD_FIELDS = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "tools",
    "tool_choice",
    "response_format",
    "reasoning",
    "reasoning_effort",
    "stream",
}
_PROTECTED_PROVIDER_HEADER_NAMES = {"authorization", "content-type"}
_EMBEDDED_PROVIDER_ERROR_FIELDS = ("error", "provider_error")
_GEMINI_STATUS_TO_HTTP_CODE = {
    "RESOURCE_EXHAUSTED": 429,
    "UNAUTHENTICATED": 401,
    "PERMISSION_DENIED": 403,
    "INVALID_ARGUMENT": 400,
    "DEADLINE_EXCEEDED": 504,
    "UNAVAILABLE": 503,
    "INTERNAL": 500,
}
_HTTP_STATUS_CODE_PATTERN = re.compile(r"\b([1-5]\d\d)\b")


class GeminiOpenAIClient:
    """HTTP-клиент Gemini OpenAI-compatible Chat Completions."""

    _STANDARD_RESPONSE_FIELDS = {"id", "object", "created", "model", "choices", "usage"}

    def __init__(
        self,
        *,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        client_pool: GatewayHttpClientPool | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport
        self._client_pool = client_pool

    async def complete(
        self,
        *,
        request: GatewayChatRequest,
        api_key: str,
        proxy_url: str | None,
    ) -> GatewayChatResponse:
        _require_proxy_url(proxy_url=proxy_url, request_id=request.request_id)
        payload = _provider_payload(request)
        headers = _merge_extra_headers(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            request.extra_headers,
        )
        try:
            if self._client_pool is not None:
                response = await self._client_pool.get(proxy_url=proxy_url).post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=request.timeout_seconds,
                )
            else:
                client_kwargs: dict[str, Any] = {
                    "timeout": self._timeout,
                    "trust_env": False,
                }
                if self._transport is not None:
                    client_kwargs["transport"] = self._transport
                elif proxy_url is not None:
                    client_kwargs["proxy"] = proxy_url
                async with httpx.AsyncClient(**client_kwargs) as client:
                    response = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=request.timeout_seconds,
                    )
        except httpx.ProxyError as exc:
            raise GatewayError(
                reason="proxy_failed",
                retryable=True,
                provider_message_safe=str(exc),
                request_id=request.request_id,
            ) from exc
        except httpx.TimeoutException as exc:
            raise GatewayError(
                reason="network_timeout",
                retryable=True,
                provider_message_safe=_timeout_error_kind(exc),
                request_id=request.request_id,
            ) from exc
        except httpx.HTTPError as exc:
            raise GatewayError(
                reason=_transport_error_reason(proxy_url),
                retryable=True,
                provider_message_safe=str(exc),
                request_id=request.request_id,
            ) from exc

        if response.status_code >= 400:
            raise build_gateway_error_from_response(
                response=response,
                request_id=request.request_id,
                supports_content_filter=True,
            )

        try:
            raw_response = response.json()
        except ValueError as exc:
            raise GatewayError(
                reason="invalid_response",
                retryable=False,
                provider_message_safe=str(exc),
                request_id=request.request_id,
                provider_called=True,
            ) from exc
        if not isinstance(raw_response, dict):
            raise GatewayError(
                reason="invalid_response",
                retryable=False,
                provider_message_safe="Gemini response payload must be a JSON object",
                request_id=request.request_id,
                provider_called=True,
            )
        return self._to_gateway_response(request=request, raw_response=raw_response)

    def _to_gateway_response(self, *, request: GatewayChatRequest, raw_response: dict[str, Any]) -> GatewayChatResponse:
        _raise_for_embedded_provider_error(raw_response=raw_response, request_id=request.request_id)
        raw_choices = raw_response.get("choices")
        if not isinstance(raw_choices, list) or not raw_choices or not isinstance(raw_choices[0], dict):
            raise GatewayError(
                reason="invalid_response",
                retryable=False,
                provider_message_safe="Gemini response does not contain valid choices",
                request_id=request.request_id,
                provider_called=True,
            )
        usage = raw_response.get("usage") or {}
        choices = sanitize_gateway_choices(raw_choices)
        provider_specific_fields = _provider_specific_fields(raw_response, self._STANDARD_RESPONSE_FIELDS)
        finish_reason = choices[0].get("finish_reason")
        if _is_content_filter_finish_reason(finish_reason):
            raise GatewayError(
                reason="content_filtered",
                retryable=False,
                provider_message_safe=str(finish_reason),
                request_id=request.request_id,
                provider_called=True,
            )

        return GatewayChatResponse(
            request_id=request.request_id,
            generation_id=raw_response.get("id"),
            model=raw_response.get("model") or request.model,
            choices=choices,
            usage=usage,
            finish_reason=finish_reason,
            raw_response=raw_response,
            provider_specific_fields=provider_specific_fields,
        )


def _provider_payload(request: GatewayChatRequest) -> dict[str, Any]:
    request_data = _dump(request)
    payload: dict[str, Any] = {}
    for field_name in (
        "model",
        "temperature",
        "max_tokens",
        "tools",
        "tool_choice",
        "response_format",
    ):
        value = request_data.get(field_name)
        if value is not None:
            payload[field_name] = value
    payload["messages"] = _normalize_messages_for_openai_compat(request_data.get("messages") or [])
    payload["model"] = _to_gemini_model_name(str(payload["model"]))
    reasoning_effort = _reasoning_effort(request_data.get("reasoning"))
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    _apply_extra_body(payload, request_data.get("extra_body"))
    metadata = request_data.get("metadata")
    if isinstance(metadata, dict) and metadata:
        existing_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        payload["metadata"] = {**existing_metadata, **deepcopy(metadata)}
    return payload


def _is_content_filter_finish_reason(value: Any) -> bool:
    finish_reason = str(value or "").strip().lower()
    if finish_reason in {"content_filter", "content_filtered", "safety"}:
        return True
    return finish_reason.startswith(("content_filter:", "content_filtered:", "safety:"))


def _apply_extra_body(payload: dict[str, Any], extra_body: Any) -> None:
    if not isinstance(extra_body, dict):
        return
    for key, value in extra_body.items():
        if key in _PROTECTED_PROVIDER_PAYLOAD_FIELDS:
            continue
        payload[key] = deepcopy(value)


def _merge_extra_headers(base_headers: dict[str, str], extra_headers: Any) -> dict[str, str]:
    headers = dict(base_headers)
    if not isinstance(extra_headers, dict):
        return headers
    for key, value in extra_headers.items():
        header_name = str(key).strip()
        if not header_name or header_name.lower() in _PROTECTED_PROVIDER_HEADER_NAMES:
            continue
        headers[header_name] = str(value)
    return headers


def _transport_error_reason(proxy_url: str | None) -> GatewayErrorReason:
    if proxy_url:
        return "proxy_failed"
    return "provider_unavailable"


def _timeout_error_kind(exc: httpx.TimeoutException) -> str:
    """Возвращает стабильный тип timeout без сырого текста transport exception."""

    if isinstance(exc, httpx.ConnectTimeout):
        return "connect_timeout"
    if isinstance(exc, httpx.ReadTimeout):
        return "read_timeout"
    if isinstance(exc, httpx.WriteTimeout):
        return "write_timeout"
    if isinstance(exc, httpx.PoolTimeout):
        return "pool_timeout"
    return "timeout"


def _require_proxy_url(*, proxy_url: str | None, request_id: str) -> None:
    """Запрещает direct-вызовы Gemini: каждый ключ обязан идти через свой proxy."""

    if proxy_url and proxy_url.strip():
        return
    raise GatewayError(
        reason="no_route",
        retryable=True,
        provider_message_safe="Gemini proxy URL is required by proxy-only policy",
        request_id=request_id,
        provider_called=False,
    )


def _to_gemini_model_name(model: str) -> str:
    if model.startswith("google/"):
        return model.removeprefix("google/")
    if model.startswith("gemini/"):
        return model.removeprefix("gemini/")
    return model


def _normalize_messages_for_openai_compat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    id_map = _tool_call_id_map(messages)
    tool_name_map = _tool_call_name_map(messages, id_map)
    normalized: list[dict[str, Any]] = []
    pending_image_followups: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if _is_tool_result_message(message):
            clean_tool_message, image_followup = _normalize_tool_result_message(
                message=message,
                id_map=id_map,
                tool_name_map=tool_name_map,
            )
            normalized.append(clean_tool_message)
            if image_followup is not None:
                pending_image_followups.append(image_followup)
            if index + 1 >= len(messages) or not _is_tool_result_message(messages[index + 1]):
                normalized.extend(pending_image_followups)
                pending_image_followups = []
            continue

        if pending_image_followups:
            normalized.extend(pending_image_followups)
            pending_image_followups = []

        clean_message: dict[str, Any] = {}
        for key, value in message.items():
            if key in {"reasoning_details", "provider_specific_fields"}:
                continue
            if key == "tool_calls" and isinstance(value, list):
                clean_message[key] = [_normalize_tool_call(call, id_map) for call in value if isinstance(call, dict)]
                continue
            if key == "tool_call_id":
                clean_message[key] = _mapped_tool_call_id(str(value), id_map)
                continue
            clean_message[key] = value
        normalized.append(clean_message)
    normalized.extend(pending_image_followups)
    return normalized


def _raise_for_embedded_provider_error(*, raw_response: dict[str, Any], request_id: str) -> None:
    for error_payload in _embedded_provider_error_payloads(raw_response):
        provider_message = _provider_error_message(error_payload)
        provider_status_code = _provider_error_status_code(error_payload, provider_message=provider_message)
        if provider_message is None and provider_status_code is None:
            continue
        raise GatewayError(
            reason=_embedded_error_reason(status_code=provider_status_code, provider_message=provider_message),
            retryable=_embedded_error_retryable(provider_status_code),
            provider_status_code=provider_status_code,
            provider_message_safe=provider_message,
            request_id=request_id,
            status_code=provider_status_code if provider_status_code == 429 else None,
            provider_called=True,
        )


def _embedded_provider_error_payloads(raw_response: dict[str, Any]) -> list[Any]:
    payloads = _provider_error_fields(raw_response)
    provider_fields = raw_response.get("provider_specific_fields")
    if isinstance(provider_fields, dict):
        payloads.extend(_provider_error_fields(provider_fields))

    choices = raw_response.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            payloads.extend(_provider_error_fields(choice))
            choice_provider_fields = choice.get("provider_specific_fields")
            if isinstance(choice_provider_fields, dict):
                payloads.extend(_provider_error_fields(choice_provider_fields))
    return [payload for payload in payloads if _has_embedded_provider_error(payload)]


def _provider_error_fields(payload: dict[str, Any]) -> list[Any]:
    return [payload.get(field_name) for field_name in _EMBEDDED_PROVIDER_ERROR_FIELDS]


def _has_embedded_provider_error(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value)
    return False


def _provider_error_message(error_payload: Any) -> str | None:
    if isinstance(error_payload, str):
        return error_payload
    if not isinstance(error_payload, dict):
        return None
    for field_name in ("message", "status", "reason", "code"):
        value = error_payload.get(field_name)
        if value is not None:
            return str(value)
    return None


def _provider_error_status_code(error_payload: Any, *, provider_message: str | None) -> int | None:
    if isinstance(error_payload, dict):
        for field_name in ("code", "status_code"):
            status_code = _int_status_code(error_payload.get(field_name))
            if status_code is not None:
                return status_code
        status_value = error_payload.get("status") or error_payload.get("reason")
        status_code = _status_code_from_gemini_status(status_value)
        if status_code is not None:
            return status_code
    return _status_code_from_text(provider_message)


def _embedded_error_reason(*, status_code: int | None, provider_message: str | None) -> GatewayErrorReason:
    if status_code is not None:
        return reason_from_gemini_status(
            status_code=status_code,
            provider_message=provider_message,
            supports_content_filter=True,
        )
    message = (provider_message or "").lower()
    if "safety" in message or "content_filter" in message or "content filtered" in message:
        return "content_filtered"
    if "api key" in message or "api_key" in message or "unauthorized" in message:
        return "auth_failed"
    return "request_failed"


def _embedded_error_retryable(status_code: int | None) -> bool:
    if status_code is None:
        return True
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def _int_status_code(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if 100 <= value <= 599 else None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            parsed = int(text)
            return parsed if 100 <= parsed <= 599 else None
    return None


def _status_code_from_gemini_status(value: Any) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if not normalized:
        return None
    return _GEMINI_STATUS_TO_HTTP_CODE.get(normalized)


def _status_code_from_text(value: str | None) -> int | None:
    if not value:
        return None
    match = _HTTP_STATUS_CODE_PATTERN.search(value)
    if match is None:
        return None
    return int(match.group(1))


def _tool_call_id_map(messages: list[dict[str, Any]]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    used: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict) or call.get("id") is None:
                continue
            raw_id = str(call["id"])
            if raw_id in mapped:
                continue
            normalized_id = _dedupe_tool_call_id(_base_tool_call_id(raw_id), raw_id, used)
            mapped[raw_id] = normalized_id
            used.add(normalized_id)
    return mapped


def _tool_call_name_map(messages: list[dict[str, Any]], id_map: dict[str, str]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict) or call.get("id") is None:
                continue
            function = call.get("function")
            if not isinstance(function, dict) or function.get("name") is None:
                continue
            mapped_id = _mapped_tool_call_id(str(call["id"]), id_map)
            mapped[mapped_id] = str(function["name"])
    return mapped


def _is_tool_result_message(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") == "tool"


def _normalize_tool_result_message(
    *,
    message: dict[str, Any],
    id_map: dict[str, str],
    tool_name_map: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    raw_tool_call_id = str(message.get("tool_call_id") or "")
    tool_call_id = _mapped_tool_call_id(raw_tool_call_id, id_map)
    tool_name = _tool_result_name(message=message, tool_call_id=tool_call_id, tool_name_map=tool_name_map)
    clean_message: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": _tool_result_text_content(message.get("content")),
    }
    if tool_name is not None:
        clean_message["name"] = tool_name
    return clean_message, _tool_image_followup_message(
        content=message.get("content"),
        tool_name=tool_name,
        tool_call_id=tool_call_id,
    )


def _tool_result_name(
    *,
    message: dict[str, Any],
    tool_call_id: str,
    tool_name_map: dict[str, str],
) -> str | None:
    name = message.get("name")
    if name is not None:
        return str(name)
    return tool_name_map.get(tool_call_id)


def _tool_result_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            part["text"]
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        ]
        if texts:
            return "\n".join(texts)
        image_count = len(_tool_image_parts(content))
        if image_count:
            return json.dumps({"status": "success", "images_count": image_count}, ensure_ascii=False)
    if content is None:
        return "{}"
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except TypeError:
        return str(content)


def _tool_image_followup_message(
    *,
    content: Any,
    tool_name: str | None,
    tool_call_id: str,
) -> dict[str, Any] | None:
    if not isinstance(content, list):
        return None
    image_parts = _tool_image_parts(content)
    if not image_parts:
        return None
    content_parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"Images returned by tool {tool_name or 'unknown'} for tool_call_id {tool_call_id}.",
        },
        *image_parts,
    ]
    return {"role": "user", "content": content_parts}


def _tool_image_parts(content: list[Any]) -> list[dict[str, Any]]:
    image_parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        image_url = part.get("image_url")
        if not isinstance(image_url, dict) or image_url.get("url") is None:
            continue
        image_parts.append({"type": "image_url", "image_url": {"url": str(image_url["url"])}})
    return image_parts


def _normalize_tool_call(call: dict[str, Any], id_map: dict[str, str]) -> dict[str, Any]:
    clean_call: dict[str, Any] = {
        "id": _mapped_tool_call_id(str(call.get("id") or ""), id_map),
        "type": str(call.get("type") or "function"),
    }
    function = call.get("function")
    if isinstance(function, dict):
        clean_function: dict[str, Any] = {}
        if function.get("name") is not None:
            clean_function["name"] = str(function["name"])
        arguments = function.get("arguments")
        clean_function["arguments"] = arguments if isinstance(arguments, str) and arguments else "{}"
        clean_call["function"] = clean_function
    extra_content = _google_extra_content(call)
    if extra_content is not None:
        clean_call["extra_content"] = extra_content
    return clean_call


def _reasoning_effort(reasoning: Any) -> str | None:
    if not isinstance(reasoning, dict):
        return None
    effort = reasoning.get("effort") or reasoning.get("reasoning_effort")
    if effort is None:
        return None
    normalized = str(effort).strip().lower()
    normalized = _REASONING_EFFORT_ALIASES.get(normalized, normalized)
    if normalized in _SUPPORTED_REASONING_EFFORTS:
        return normalized
    return None


def _google_extra_content(call: dict[str, Any]) -> dict[str, Any] | None:
    thought_signature = _google_thought_signature(call)
    if thought_signature is None:
        return None
    return {"google": {"thought_signature": thought_signature}}


def _google_thought_signature(call: dict[str, Any]) -> str | None:
    extra_content = call.get("extra_content")
    if isinstance(extra_content, dict):
        google = extra_content.get("google")
        if isinstance(google, dict) and google.get("thought_signature") is not None:
            return str(google["thought_signature"])

    provider_fields = call.get("provider_specific_fields")
    if not isinstance(provider_fields, dict):
        return None
    if provider_fields.get("thought_signature") is not None:
        return str(provider_fields["thought_signature"])
    google = provider_fields.get("google")
    if isinstance(google, dict) and google.get("thought_signature") is not None:
        return str(google["thought_signature"])
    return None


def _mapped_tool_call_id(raw_id: str, id_map: dict[str, str]) -> str:
    if raw_id in id_map:
        return id_map[raw_id]
    return _base_tool_call_id(raw_id)


def _base_tool_call_id(raw_id: str) -> str:
    base_id = raw_id.split("__thought__", 1)[0].strip()
    if not base_id:
        base_id = f"call_{_short_hash(raw_id)}"
    if len(base_id) > 120:
        return f"call_{_short_hash(base_id)}"
    return base_id


def _dedupe_tool_call_id(base_id: str, raw_id: str, used: set[str]) -> str:
    if base_id not in used:
        return base_id
    return f"{base_id[:95]}_{_short_hash(raw_id)}"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _provider_specific_fields(raw_response: dict[str, Any], standard_fields: set[str]) -> dict[str, Any]:
    fields = sanitize_gateway_provider_specific_fields(raw_response)
    existing = raw_response.get("provider_specific_fields")
    if isinstance(existing, dict):
        fields.update(sanitize_gateway_provider_specific_fields(existing))
    for key in standard_fields:
        fields.pop(key, None)
    return fields


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return vars(value)
