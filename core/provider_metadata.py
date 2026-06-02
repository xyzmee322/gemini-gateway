from __future__ import annotations

from copy import deepcopy
from typing import Any

_SAFE_PROVIDER_SPECIFIC_FIELD_NAMES = (
    "thought_signature",
    "thoughtSignature",
    "reasoning_details",
    "reasoningDetails",
)
_SAFE_GOOGLE_PROVIDER_FIELD_NAMES = ("thought_signature", "thoughtSignature")


def allowlist_provider_specific_fields(payload: Any) -> dict[str, Any]:
    """Оставляет только provider metadata, нужную для continuity."""

    if not isinstance(payload, dict):
        return {}

    safe_fields: dict[str, Any] = {}
    for field_name in _SAFE_PROVIDER_SPECIFIC_FIELD_NAMES:
        if payload.get(field_name) is not None:
            safe_fields[field_name] = deepcopy(payload[field_name])

    google = payload.get("google")
    if isinstance(google, dict):
        safe_google = _allowlist_google_provider_fields(google)
        if safe_google:
            safe_fields["google"] = safe_google

    return safe_fields


def allowlist_extra_content(payload: Any) -> dict[str, Any]:
    """Оставляет безопасную часть extra_content для provider continuity."""

    if not isinstance(payload, dict):
        return {}
    google = payload.get("google")
    if not isinstance(google, dict):
        return {}
    safe_google = _allowlist_google_provider_fields(google)
    return {"google": safe_google} if safe_google else {}


def allowlist_provider_metadata_tree(value: Any) -> Any:
    """Удаляет произвольную provider metadata из вложенной структуры."""

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _normalize_key(key)
            if normalized_key == "provider_specific_fields":
                safe_fields = allowlist_provider_specific_fields(item)
                if safe_fields:
                    sanitized[str(key)] = safe_fields
                continue
            if normalized_key == "extra_content":
                safe_extra_content = allowlist_extra_content(item)
                if safe_extra_content:
                    sanitized[str(key)] = safe_extra_content
                continue
            sanitized[str(key)] = allowlist_provider_metadata_tree(item)
        return sanitized
    if isinstance(value, list):
        return [allowlist_provider_metadata_tree(item) for item in value]
    return deepcopy(value)


def _allowlist_google_provider_fields(payload: dict[str, Any]) -> dict[str, Any]:
    safe_fields: dict[str, Any] = {}
    for field_name in _SAFE_GOOGLE_PROVIDER_FIELD_NAMES:
        if payload.get(field_name) is not None:
            safe_fields[field_name] = deepcopy(payload[field_name])
    return safe_fields


def _normalize_key(value: Any) -> str:
    return str(value or "").lower().strip().replace("-", "_")
