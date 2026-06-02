from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from core.provider_metadata import allowlist_extra_content, allowlist_provider_specific_fields
from core.secret_redaction import redact_secrets_in_text
from core.string_parsing import parse_optional_string as _optional_string

_DATA_URL_PATTERN = re.compile(r"^data:(?P<kind>image|audio|video)/[^;]+;base64,(?P<data>.+)$", re.DOTALL)
_MEDIA_PLACEHOLDER_PATTERN = re.compile(r"^<(image|audio|video):\d+kb>$")
_THOUGHT_SIGNATURE_PLACEHOLDER_PATTERN = re.compile(r"^<thought_signature:sha256=[0-9a-fA-F]+:bytes=\d+>$")
_PRESIGNED_URL_MARKERS = (
    "x-amz-",
    "x-goog-",
    "awsaccesskeyid=",
    "signature=",
)
_SECRET_KEYS = {
    "access_key",
    "api_key",
    "authorization",
    "access_token",
    "refresh_token",
    "encryption_key",
    "hmac_key",
    "private_key",
    "token",
    "secret_key",
    "secret",
    "signing_key",
    "password",
}
_SECRET_KEY_SUFFIXES = (
    "_api_key",
    "_access_key",
    "_encryption_key",
    "_hmac_key",
    "_private_key",
    "_secret_key",
    "_signing_key",
    "_token",
    "_secret",
    "_password",
)
_SECRET_QUERY_KEYS = {
    "api_key",
    "apikey",
    "access_key",
    "accesskey",
    "access_token",
    "client_secret",
    "encryption_key",
    "hmac_key",
    "id_token",
    "private_key",
    "refresh_token",
    "secret_key",
    "session_token",
    "signing_key",
    "authorization",
    "auth",
    "key",
    "password",
    "secret",
    "signature",
    "sig",
    "token",
}
_MEDIA_KINDS = frozenset({"image", "audio", "video"})
_PROVIDER_REASONING_KEYS = frozenset(
    {
        "reasoning",
        "reasoning_content",
        "reasoningcontent",
        "reasoning_details",
        "reasoningdetails",
    }
)


def sanitize_payload_for_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_value(copy.deepcopy(dict(payload)), parent_key=None)
    if not isinstance(sanitized, dict):
        return {}
    return sanitized


def _sanitize_value(value: Any, *, parent_key: str | None) -> Any:
    if _is_secret_key(parent_key):
        return "***"

    if isinstance(value, Mapping):
        return _sanitize_mapping(value, parent_key=parent_key)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_value(item, parent_key=parent_key) for item in value]

    if isinstance(value, str):
        return _sanitize_string(value)

    if isinstance(value, bytes):
        return f"<bytes:{_format_kb(len(value))}>"

    return value


def _sanitize_mapping(value: Mapping[str, Any], *, parent_key: str | None) -> dict[str, Any]:
    normalized_parent_key = _normalize_key(parent_key)
    if normalized_parent_key == "provider_specific_fields":
        value = allowlist_provider_specific_fields(value)
    elif normalized_parent_key == "extra_content":
        value = allowlist_extra_content(value)

    media_container_kind = _media_kind_from_container(parent_key, value)
    sanitized: dict[str, Any] = {}

    for key, item in value.items():
        key_text = str(key)
        if _should_strip_provider_reasoning_key(parent_key=normalized_parent_key, key=key_text):
            continue

        if _is_thought_signature_key(key_text) and isinstance(item, str):
            sanitized[key_text] = _sanitize_thought_signature(item)
            continue

        media_field_kind = _media_kind_from_base64_field(key_text)
        if media_field_kind and isinstance(item, str):
            sanitized[key_text] = _sanitize_base64_media(media_field_kind, item)
            continue

        if media_container_kind and _normalize_key(key_text) == "data" and isinstance(item, str):
            sanitized[key_text] = _sanitize_base64_media(media_container_kind, item)
            continue

        sanitized[key_text] = _sanitize_value(item, parent_key=key_text)

    return sanitized


def _sanitize_string(value: str) -> str:
    clean_value = value.replace("\x00", "")
    match = _DATA_URL_PATTERN.match(clean_value)
    if match is None:
        if "__thought__" in clean_value:
            return _sanitize_embedded_thought_signature(clean_value)
        if _looks_like_presigned_url(clean_value):
            return "<presigned_url:redacted>"
        if _is_absolute_url(clean_value):
            return _redact_url_secrets(clean_value)
        return redact_secrets_in_text(clean_value)

    media_kind = match.group("kind")
    encoded_data = match.group("data")
    byte_size = _estimate_base64_size(encoded_data)
    return f"<{media_kind}:{_format_kb(byte_size)}>"


def _sanitize_base64_media(media_kind: str, value: str) -> str:
    clean_value = value.replace("\x00", "")
    if _MEDIA_PLACEHOLDER_PATTERN.match(clean_value):
        return clean_value

    data_url_match = _DATA_URL_PATTERN.match(clean_value)
    if data_url_match is not None:
        data_url_kind = data_url_match.group("kind")
        byte_size = _estimate_base64_size(data_url_match.group("data"))
        return f"<{data_url_kind}:{_format_kb(byte_size)}>"

    byte_size = _estimate_base64_size(clean_value)
    return f"<{media_kind}:{_format_kb(byte_size)}>"


def _sanitize_thought_signature(value: str) -> str:
    clean_value = value.replace("\x00", "")
    if _THOUGHT_SIGNATURE_PLACEHOLDER_PATTERN.match(clean_value):
        return clean_value
    digest = hashlib.sha256(clean_value.encode("utf-8")).hexdigest()
    return f"<thought_signature:sha256={digest}:bytes={len(clean_value.encode('utf-8'))}>"


def _sanitize_embedded_thought_signature(value: str) -> str:
    prefix, signature = value.split("__thought__", 1)
    if not signature:
        return value
    return f"{prefix}__thought__{_sanitize_thought_signature(signature)}"


def _media_kind_from_container(parent_key: str | None, value: Mapping[str, Any]) -> str | None:
    normalized_key = _normalize_key(parent_key)
    if normalized_key == "input_audio":
        return "audio"

    if normalized_key in {"inline_data", "inlinedata"}:
        return _media_kind_from_mime_type(
            _optional_string(value.get("mime_type")) or _optional_string(value.get("mimeType"))
        )

    return None


def _media_kind_from_base64_field(key: str) -> str | None:
    normalized_key = _normalize_key(key)
    for media_kind in _MEDIA_KINDS:
        if normalized_key in {
            f"{media_kind}_base64",
            f"{media_kind}_b64",
            f"{media_kind}_data",
        }:
            return media_kind
    return None


def _media_kind_from_mime_type(mime_type: str | None) -> str | None:
    if not mime_type:
        return None
    media_kind = mime_type.split("/", 1)[0].lower()
    return media_kind if media_kind in _MEDIA_KINDS else None


def _estimate_base64_size(encoded_data: str) -> int:
    normalized = "".join(encoded_data.split())
    if not normalized:
        return 0
    padding = len(normalized) - len(normalized.rstrip("="))
    return max(0, int((len(normalized) * 3) / 4) - padding)


def _format_kb(byte_size: int) -> str:
    return f"{max(1, round(byte_size / 1024))}kb"


def _is_secret_key(key: str | None) -> bool:
    if key is None:
        return False
    normalized = _normalize_key(key)
    return normalized in _SECRET_KEYS or normalized.endswith(_SECRET_KEY_SUFFIXES)


def _redact_url_secrets(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value

    netloc = parsed.netloc
    if parsed.username is not None or parsed.password is not None:
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = f"***@{hostname}"
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"

    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    redacted_query = urlencode(
        [
            (key, "***" if _normalize_key(key) in _SECRET_QUERY_KEYS else value)
            for key, value in query_items
        ]
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, redacted_query, parsed.fragment))


def _is_thought_signature_key(key: str) -> bool:
    return _normalize_key(key) in {"thought_signature", "thoughtsignature"}


def _is_provider_reasoning_key(key: str) -> bool:
    return _normalize_key(key) in _PROVIDER_REASONING_KEYS


def _should_strip_provider_reasoning_key(*, parent_key: str, key: str) -> bool:
    if parent_key in {"", "provider_specific_fields"}:
        return False
    return _is_provider_reasoning_key(key)


def _looks_like_presigned_url(value: str) -> bool:
    normalized = value.lower()
    if not normalized.startswith(("http://", "https://")) or "?" not in normalized:
        return False
    return any(marker in normalized for marker in _PRESIGNED_URL_MARKERS)


def _is_absolute_url(value: str) -> bool:
    parsed = urlsplit(value)
    return bool(parsed.scheme and parsed.netloc)


def _normalize_key(key: str | None) -> str:
    return (key or "").lower().strip().replace("-", "_")
