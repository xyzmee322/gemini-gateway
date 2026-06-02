from __future__ import annotations

import base64
import hashlib
import hmac
import re
from dataclasses import dataclass
from urllib.parse import quote

from cryptography.fernet import Fernet, InvalidToken
from pydantic import SecretStr

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_HOST_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")


@dataclass(frozen=True)
class ProxySecretParts:
    """Безопасное представление прокси для сохранения в БД."""

    scheme: str
    host: str
    port: int
    username: str | None = None
    password: str | None = None


class SecretVault:
    """Шифрует секреты и строит стабильные отпечатки без раскрытия значений."""

    def __init__(self, *, encryption_key: SecretStr | str, hmac_key: SecretStr | str) -> None:
        self._fernet = Fernet(_secret_value(encryption_key).encode("ascii"))
        self._hmac_key = _secret_value(hmac_key).encode("utf-8")

    def encrypt(self, value: str) -> str:
        normalized = _validate_secret_value(value)
        return self._fernet.encrypt(normalized.encode("utf-8")).decode("ascii")

    def decrypt(self, encrypted_value: str | None) -> str | None:
        if encrypted_value is None:
            return None
        try:
            return self._fernet.decrypt(encrypted_value.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise SecretDecryptionError("secret decryption failed") from exc

    def fingerprint(self, namespace: str, value: str) -> str:
        normalized = _validate_secret_value(value)
        digest = hmac.new(
            self._hmac_key,
            f"{namespace}\0{normalized}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def normalize_api_key(api_key: str) -> str:
    return _validate_secret_value(api_key)


def normalize_proxy_parts(parts: ProxySecretParts) -> ProxySecretParts:
    scheme = parts.scheme.strip().lower()
    host = parts.host.strip()
    if scheme not in {"http", "https"}:
        raise ValueError("proxy scheme must be http or https")
    if not host or _CONTROL_CHARS.search(host) or not _HOST_PATTERN.match(host):
        raise ValueError("proxy host is invalid")
    if parts.port < 1 or parts.port > 65535:
        raise ValueError("proxy port must be in range 1..65535")
    username = _normalize_optional_secret(parts.username)
    password = _normalize_optional_secret(parts.password)
    if password and not username:
        raise ValueError("proxy username is required when password is set")
    return ProxySecretParts(scheme=scheme, host=host, port=parts.port, username=username, password=password)


def build_proxy_url(parts: ProxySecretParts) -> str:
    normalized = normalize_proxy_parts(parts)
    credentials = ""
    if normalized.username:
        credentials = quote(normalized.username, safe="")
        if normalized.password:
            credentials += f":{quote(normalized.password, safe='')}"
        credentials += "@"
    return f"{normalized.scheme}://{credentials}{normalized.host}:{normalized.port}"


def proxy_fingerprint_source(parts: ProxySecretParts) -> str:
    normalized = normalize_proxy_parts(parts)
    username = normalized.username or ""
    return f"{normalized.scheme}://{username}@{normalized.host}:{normalized.port}"


def _secret_value(value: SecretStr | str) -> str:
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


class SecretDecryptionError(RuntimeError):
    """Ошибка расшифровки секрета без деталей криптобиблиотеки."""


def _validate_secret_value(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("secret must not be empty")
    if _CONTROL_CHARS.search(normalized):
        raise ValueError("secret must not contain control characters")
    return normalized


def _normalize_optional_secret(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if _CONTROL_CHARS.search(normalized):
        raise ValueError("proxy credentials must not contain control characters")
    return normalized
