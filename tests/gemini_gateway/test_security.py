from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from gemini_gateway.security import (
    ProxySecretParts,
    SecretVault,
    build_proxy_url,
    normalize_api_key,
    normalize_proxy_parts,
    proxy_fingerprint_source,
)


def test_secret_vault_encrypts_and_fingerprints_without_plaintext() -> None:
    vault = SecretVault(encryption_key=Fernet.generate_key().decode("ascii"), hmac_key="h" * 32)

    encrypted = vault.encrypt("AIza-secret-value")

    assert encrypted != "AIza-secret-value"
    assert vault.decrypt(encrypted) == "AIza-secret-value"
    assert vault.fingerprint("api-key", "AIza-secret-value") == vault.fingerprint("api-key", "AIza-secret-value")


def test_secret_vault_rejects_control_characters() -> None:
    vault = SecretVault(encryption_key=Fernet.generate_key().decode("ascii"), hmac_key="h" * 32)

    with pytest.raises(ValueError):
        vault.encrypt("bad\nsecret")


def test_proxy_url_escapes_credentials_and_normalizes_parts() -> None:
    parts = ProxySecretParts(
        scheme="HTTP",
        host="127.0.0.1",
        port=8080,
        username="user name",
        password="p@ss",
    )

    normalized = normalize_proxy_parts(parts)

    assert normalized.scheme == "http"
    assert build_proxy_url(parts) == "http://user%20name:p%40ss@127.0.0.1:8080"
    assert proxy_fingerprint_source(parts) == "http://user name@127.0.0.1:8080"


def test_proxy_validation_rejects_bad_scheme_and_host() -> None:
    with pytest.raises(ValueError):
        normalize_proxy_parts(ProxySecretParts(scheme="socks5", host="127.0.0.1", port=8080))
    with pytest.raises(ValueError):
        normalize_proxy_parts(ProxySecretParts(scheme="http", host="bad host", port=8080))


def test_api_key_normalization_rejects_blank_value() -> None:
    with pytest.raises(ValueError):
        normalize_api_key("   ")

