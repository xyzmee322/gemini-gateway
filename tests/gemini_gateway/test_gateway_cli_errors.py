from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from gemini_gateway.config import GeminiGatewaySettings
from scripts import audit_gemini_gateway_routes as audit_cli
from scripts import probe_gemini_gateway_proxies as probe_cli
from scripts.gemini_gateway_cli_errors import gateway_cli_error_payload


def test_gateway_cli_error_payload_explains_missing_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GEMINI_GATEWAY_POSTGRES_DSN",
        "GEMINI_GATEWAY_POSTGRES_SYNC_DSN",
        "GEMINI_GATEWAY_ENCRYPTION_KEY",
        "GEMINI_GATEWAY_HMAC_KEY",
        "GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValidationError) as exc_info:
        GeminiGatewaySettings(_env_file=None)

    payload = gateway_cli_error_payload(exc_info.value, fallback_error="Не удалось проверить gateway.")

    assert payload["status"] == "error"
    assert payload["reason"] == "gateway_env_missing"
    assert payload["missing_env"] == [
        "GEMINI_GATEWAY_ENCRYPTION_KEY",
        "GEMINI_GATEWAY_HMAC_KEY",
        "GEMINI_GATEWAY_INTERNAL_AUTH_TOKEN",
        "GEMINI_GATEWAY_POSTGRES_DSN",
        "GEMINI_GATEWAY_POSTGRES_SYNC_DSN",
    ]
    assert "gemini-gateway" in payload["error"]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "ValidationError" not in serialized
    assert "secret-token" not in serialized


def test_gateway_cli_error_payload_keeps_unknown_errors_generic() -> None:
    payload = gateway_cli_error_payload(RuntimeError("raw database trace"), fallback_error="Не удалось проверить gateway.")

    assert payload == {
        "status": "error",
        "reason": "gateway_cli_error",
        "error": "Не удалось проверить gateway.",
    }


def test_audit_cli_honors_exit_zero_on_safe_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def failing_audit(**_: object) -> dict[str, object]:
        raise RuntimeError("raw database trace")

    monkeypatch.setattr(audit_cli, "audit_gemini_gateway_routes", failing_audit)

    exit_code = audit_cli.main(["--exit-zero"])

    payload = json.loads(capsys.readouterr().err)
    assert exit_code == 0
    assert payload["reason"] == "gateway_cli_error"
    assert "raw database trace" not in json.dumps(payload, ensure_ascii=False)


def test_probe_cli_honors_exit_zero_on_safe_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def failing_probe(**_: object) -> dict[str, object]:
        raise RuntimeError("raw proxy trace")

    monkeypatch.setattr(probe_cli, "probe_gemini_gateway_proxies", failing_probe)

    exit_code = probe_cli.main(["--exit-zero"])

    payload = json.loads(capsys.readouterr().err)
    assert exit_code == 0
    assert payload["reason"] == "gateway_cli_error"
    assert "raw proxy trace" not in json.dumps(payload, ensure_ascii=False)
