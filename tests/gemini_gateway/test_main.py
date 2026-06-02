from __future__ import annotations

from pathlib import Path


def test_gateway_main_passes_environment_to_api_app() -> None:
    source = Path("gemini_gateway/main.py").read_text(encoding="utf-8")
    app_start = source.index("app = create_app(")
    app_end = source.index("    retention_service = GatewayRetentionService(")

    assert "environment=settings.environment" in source[app_start:app_end]
    assert "service_name=settings.service_name" in source[app_start:app_end]


def test_gateway_main_passes_service_name_to_retention_service() -> None:
    source = Path("gemini_gateway/main.py").read_text(encoding="utf-8")
    retention_start = source.index("retention_service = GatewayRetentionService(")
    retention_end = source.index("    retention_stop_event = asyncio.Event()")

    assert "service_name=settings.service_name" in source[retention_start:retention_end]
