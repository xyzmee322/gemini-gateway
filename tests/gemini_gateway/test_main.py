from __future__ import annotations

from pathlib import Path


def test_gateway_main_passes_environment_to_api_app() -> None:
    source = Path("gemini_gateway/main.py").read_text(encoding="utf-8")
    app_start = source.index("app = create_app(")
    app_end = source.index("    retention_service = GatewayRetentionService(")

    assert "environment=settings.environment" in source[app_start:app_end]
    assert "service_name=settings.service_name" in source[app_start:app_end]


def test_gateway_main_passes_max_route_attempts_to_completion_service() -> None:
    source = Path("gemini_gateway/main.py").read_text(encoding="utf-8")
    service_start = source.index("service = CompletionService(")
    service_end = source.index("    app = create_app(")

    assert "max_route_attempts=settings.max_route_attempts" in source[service_start:service_end]


def test_gateway_main_wires_openlux_chat_fallback_and_pricing() -> None:
    source = Path("gemini_gateway/main.py").read_text(encoding="utf-8")
    service_start = source.index("service = CompletionService(")
    service_end = source.index("    app = create_app(")

    assert "from gemini_gateway.openlux_chat_client import OpenLuxChatClient" in source
    assert "from gemini_gateway.openlux_pricing import OpenLuxPricingCatalog" in source
    assert source.index("openlux_pricing_catalog = OpenLuxPricingCatalog(") < service_start
    assert source.index("openlux_chat_client = OpenLuxChatClient(") < service_start
    assert "openlux_chat_client=openlux_chat_client" in source[service_start:service_end]
    assert "openlux_api_key=openlux_api_key" in source[service_start:service_end]
    assert "openlux_chat_mode=settings.openlux_chat_mode" in source[service_start:service_end]
    assert "openlux_chat_model=settings.openlux_chat_model" in source[service_start:service_end]


def test_gateway_main_passes_service_name_to_retention_service() -> None:
    source = Path("gemini_gateway/main.py").read_text(encoding="utf-8")
    retention_start = source.index("retention_service = GatewayRetentionService(")
    retention_end = source.index("    retention_stop_event = asyncio.Event()")

    assert "service_name=settings.service_name" in source[retention_start:retention_end]
