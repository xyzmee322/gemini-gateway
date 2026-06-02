from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from gemini_gateway.contracts import GatewayChatRequest, GatewayRouteMetadata, RouteLease, SeedConfig
from gemini_gateway.db.models import GatewayBase, KeyProxyBinding

ROOT = Path(__file__).resolve().parents[2]


def test_gateway_models_use_project_schema() -> None:
    schemas = {table.schema for table in GatewayBase.metadata.tables.values()}
    names = {table.name for table in GatewayBase.metadata.tables.values()}

    assert schemas == {"gemini_gateway"}
    assert {
        "google_projects",
        "api_keys",
        "proxy_endpoints",
        "key_proxy_bindings",
        "model_limits",
        "quota_windows",
        "cooldowns",
        "route_attempts",
    }.issubset(names)


def test_key_proxy_binding_model_is_proxy_only() -> None:
    table = KeyProxyBinding.__table__
    constraints = {constraint.name: str(constraint.sqltext) for constraint in table.constraints if constraint.name}

    assert table.c.proxy_id.nullable is False
    assert constraints["ck_key_proxy_bindings_transport_mode"] == "transport_mode = 'proxy'"
    assert constraints["ck_key_proxy_bindings_transport_proxy"] == "proxy_id IS NOT NULL"


def test_gateway_schema_migration_is_proxy_only() -> None:
    migration = (ROOT / "migrations/versions/0001_gateway_schema.py").read_text(encoding="utf-8")

    assert 'revision = "0001_gateway_schema"' in migration
    assert "down_revision = None" in migration
    assert "transport_mode = 'proxy'" in migration
    assert "proxy_id IS NOT NULL" in migration
    assert "ck_route_attempts_retry_count" in migration
    assert "source_service" in migration
    assert "chat_id" in migration


def test_gateway_chat_request_preserves_openai_shape() -> None:
    request = GatewayChatRequest(
        request_id="req-1",
        source_service="worker",
        model="google/gemini-3.5-flash",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "send_message"}}],
        tool_choice="auto",
        response_format={"type": "json_object"},
        reasoning={"effort": "high"},
    )

    assert request.messages[0]["role"] == "user"
    assert request.tools is not None
    assert request.reasoning == {"effort": "high"}


def test_gateway_chat_request_rejects_empty_messages() -> None:
    with pytest.raises(ValidationError):
        GatewayChatRequest(request_id="req-1", source_service="worker", model="m", messages=[])


def test_gateway_route_metadata_rejects_direct_transport_mode() -> None:
    with pytest.raises(ValidationError):
        GatewayRouteMetadata(
            route_label="route-direct",
            project_label="local",
            key_label="key",
            transport_mode="direct",
        )


def test_gateway_route_metadata_requires_proxy_label() -> None:
    with pytest.raises(ValidationError):
        GatewayRouteMetadata(
            route_label="route-proxy",
            project_label="local",
            key_label="key",
        )


def test_route_lease_requires_proxy_identity() -> None:
    payload = {
        "attempt_id": "attempt-1",
        "binding_id": 1,
        "project_id": 2,
        "api_key_id": 3,
        "proxy_id": 4,
        "api_key": "AIza-key",
        "proxy_url": "http://user:pass@127.0.0.1:8080",
        "model": "google/gemini-3.5-flash",
        "route_label": "route-a",
        "project_label": "project-a",
        "key_label": "key-a",
        "proxy_label": "proxy-a",
        "transport_mode": "proxy",
        "estimated_tokens": 100,
        "leased_at": datetime(2026, 5, 30, tzinfo=UTC),
    }

    for field_name in ("proxy_id", "proxy_url", "proxy_label"):
        with pytest.raises(ValidationError):
            RouteLease.model_validate({**payload, field_name: None})


def test_seed_config_requires_model_limits_and_valid_refs() -> None:
    config = SeedConfig.model_validate(
        {
            "projects": [
                {
                    "label": "friend-a",
                    "owner_name": "Friend",
                    "model_limits": [
                        {
                            "model": "google/gemini-3.5-flash",
                            "requests_per_minute": 10,
                            "tokens_per_minute": 100000,
                            "requests_per_day": 1000,
                        }
                    ],
                }
            ],
            "api_keys": [{"project_label": "friend-a", "label": "key-a", "api_key": "secret"}],
            "proxies": [{"label": "proxy-a", "host": "127.0.0.1", "port": 8080}],
            "bindings": [{"label": "route-a", "api_key_label": "key-a", "proxy_label": "proxy-a"}],
        }
    )

    assert config.projects[0].model_limits[0].requests_per_minute == 10


def test_seed_config_rejects_direct_binding_without_proxy() -> None:
    with pytest.raises(ValidationError, match="direct bindings are disabled"):
        SeedConfig.model_validate(
            {
                "projects": [
                    {
                        "label": "local",
                        "owner_name": "Local",
                        "model_limits": [
                            {
                                "model": "google/gemini-3.5-flash",
                                "requests_per_minute": 10,
                                "tokens_per_minute": 100000,
                                "requests_per_day": 1000,
                            }
                        ],
                    }
                ],
                "api_keys": [{"project_label": "local", "label": "local-key", "api_key": "secret"}],
                "bindings": [
                    {
                        "label": "local-direct",
                        "project_label": "local",
                        "api_key_label": "local-key",
                        "transport_mode": "direct",
                    }
                ],
            }
        )


def test_seed_config_rejects_proxy_binding_without_proxy_label() -> None:
    with pytest.raises(ValidationError, match="proxy binding requires proxy_label"):
        SeedConfig.model_validate(
            {
                "projects": [
                    {
                        "label": "friend-a",
                        "owner_name": "Friend",
                        "model_limits": [
                            {
                                "model": "google/gemini-3.5-flash",
                                "requests_per_minute": 10,
                                "tokens_per_minute": 100000,
                                "requests_per_day": 1000,
                            }
                        ],
                    }
                ],
                "api_keys": [{"project_label": "friend-a", "label": "key-a", "api_key": "secret"}],
                "bindings": [{"label": "route-a", "api_key_label": "key-a"}],
            }
        )


def test_seed_config_rejects_direct_binding_with_proxy_label() -> None:
    with pytest.raises(ValidationError, match="direct bindings are disabled"):
        SeedConfig.model_validate(
            {
                "projects": [
                    {
                        "label": "local",
                        "owner_name": "Local",
                        "model_limits": [
                            {
                                "model": "google/gemini-3.5-flash",
                                "requests_per_minute": 10,
                                "tokens_per_minute": 100000,
                                "requests_per_day": 1000,
                            }
                        ],
                    }
                ],
                "api_keys": [{"project_label": "local", "label": "local-key", "api_key": "secret"}],
                "proxies": [{"label": "proxy-a", "host": "127.0.0.1", "port": 8080}],
                "bindings": [
                    {
                        "label": "local-direct",
                        "api_key_label": "local-key",
                        "transport_mode": "direct",
                        "proxy_label": "proxy-a",
                    }
                ],
            }
        )


def test_seed_config_rejects_unknown_binding_refs() -> None:
    with pytest.raises(ValidationError):
        SeedConfig.model_validate(
            {
                "projects": [
                    {
                        "label": "friend-a",
                        "owner_name": "Friend",
                        "model_limits": [
                            {
                                "model": "google/gemini-3.5-flash",
                                "requests_per_minute": 10,
                                "tokens_per_minute": 100000,
                                "requests_per_day": 1000,
                            }
                        ],
                    }
                ],
                "api_keys": [{"project_label": "friend-a", "label": "key-a", "api_key": "secret"}],
                "proxies": [{"label": "proxy-a", "host": "127.0.0.1", "port": 8080}],
                "bindings": [{"label": "route-a", "api_key_label": "missing", "proxy_label": "proxy-a"}],
            }
        )
