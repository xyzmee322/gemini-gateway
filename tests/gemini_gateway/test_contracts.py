from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from gemini_gateway.contracts import (
    GatewayChatRequest,
    GatewayEmbeddingInputPart,
    GatewayEmbeddingResponse,
    GatewayRouteMetadata,
    RouteCandidate,
    RouteLease,
    SeedBinding,
    SeedConfig,
)
from gemini_gateway.db.models import GatewayBase, KeyProxyBinding

ROOT = Path(__file__).resolve().parents[2]


def _transport_mode_schema_values(model_type: type) -> list[str]:
    schema = model_type.model_json_schema()
    transport_schema = schema["properties"]["transport_mode"]
    if "$ref" in transport_schema:
        ref_name = transport_schema["$ref"].rpartition("/")[-1]
        transport_schema = schema["$defs"][ref_name]
    if "enum" in transport_schema:
        return transport_schema["enum"]
    if "const" in transport_schema:
        return [transport_schema["const"]]
    raise AssertionError(f"transport_mode schema does not expose enum/const: {transport_schema}")


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


def test_gateway_route_metadata_allows_direct_transport_mode_without_proxy_label() -> None:
    metadata = GatewayRouteMetadata(
        route_label="route-direct",
        project_label="local",
        key_label="key",
        transport_mode="direct",
    )

    assert metadata.transport_mode == "direct"
    assert metadata.proxy_label is None


def test_gateway_route_metadata_allows_missing_proxy_label_for_response_metadata() -> None:
    metadata = GatewayRouteMetadata(
        route_label="route-proxy",
        project_label="local",
        key_label="key",
    )

    assert metadata.transport_mode == "proxy"
    assert metadata.proxy_label is None


def test_gateway_route_metadata_rejects_blank_proxy_label_when_present() -> None:
    with pytest.raises(ValidationError):
        GatewayRouteMetadata(
            route_label="route-proxy",
            project_label="local",
            key_label="key",
            proxy_label="",
        )


def test_embedding_response_allows_direct_fallback_route_metadata() -> None:
    response = GatewayEmbeddingResponse(
        request_id="req-direct-route",
        model="google/gemini-embedding-2",
        embedding=[0.1, 0.2, 0.3],
        dimensions=3,
        route={
            "project_label": "openrouter-fallback",
            "route_label": "openrouter-embedding-fallback",
            "key_label": "openrouter-api-key",
            "proxy_label": None,
            "transport_mode": "direct",
        },
    )

    assert response.route["transport_mode"] == "direct"
    assert response.route["proxy_label"] is None
    serialized_route = response.model_dump(mode="json", exclude_none=True)["route"]
    assert "proxy_label" not in serialized_route


def test_embedding_response_rejects_blank_proxy_label_in_route_dict() -> None:
    with pytest.raises(ValidationError):
        GatewayEmbeddingResponse(
            request_id="req-direct-route-invalid",
            model="google/gemini-embedding-2",
            embedding=[0.1, 0.2, 0.3],
            dimensions=3,
            route={
                "project_label": "openrouter-fallback",
                "route_label": "openrouter-embedding-fallback",
                "key_label": "openrouter-api-key",
                "proxy_label": "",
                "transport_mode": "direct",
            },
        )


def test_embedding_response_normalizes_valid_route_dict_and_omits_none_proxy_label() -> None:
    response = GatewayEmbeddingResponse(
        request_id="req-direct-route-valid",
        model="google/gemini-embedding-2",
        embedding=[0.1, 0.2, 0.3],
        dimensions=3,
        route={
            "project_label": "openrouter-fallback",
            "route_label": "openrouter-embedding-fallback",
            "key_label": "openrouter-api-key",
            "proxy_label": None,
            "transport_mode": "direct",
            "fallback_reason": "primary_route_unavailable",
        },
    )

    assert response.route["transport_mode"] == "direct"
    assert response.route["fallback_reason"] == "primary_route_unavailable"
    serialized_route = response.model_dump(mode="json", exclude_none=True)["route"]
    assert "proxy_label" not in serialized_route
    assert serialized_route["fallback_reason"] == "primary_route_unavailable"


def test_embedding_response_allows_empty_default_route_metadata() -> None:
    response = GatewayEmbeddingResponse(
        request_id="req-empty-route",
        model="google/gemini-embedding-2",
        embedding=[0.1, 0.2, 0.3],
        dimensions=3,
    )

    assert response.route == {}


def test_embedding_input_part_accepts_inline_audio_data() -> None:
    part = GatewayEmbeddingInputPart.model_validate(
        {
            "type": "inline_data",
            "inline_data": {"mimeType": " audio/wav ", "data": " UklGRg== "},
        }
    )

    assert part.inline_data == {"mime_type": "audio/wav", "data": "UklGRg=="}
    assert part.model_dump(mode="python", exclude_none=True) == {
        "type": "inline_data",
        "inline_data": {"mime_type": "audio/wav", "data": "UklGRg=="},
    }


def test_embedding_input_part_rejects_non_image_data_url_image_url() -> None:
    with pytest.raises(ValidationError):
        GatewayEmbeddingInputPart.model_validate(
            {
                "type": "image_url",
                "image_url": {"url": "data:audio/wav;base64,UklGRg=="},
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "inline_data"},
        {"type": "inline_data", "inline_data": {}},
        {"type": "inline_data", "inline_data": {"mime_type": "audio/wav"}},
        {"type": "inline_data", "inline_data": {"mimeType": "audio/wav", "data": "   "}},
        {"type": "inline_data", "inline_data": {"mime_type": "   ", "data": "UklGRg=="}},
    ],
)
def test_embedding_input_part_rejects_inline_data_without_payload(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GatewayEmbeddingInputPart.model_validate(payload)


def test_route_transport_schema_is_proxy_only_for_gemini_routes() -> None:
    assert _transport_mode_schema_values(RouteCandidate) == ["proxy"]
    assert _transport_mode_schema_values(RouteLease) == ["proxy"]
    assert _transport_mode_schema_values(SeedBinding) == ["proxy"]


def test_gateway_route_metadata_transport_schema_allows_direct_metadata() -> None:
    assert _transport_mode_schema_values(GatewayRouteMetadata) == ["proxy", "direct"]


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


def test_route_lease_rejects_direct_transport_mode_for_gemini_routes() -> None:
    with pytest.raises(ValidationError, match="direct transport is disabled"):
        RouteLease(
            attempt_id="attempt-1",
            binding_id=1,
            project_id=2,
            api_key_id=3,
            proxy_id=4,
            api_key="AIza-key",
            proxy_url="http://user:pass@127.0.0.1:8080",
            model="google/gemini-3.5-flash",
            route_label="route-a",
            project_label="project-a",
            key_label="key-a",
            proxy_label="proxy-a",
            transport_mode="direct",
            estimated_tokens=100,
            leased_at=datetime(2026, 5, 30, tzinfo=UTC),
        )


def test_route_candidate_rejects_direct_transport_mode_for_gemini_routing() -> None:
    with pytest.raises(ValidationError, match="direct transport is disabled"):
        RouteCandidate(
            binding_id=1,
            project_id=2,
            api_key_id=3,
            proxy_id=4,
            api_key="AIza-key",
            proxy_url="http://user:pass@127.0.0.1:8080",
            model="google/gemini-3.5-flash",
            route_label="route-a",
            project_label="project-a",
            key_label="key-a",
            proxy_label="proxy-a",
            transport_mode="direct",
            requests_per_minute=10,
            tokens_per_minute=100000,
            requests_per_day=1000,
        )


def test_seed_binding_rejects_direct_transport_mode() -> None:
    with pytest.raises(ValidationError, match="direct bindings are disabled"):
        SeedBinding(
            label="route",
            api_key_label="key",
            transport_mode="direct",
            proxy_label="proxy",
        )


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
