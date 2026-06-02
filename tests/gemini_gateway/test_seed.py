from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

import gemini_gateway.seed as seed_module
from gemini_gateway.seed import import_seed_file, load_seed_config


@dataclass
class _FakeGatewaySettings:
    environment: str


def test_load_seed_config_requires_model_limits(tmp_path) -> None:
    seed_path = tmp_path / "gateway-seed.yaml"
    seed_path.write_text(
        """
projects:
  - label: friend-a
    owner_name: Friend A
    model_limits:
      - model: google/gemini-3.5-flash
        requests_per_minute: 10
        tokens_per_minute: 100000
        requests_per_day: 1000
api_keys:
  - project_label: friend-a
    label: key-a
    api_key: AIza-secret
proxies:
  - label: proxy-a
    host: 127.0.0.1
    port: 8080
bindings:
  - label: route-a
    api_key_label: key-a
    proxy_label: proxy-a
""".strip(),
        encoding="utf-8",
    )

    config = load_seed_config(seed_path)

    assert config.projects[0].model_limits[0].model == "google/gemini-3.5-flash"


def test_load_seed_config_expands_environment_variables(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SEED_TEST_API_KEY", "AIza-expanded-from-env")
    seed_path = tmp_path / "env-seed.yaml"
    seed_path.write_text(
        """
projects:
  - label: friend-a
    owner_name: Friend A
    model_limits:
      - model: google/gemini-3.5-flash
        requests_per_minute: 10
        tokens_per_minute: 100000
        requests_per_day: 1000
api_keys:
  - project_label: friend-a
    label: key-a
    api_key: ${SEED_TEST_API_KEY}
proxies:
  - label: proxy-a
    host: 127.0.0.1
    port: 8080
bindings:
  - label: route-a
    project_label: friend-a
    api_key_label: key-a
    proxy_label: proxy-a
""".strip(),
        encoding="utf-8",
    )

    config = load_seed_config(seed_path)

    assert config.api_keys[0].api_key == "AIza-expanded-from-env"


def test_load_seed_config_rejects_missing_environment_variable(tmp_path) -> None:
    seed_path = tmp_path / "missing-env-seed.yaml"
    seed_path.write_text(
        """
projects:
  - label: friend-a
    owner_name: Friend A
    model_limits:
      - model: google/gemini-3.5-flash
        requests_per_minute: 10
        tokens_per_minute: 100000
        requests_per_day: 1000
api_keys:
  - project_label: friend-a
    label: key-a
    api_key: ${MISSING_SEED_TEST_API_KEY}
proxies:
  - label: proxy-a
    host: 127.0.0.1
    port: 8080
bindings:
  - label: route-a
    project_label: friend-a
    api_key_label: key-a
    proxy_label: proxy-a
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="MISSING_SEED_TEST_API_KEY"):
        load_seed_config(seed_path)


@pytest.mark.asyncio
async def test_import_seed_file_skips_missing_dev_env_when_routes_exist(tmp_path, monkeypatch) -> None:
    seed_path = tmp_path / "missing-dev-env-seed.yaml"
    seed_path.write_text(
        """
projects:
  - label: local-dev
    owner_name: Local Dev
    model_limits:
      - model: google/gemini-3.5-flash
        requests_per_minute: 10
        tokens_per_minute: 100000
        requests_per_day: 1000
api_keys:
  - project_label: local-dev
    label: key-a
    api_key: AIza-secret
proxies:
  - label: proxy-a
    url: ${MISSING_DEV_PROXY_URL}
bindings:
  - label: route-a
    project_label: local-dev
    api_key_label: key-a
    proxy_label: proxy-a
""".strip(),
        encoding="utf-8",
    )

    class _FakeSeedRepository:
        def __init__(self) -> None:
            self.health_require_routes: bool | None = None
            self.upsert_calls = 0

        async def health_check(self, *, require_routes: bool = False) -> dict[str, object]:
            self.health_require_routes = require_routes
            return {"ok": True, "checks": {"routes": True}}

        async def upsert_seed_config(self, config: object) -> None:
            self.upsert_calls += 1

    repository = _FakeSeedRepository()
    monkeypatch.setattr(seed_module, "get_gateway_settings", lambda: _FakeGatewaySettings(environment="development"))
    monkeypatch.setattr(seed_module, "_build_seed_repository", lambda settings: repository)

    await import_seed_file(seed_path)

    assert repository.health_require_routes is True
    assert repository.upsert_calls == 0


@pytest.mark.asyncio
async def test_import_seed_file_rejects_missing_dev_env_when_routes_absent(tmp_path, monkeypatch) -> None:
    seed_path = tmp_path / "missing-dev-env-empty-routes-seed.yaml"
    seed_path.write_text(
        """
projects:
  - label: local-dev
    owner_name: Local Dev
    model_limits:
      - model: google/gemini-3.5-flash
        requests_per_minute: 10
        tokens_per_minute: 100000
        requests_per_day: 1000
api_keys:
  - project_label: local-dev
    label: key-a
    api_key: AIza-secret
proxies:
  - label: proxy-a
    url: ${MISSING_DEV_PROXY_URL}
bindings:
  - label: route-a
    project_label: local-dev
    api_key_label: key-a
    proxy_label: proxy-a
""".strip(),
        encoding="utf-8",
    )

    class _FakeSeedRepository:
        async def health_check(self, *, require_routes: bool = False) -> dict[str, object]:
            return {"ok": False, "checks": {"routes": False}}

        async def upsert_seed_config(self, config: object) -> None:
            raise AssertionError("seed import must not upsert incomplete seed config")

    monkeypatch.setattr(seed_module, "get_gateway_settings", lambda: _FakeGatewaySettings(environment="development"))
    monkeypatch.setattr(seed_module, "_build_seed_repository", lambda settings: _FakeSeedRepository())

    with pytest.raises(ValueError, match="MISSING_DEV_PROXY_URL"):
        await import_seed_file(seed_path)


@pytest.mark.asyncio
async def test_import_seed_file_rejects_missing_env_outside_development(tmp_path, monkeypatch) -> None:
    seed_path = tmp_path / "missing-production-env-seed.yaml"
    seed_path.write_text(
        """
projects:
  - label: prod
    owner_name: Production
    model_limits:
      - model: google/gemini-3.5-flash
        requests_per_minute: 10
        tokens_per_minute: 100000
        requests_per_day: 1000
api_keys:
  - project_label: prod
    label: key-a
    api_key: AIza-secret
proxies:
  - label: proxy-a
    url: ${MISSING_PRODUCTION_PROXY_URL}
bindings:
  - label: route-a
    project_label: prod
    api_key_label: key-a
    proxy_label: proxy-a
""".strip(),
        encoding="utf-8",
    )

    class _FakeSeedRepository:
        async def health_check(self, *, require_routes: bool = False) -> dict[str, object]:
            raise AssertionError("production seed must not skip missing environment")

        async def upsert_seed_config(self, config: object) -> None:
            raise AssertionError("seed import must not upsert incomplete seed config")

    monkeypatch.setattr(seed_module, "get_gateway_settings", lambda: _FakeGatewaySettings(environment="production"))
    monkeypatch.setattr(seed_module, "_build_seed_repository", lambda settings: _FakeSeedRepository())

    with pytest.raises(ValueError, match="MISSING_PRODUCTION_PROXY_URL"):
        await import_seed_file(seed_path)


def test_load_seed_config_rejects_missing_model_limits(tmp_path) -> None:
    seed_path = tmp_path / "bad-seed.yaml"
    seed_path.write_text(
        """
projects:
  - label: friend-a
    owner_name: Friend A
api_keys:
  - project_label: friend-a
    label: key-a
    api_key: AIza-secret
proxies:
  - label: proxy-a
    host: 127.0.0.1
    port: 8080
bindings:
  - label: route-a
    api_key_label: key-a
    proxy_label: proxy-a
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_seed_config(seed_path)


def test_load_seed_config_accepts_proxy_url_shortcut(tmp_path) -> None:
    seed_path = tmp_path / "proxy-url-seed.yaml"
    seed_path.write_text(
        """
projects:
  - label: friend-a
    owner_name: Friend A
    model_limits:
      - model: google/gemini-3.5-flash
        requests_per_minute: 10
        tokens_per_minute: 100000
        requests_per_day: 1000
api_keys:
  - project_label: friend-a
    label: key-a
    api_key: AIza-secret
proxies:
  - label: proxy-a
    url: http://user:pa%40ss@127.0.0.1:8080
bindings:
  - label: route-a
    api_key_label: key-a
    proxy_label: proxy-a
""".strip(),
        encoding="utf-8",
    )

    config = load_seed_config(seed_path)
    proxy = config.proxies[0]

    assert proxy.scheme == "http"
    assert proxy.host == "127.0.0.1"
    assert proxy.port == 8080
    assert proxy.username == "user"
    assert proxy.password == "pa@ss"


def test_seed_config_rejects_ambiguous_key_label_without_project_label(tmp_path) -> None:
    seed_path = tmp_path / "ambiguous-seed.yaml"
    seed_path.write_text(
        """
projects:
  - label: friend-a
    owner_name: Friend A
    model_limits:
      - model: google/gemini-3.5-flash
        requests_per_minute: 10
        tokens_per_minute: 100000
        requests_per_day: 1000
  - label: friend-b
    owner_name: Friend B
    model_limits:
      - model: google/gemini-3.5-flash
        requests_per_minute: 10
        tokens_per_minute: 100000
        requests_per_day: 1000
api_keys:
  - project_label: friend-a
    label: shared-key
    api_key: AIza-secret-a
  - project_label: friend-b
    label: shared-key
    api_key: AIza-secret-b
proxies:
  - label: proxy-a
    host: 127.0.0.1
    port: 8080
bindings:
  - label: route-a
    api_key_label: shared-key
    proxy_label: proxy-a
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="ambiguous api key label"):
        load_seed_config(seed_path)


def test_seed_config_allows_duplicate_key_label_with_project_label(tmp_path) -> None:
    seed_path = tmp_path / "scoped-seed.yaml"
    seed_path.write_text(
        """
projects:
  - label: friend-a
    owner_name: Friend A
    model_limits:
      - model: google/gemini-3.5-flash
        requests_per_minute: 10
        tokens_per_minute: 100000
        requests_per_day: 1000
  - label: friend-b
    owner_name: Friend B
    model_limits:
      - model: google/gemini-3.5-flash
        requests_per_minute: 10
        tokens_per_minute: 100000
        requests_per_day: 1000
api_keys:
  - project_label: friend-a
    label: shared-key
    api_key: AIza-secret-a
  - project_label: friend-b
    label: shared-key
    api_key: AIza-secret-b
proxies:
  - label: proxy-a
    host: 127.0.0.1
    port: 8080
bindings:
  - label: route-a
    project_label: friend-b
    api_key_label: shared-key
    proxy_label: proxy-a
""".strip(),
        encoding="utf-8",
    )

    config = load_seed_config(seed_path)

    assert config.bindings[0].project_label == "friend-b"
