from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_compose_config() -> dict[str, Any]:
    with (_PROJECT_ROOT / "docker-compose.yml").open(encoding="utf-8") as compose_file:
        return yaml.safe_load(compose_file)


def _service_environment(compose_config: dict[str, Any], service_name: str) -> dict[str, Any]:
    environment = compose_config["services"][service_name].get("environment", {})
    assert isinstance(environment, dict)
    return environment


def test_gateway_container_uses_init_reaper() -> None:
    compose_config = _load_compose_config()

    gateway_service = compose_config["services"]["gateway"]

    assert gateway_service.get("init") is True


def test_gateway_healthcheck_avoids_shell_wrapper_process() -> None:
    compose_config = _load_compose_config()

    healthcheck_command = compose_config["services"]["gateway"]["healthcheck"]["test"]

    assert healthcheck_command[0] == "CMD"
    assert healthcheck_command[1] == "python"
    assert "CMD-SHELL" not in healthcheck_command


def test_gateway_healthcheck_suppresses_python_tracebacks() -> None:
    compose_config = _load_compose_config()

    healthcheck_script = compose_config["services"]["gateway"]["healthcheck"]["test"][-1]

    assert "try:" in healthcheck_script
    assert "sys.exit(1)" in healthcheck_script


def test_gateway_compose_wires_openrouter_fallback_env() -> None:
    compose_config = _load_compose_config()
    environment = _service_environment(compose_config, "gateway")

    assert environment["GEMINI_GATEWAY_OPENROUTER_API_KEY"] == "${GEMINI_GATEWAY_OPENROUTER_API_KEY:-}"
    assert (
        environment["GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED"]
        == "${GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED:-false}"
    )
    assert (
        environment["GEMINI_GATEWAY_OPENROUTER_BASE_URL"]
        == "${GEMINI_GATEWAY_OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
    )


def test_dev_compose_wires_openrouter_fallback_env() -> None:
    with (_PROJECT_ROOT / "docker-compose.dev.yml").open(encoding="utf-8") as compose_file:
        compose_config = yaml.safe_load(compose_file)

    environment = _service_environment(compose_config, "gateway")

    assert environment["GEMINI_GATEWAY_OPENROUTER_API_KEY"] == "${GEMINI_GATEWAY_OPENROUTER_API_KEY:-}"
    assert (
        environment["GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED"]
        == "${GEMINI_GATEWAY_OPENROUTER_EMBEDDINGS_FALLBACK_ENABLED:-false}"
    )
    assert (
        environment["GEMINI_GATEWAY_OPENROUTER_BASE_URL"]
        == "${GEMINI_GATEWAY_OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
    )


def test_readme_documents_dev_compose_overlay_command() -> None:
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert (
        "docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres migrations gateway"
        in readme
    )
