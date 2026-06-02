from __future__ import annotations

from pathlib import Path

from gemini_gateway.db.models import GATEWAY_SCHEMA

ROOT = Path(__file__).resolve().parents[2]


def test_gateway_owns_dedicated_schema() -> None:
    assert GATEWAY_SCHEMA == "gemini_gateway"


def test_runtime_sql_does_not_reference_soybob_schema() -> None:
    runtime_files = [
        ROOT / "gemini_gateway" / "repository.py",
        ROOT / "scripts" / "audit_gemini_gateway_routes.py",
        ROOT / "scripts" / "probe_gemini_gateway_proxies.py",
    ]

    for path in runtime_files:
        assert "soybob_v3" not in path.read_text(encoding="utf-8"), path


def test_gateway_models_do_not_import_soybob_database_package() -> None:
    source = (ROOT / "gemini_gateway" / "db" / "models.py").read_text(encoding="utf-8")

    assert "database.models" not in source
