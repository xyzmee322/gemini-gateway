from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_clone_gateway_data_sql_preserves_fk_order_and_sequences() -> None:
    sql = (ROOT / "scripts" / "clone_gateway_data.sql").read_text(encoding="utf-8")

    expected_order = [
        "google_projects",
        "proxy_endpoints",
        "api_keys",
        "key_proxy_bindings",
        "model_limits",
        "quota_windows",
        "cooldowns",
        "route_attempts",
    ]
    insert_positions = [sql.index(f"INSERT INTO gemini_gateway.{table}") for table in expected_order]

    assert insert_positions == sorted(insert_positions)
    for table in expected_order:
        assert f"SELECT setval(pg_get_serial_sequence('gemini_gateway.{table}', 'id')" in sql
    assert "TRUNCATE TABLE" in sql
    assert "soybob_v3" in sql
