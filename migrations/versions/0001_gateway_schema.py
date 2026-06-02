"""Create standalone Gemini Gateway schema.

Revision ID: 0001_gateway_schema
Revises: 
Create Date: 2026-06-02 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_gateway_schema"
down_revision = None
branch_labels = None
depends_on = None

SCHEMA = "gemini_gateway"


def upgrade() -> None:
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
    op.create_table(
        "google_projects",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("label", sa.String(length=80), nullable=False),
        sa.Column("owner_name", sa.String(length=160), nullable=False),
        sa.Column("owner_contact", sa.String(length=255), nullable=True),
        sa.Column("google_project_ref", sa.String(length=255), nullable=True),
        sa.Column("tier", sa.String(length=64), server_default=sa.text("'free'"), nullable=False),
        sa.Column("status", sa.String(length=32), server_default=sa.text("'active'"), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "status IN ('active', 'paused', 'disabled_needs_review')",
            name="ck_google_projects_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("label", name="uq_google_projects_label"),
        schema=SCHEMA,
    )
    op.create_table(
        "api_keys",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("project_id", sa.BigInteger(), nullable=False),
        sa.Column("label", sa.String(length=80), nullable=False),
        sa.Column("encrypted_api_key", sa.Text(), nullable=False),
        sa.Column("key_fingerprint", sa.String(length=96), nullable=False),
        sa.Column("status", sa.String(length=32), server_default=sa.text("'active'"), nullable=False),
        sa.Column("cooldown_level", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("sleep_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_failure_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error_type", sa.String(length=64), nullable=True),
        sa.Column("last_error_message_safe", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('active', 'paused', 'disabled_needs_review')", name="ck_api_keys_status"),
        sa.CheckConstraint("cooldown_level >= 0", name="ck_api_keys_cooldown_level"),
        sa.ForeignKeyConstraint(["project_id"], [f"{SCHEMA}.google_projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key_fingerprint", name="uq_api_keys_fingerprint"),
        sa.UniqueConstraint("project_id", "label", name="uq_api_keys_project_label"),
        schema=SCHEMA,
    )
    op.create_table(
        "proxy_endpoints",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("label", sa.String(length=80), nullable=False),
        sa.Column("scheme", sa.String(length=16), server_default=sa.text("'http'"), nullable=False),
        sa.Column("host", sa.String(length=255), nullable=False),
        sa.Column("port", sa.Integer(), nullable=False),
        sa.Column("encrypted_username", sa.Text(), nullable=True),
        sa.Column("encrypted_password", sa.Text(), nullable=True),
        sa.Column("proxy_fingerprint", sa.String(length=96), nullable=False),
        sa.Column("status", sa.String(length=32), server_default=sa.text("'active'"), nullable=False),
        sa.Column("cooldown_level", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("sleep_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_failure_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error_type", sa.String(length=64), nullable=True),
        sa.Column("last_error_message_safe", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('active', 'paused', 'disabled_needs_review')", name="ck_proxy_endpoints_status"),
        sa.CheckConstraint("scheme IN ('http', 'https')", name="ck_proxy_endpoints_scheme"),
        sa.CheckConstraint("port BETWEEN 1 AND 65535", name="ck_proxy_endpoints_port"),
        sa.CheckConstraint("cooldown_level >= 0", name="ck_proxy_endpoints_cooldown_level"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("label", name="uq_proxy_endpoints_label"),
        sa.UniqueConstraint("proxy_fingerprint", name="uq_proxy_endpoints_fingerprint"),
        schema=SCHEMA,
    )
    op.create_table(
        "key_proxy_bindings",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("api_key_id", sa.BigInteger(), nullable=False),
        sa.Column("proxy_id", sa.BigInteger(), nullable=False),
        sa.Column("label", sa.String(length=80), nullable=False),
        sa.Column("transport_mode", sa.String(length=16), server_default=sa.text("'proxy'"), nullable=False),
        sa.Column("status", sa.String(length=32), server_default=sa.text("'active'"), nullable=False),
        sa.Column("cooldown_level", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("sleep_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consecutive_failures", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_failure_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error_type", sa.String(length=64), nullable=True),
        sa.Column("last_error_message_safe", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "status IN ('active', 'paused', 'disabled_needs_review')",
            name="ck_key_proxy_bindings_status",
        ),
        sa.CheckConstraint("transport_mode = 'proxy'", name="ck_key_proxy_bindings_transport_mode"),
        sa.CheckConstraint("proxy_id IS NOT NULL", name="ck_key_proxy_bindings_transport_proxy"),
        sa.CheckConstraint("consecutive_failures >= 0", name="ck_key_proxy_bindings_failures"),
        sa.CheckConstraint("cooldown_level >= 0", name="ck_key_proxy_bindings_cooldown_level"),
        sa.ForeignKeyConstraint(["api_key_id"], [f"{SCHEMA}.api_keys.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["proxy_id"], [f"{SCHEMA}.proxy_endpoints.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("label", name="uq_key_proxy_bindings_label"),
        schema=SCHEMA,
    )
    op.create_index(
        "uq_key_proxy_bindings_active_key",
        "key_proxy_bindings",
        ["api_key_id"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("status = 'active' AND proxy_id IS NOT NULL"),
    )
    op.create_index(
        "uq_key_proxy_bindings_active_proxy",
        "key_proxy_bindings",
        ["proxy_id"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_table(
        "model_limits",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("project_id", sa.BigInteger(), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("requests_per_minute", sa.Integer(), nullable=False),
        sa.Column("tokens_per_minute", sa.Integer(), nullable=False),
        sa.Column("requests_per_day", sa.Integer(), nullable=False),
        sa.Column("tokens_per_day", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), server_default=sa.text("'active'"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('active', 'paused')", name="ck_model_limits_status"),
        sa.CheckConstraint(
            "requests_per_minute > 0 AND tokens_per_minute > 0 AND requests_per_day > 0",
            name="ck_model_limits_positive",
        ),
        sa.ForeignKeyConstraint(["project_id"], [f"{SCHEMA}.google_projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "model", name="uq_model_limits_project_model"),
        schema=SCHEMA,
    )
    op.create_table(
        "quota_windows",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("project_id", sa.BigInteger(), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("window_kind", sa.String(length=16), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("requests_used", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("input_tokens_reserved", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("input_tokens_used", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("output_tokens_used", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("total_tokens_used", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("window_kind IN ('minute', 'day')", name="ck_quota_windows_kind"),
        sa.CheckConstraint(
            "requests_used >= 0 AND input_tokens_reserved >= 0 AND input_tokens_used >= 0 "
            "AND output_tokens_used >= 0 AND total_tokens_used >= 0",
            name="ck_quota_windows_non_negative",
        ),
        sa.ForeignKeyConstraint(["project_id"], [f"{SCHEMA}.google_projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "model",
            "window_kind",
            "window_start",
            name="uq_quota_windows_project_model_kind_start",
        ),
        schema=SCHEMA,
    )
    op.create_table(
        "cooldowns",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("scope", sa.String(length=32), nullable=False),
        sa.Column("scope_key", sa.String(length=255), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=True),
        sa.Column("error_type", sa.String(length=64), nullable=False),
        sa.Column("cooldown_level", sa.Integer(), nullable=False),
        sa.Column("sleep_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("half_open_after", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=32), server_default=sa.text("'active'"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "scope IN ('project_model', 'key', 'proxy', 'binding', 'provider_model')",
            name="ck_cooldowns_scope",
        ),
        sa.CheckConstraint("status IN ('active', 'cleared', 'expired')", name="ck_cooldowns_status"),
        sa.CheckConstraint("cooldown_level >= 0", name="ck_cooldowns_level"),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )
    op.create_index(
        "uq_cooldowns_active_scope_null_model",
        "cooldowns",
        ["scope", "scope_key", "status"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("model IS NULL AND status = 'active'"),
    )
    op.create_index(
        "uq_cooldowns_active_scope_model",
        "cooldowns",
        ["scope", "scope_key", "model", "status"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("model IS NOT NULL AND status = 'active'"),
    )
    op.create_table(
        "route_attempts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.String(length=80), nullable=False),
        sa.Column("soybob_request_id", sa.String(length=80), nullable=True),
        sa.Column("source_service", sa.String(length=80), nullable=True),
        sa.Column("chat_id", sa.BigInteger(), nullable=True),
        sa.Column("telegram_message_id", sa.Integer(), nullable=True),
        sa.Column("project_id", sa.BigInteger(), nullable=True),
        sa.Column("api_key_id", sa.BigInteger(), nullable=True),
        sa.Column("proxy_id", sa.BigInteger(), nullable=True),
        sa.Column("binding_id", sa.BigInteger(), nullable=True),
        sa.Column("route_label", sa.String(length=80), nullable=True),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_type", sa.String(length=64), nullable=True),
        sa.Column("error_message_safe", sa.Text(), nullable=True),
        sa.Column("retryable", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("retry_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("estimated_input_tokens", sa.Integer(), nullable=True),
        sa.Column("reserved_input_tokens", sa.Integer(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("generation_id", sa.String(length=255), nullable=True),
        sa.Column("finish_reason", sa.String(length=64), nullable=True),
        sa.Column("provider_response_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "status IN ('leased', 'success', 'failed', 'skipped_no_route')",
            name="ck_route_attempts_status",
        ),
        sa.CheckConstraint("latency_ms IS NULL OR latency_ms >= 0", name="ck_route_attempts_latency"),
        sa.CheckConstraint("retry_count >= 0", name="ck_route_attempts_retry_count"),
        sa.ForeignKeyConstraint(["api_key_id"], [f"{SCHEMA}.api_keys.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["binding_id"], [f"{SCHEMA}.key_proxy_bindings.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["project_id"], [f"{SCHEMA}.google_projects.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["proxy_id"], [f"{SCHEMA}.proxy_endpoints.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )
    op.create_index("ix_route_attempts_created_at", "route_attempts", ["created_at"], schema=SCHEMA)
    op.create_index("ix_route_attempts_request_id", "route_attempts", ["request_id"], schema=SCHEMA)
    op.create_index(
        "ix_route_attempts_source_created",
        "route_attempts",
        ["source_service", "created_at"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_route_attempts_chat_created",
        "route_attempts",
        ["chat_id", "created_at"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_route_attempts_project_model_created",
        "route_attempts",
        ["project_id", "model", "created_at"],
        schema=SCHEMA,
    )
    op.create_index("ix_route_attempts_status_error", "route_attempts", ["status", "error_type"], schema=SCHEMA)


def downgrade() -> None:
    op.drop_index("ix_route_attempts_status_error", table_name="route_attempts", schema=SCHEMA)
    op.drop_index("ix_route_attempts_project_model_created", table_name="route_attempts", schema=SCHEMA)
    op.drop_index("ix_route_attempts_chat_created", table_name="route_attempts", schema=SCHEMA)
    op.drop_index("ix_route_attempts_source_created", table_name="route_attempts", schema=SCHEMA)
    op.drop_index("ix_route_attempts_request_id", table_name="route_attempts", schema=SCHEMA)
    op.drop_index("ix_route_attempts_created_at", table_name="route_attempts", schema=SCHEMA)
    op.drop_table("route_attempts", schema=SCHEMA)
    op.drop_index("uq_cooldowns_active_scope_model", table_name="cooldowns", schema=SCHEMA)
    op.drop_index("uq_cooldowns_active_scope_null_model", table_name="cooldowns", schema=SCHEMA)
    op.drop_table("cooldowns", schema=SCHEMA)
    op.drop_table("quota_windows", schema=SCHEMA)
    op.drop_table("model_limits", schema=SCHEMA)
    op.drop_index("uq_key_proxy_bindings_active_proxy", table_name="key_proxy_bindings", schema=SCHEMA)
    op.drop_index("uq_key_proxy_bindings_active_key", table_name="key_proxy_bindings", schema=SCHEMA)
    op.drop_table("key_proxy_bindings", schema=SCHEMA)
    op.drop_table("proxy_endpoints", schema=SCHEMA)
    op.drop_table("api_keys", schema=SCHEMA)
    op.drop_table("google_projects", schema=SCHEMA)
    op.execute(sa.text(f"DROP SCHEMA IF EXISTS {SCHEMA}"))
