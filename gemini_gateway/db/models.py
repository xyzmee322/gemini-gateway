from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

GATEWAY_SCHEMA = "gemini_gateway"


class GatewayBase(DeclarativeBase):
    """Base class for Gemini gateway models."""


class GatewayTimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class GoogleProject(GatewayTimestampMixin, GatewayBase):
    __tablename__ = "google_projects"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'paused', 'disabled_needs_review')", name="ck_google_projects_status"),
        {"schema": GATEWAY_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    owner_name: Mapped[str] = mapped_column(String(160), nullable=False)
    owner_contact: Mapped[str | None] = mapped_column(String(255))
    google_project_ref: Mapped[str | None] = mapped_column(String(255))
    tier: Mapped[str] = mapped_column(String(64), default="free", server_default=text("'free'"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", server_default=text("'active'"), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)


class ApiKey(GatewayTimestampMixin, GatewayBase):
    __tablename__ = "api_keys"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'paused', 'disabled_needs_review')", name="ck_api_keys_status"),
        CheckConstraint("cooldown_level >= 0", name="ck_api_keys_cooldown_level"),
        UniqueConstraint("project_id", "label", name="uq_api_keys_project_label"),
        {"schema": GATEWAY_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey(f"{GATEWAY_SCHEMA}.google_projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    label: Mapped[str] = mapped_column(String(80), nullable=False)
    encrypted_api_key: Mapped[str] = mapped_column(Text, nullable=False)
    key_fingerprint: Mapped[str] = mapped_column(String(96), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", server_default=text("'active'"), nullable=False)
    cooldown_level: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)
    sleep_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_type: Mapped[str | None] = mapped_column(String(64))
    last_error_message_safe: Mapped[str | None] = mapped_column(Text)


class ProxyEndpoint(GatewayTimestampMixin, GatewayBase):
    __tablename__ = "proxy_endpoints"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'paused', 'disabled_needs_review')", name="ck_proxy_endpoints_status"),
        CheckConstraint("scheme IN ('http', 'https')", name="ck_proxy_endpoints_scheme"),
        CheckConstraint("port BETWEEN 1 AND 65535", name="ck_proxy_endpoints_port"),
        CheckConstraint("cooldown_level >= 0", name="ck_proxy_endpoints_cooldown_level"),
        {"schema": GATEWAY_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    scheme: Mapped[str] = mapped_column(String(16), default="http", server_default=text("'http'"), nullable=False)
    host: Mapped[str] = mapped_column(String(255), nullable=False)
    port: Mapped[int] = mapped_column(Integer, nullable=False)
    encrypted_username: Mapped[str | None] = mapped_column(Text)
    encrypted_password: Mapped[str | None] = mapped_column(Text)
    proxy_fingerprint: Mapped[str] = mapped_column(String(96), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", server_default=text("'active'"), nullable=False)
    cooldown_level: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)
    sleep_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_type: Mapped[str | None] = mapped_column(String(64))
    last_error_message_safe: Mapped[str | None] = mapped_column(Text)


class KeyProxyBinding(GatewayTimestampMixin, GatewayBase):
    __tablename__ = "key_proxy_bindings"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'paused', 'disabled_needs_review')",
            name="ck_key_proxy_bindings_status",
        ),
        CheckConstraint("transport_mode = 'proxy'", name="ck_key_proxy_bindings_transport_mode"),
        CheckConstraint("proxy_id IS NOT NULL", name="ck_key_proxy_bindings_transport_proxy"),
        CheckConstraint("consecutive_failures >= 0", name="ck_key_proxy_bindings_failures"),
        CheckConstraint("cooldown_level >= 0", name="ck_key_proxy_bindings_cooldown_level"),
        Index(
            "uq_key_proxy_bindings_active_key",
            "api_key_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
        Index(
            "uq_key_proxy_bindings_active_proxy",
            "proxy_id",
            unique=True,
            postgresql_where=text("status = 'active' AND proxy_id IS NOT NULL"),
        ),
        {"schema": GATEWAY_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    api_key_id: Mapped[int] = mapped_column(
        ForeignKey(f"{GATEWAY_SCHEMA}.api_keys.id", ondelete="CASCADE"),
        nullable=False,
    )
    proxy_id: Mapped[int] = mapped_column(
        ForeignKey(f"{GATEWAY_SCHEMA}.proxy_endpoints.id", ondelete="CASCADE"),
        nullable=False,
    )
    label: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    transport_mode: Mapped[str] = mapped_column(String(16), default="proxy", server_default=text("'proxy'"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", server_default=text("'active'"), nullable=False)
    cooldown_level: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)
    sleep_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_type: Mapped[str | None] = mapped_column(String(64))
    last_error_message_safe: Mapped[str | None] = mapped_column(Text)


class ModelLimit(GatewayTimestampMixin, GatewayBase):
    __tablename__ = "model_limits"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'paused')", name="ck_model_limits_status"),
        CheckConstraint(
            "requests_per_minute > 0 AND tokens_per_minute > 0 AND requests_per_day > 0",
            name="ck_model_limits_positive",
        ),
        UniqueConstraint("project_id", "model", name="uq_model_limits_project_model"),
        {"schema": GATEWAY_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey(f"{GATEWAY_SCHEMA}.google_projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    requests_per_minute: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_per_minute: Mapped[int] = mapped_column(Integer, nullable=False)
    requests_per_day: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_per_day: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="active", server_default=text("'active'"), nullable=False)


class QuotaWindow(GatewayTimestampMixin, GatewayBase):
    __tablename__ = "quota_windows"
    __table_args__ = (
        CheckConstraint("window_kind IN ('minute', 'day')", name="ck_quota_windows_kind"),
        CheckConstraint(
            "requests_used >= 0 AND input_tokens_reserved >= 0 AND input_tokens_used >= 0 "
            "AND output_tokens_used >= 0 AND total_tokens_used >= 0",
            name="ck_quota_windows_non_negative",
        ),
        UniqueConstraint(
            "project_id",
            "model",
            "window_kind",
            "window_start",
            name="uq_quota_windows_project_model_kind_start",
        ),
        {"schema": GATEWAY_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey(f"{GATEWAY_SCHEMA}.google_projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    window_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    requests_used: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)
    input_tokens_reserved: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)
    input_tokens_used: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)
    output_tokens_used: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)
    total_tokens_used: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)


class Cooldown(GatewayTimestampMixin, GatewayBase):
    __tablename__ = "cooldowns"
    __table_args__ = (
        CheckConstraint(
            "scope IN ('project_model', 'key', 'proxy', 'binding', 'provider_model')",
            name="ck_cooldowns_scope",
        ),
        CheckConstraint("status IN ('active', 'cleared', 'expired')", name="ck_cooldowns_status"),
        CheckConstraint("cooldown_level >= 0", name="ck_cooldowns_level"),
        Index(
            "uq_cooldowns_active_scope_null_model",
            "scope",
            "scope_key",
            "status",
            unique=True,
            postgresql_where=text("model IS NULL AND status = 'active'"),
        ),
        Index(
            "uq_cooldowns_active_scope_model",
            "scope",
            "scope_key",
            "model",
            "status",
            unique=True,
            postgresql_where=text("model IS NOT NULL AND status = 'active'"),
        ),
        {"schema": GATEWAY_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String(32), nullable=False)
    scope_key: Mapped[str] = mapped_column(String(255), nullable=False)
    model: Mapped[str | None] = mapped_column(String(255))
    error_type: Mapped[str] = mapped_column(String(64), nullable=False)
    cooldown_level: Mapped[int] = mapped_column(Integer, nullable=False)
    sleep_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    half_open_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", server_default=text("'active'"), nullable=False)


class RouteAttempt(GatewayTimestampMixin, GatewayBase):
    __tablename__ = "route_attempts"
    __table_args__ = (
        CheckConstraint(
            "status IN ('leased', 'success', 'failed', 'skipped_no_route')",
            name="ck_route_attempts_status",
        ),
        CheckConstraint("latency_ms IS NULL OR latency_ms >= 0", name="ck_route_attempts_latency"),
        CheckConstraint("retry_count >= 0", name="ck_route_attempts_retry_count"),
        Index("ix_route_attempts_created_at", "created_at"),
        Index("ix_route_attempts_request_id", "request_id"),
        Index("ix_route_attempts_source_created", "source_service", "created_at"),
        Index("ix_route_attempts_chat_created", "chat_id", "created_at"),
        Index("ix_route_attempts_project_model_created", "project_id", "model", "created_at"),
        Index("ix_route_attempts_status_error", "status", "error_type"),
        {"schema": GATEWAY_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(80), nullable=False)
    soybob_request_id: Mapped[str | None] = mapped_column(String(80))
    source_service: Mapped[str | None] = mapped_column(String(80))
    chat_id: Mapped[int | None] = mapped_column(BigInteger)
    telegram_message_id: Mapped[int | None] = mapped_column(Integer)
    project_id: Mapped[int | None] = mapped_column(ForeignKey(f"{GATEWAY_SCHEMA}.google_projects.id", ondelete="SET NULL"))
    api_key_id: Mapped[int | None] = mapped_column(ForeignKey(f"{GATEWAY_SCHEMA}.api_keys.id", ondelete="SET NULL"))
    proxy_id: Mapped[int | None] = mapped_column(ForeignKey(f"{GATEWAY_SCHEMA}.proxy_endpoints.id", ondelete="SET NULL"))
    binding_id: Mapped[int | None] = mapped_column(ForeignKey(f"{GATEWAY_SCHEMA}.key_proxy_bindings.id", ondelete="SET NULL"))
    route_label: Mapped[str | None] = mapped_column(String(80))
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(64))
    error_message_safe: Mapped[str | None] = mapped_column(Text)
    retryable: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"), nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    estimated_input_tokens: Mapped[int | None] = mapped_column(Integer)
    reserved_input_tokens: Mapped[int | None] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    generation_id: Mapped[str | None] = mapped_column(String(255))
    finish_reason: Mapped[str | None] = mapped_column(String(64))
    provider_response_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
