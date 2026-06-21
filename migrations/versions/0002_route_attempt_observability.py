"""Add route attempt provider observability columns.

Revision ID: 0002_route_attempt_observability
Revises: 0001_gateway_schema
Create Date: 2026-06-20 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_route_attempt_observability"
down_revision = "0001_gateway_schema"
branch_labels = None
depends_on = None

SCHEMA = "gemini_gateway"

_COLUMNS = (
    "operation_type",
    "payload_kind",
    "request_bytes",
    "response_bytes",
    "media_count",
    "image_count",
    "provider_total_ms",
    "request_prepare_ms",
    "response_headers_ms",
    "response_body_ms",
    "response_parse_ms",
    "timeout_kind",
    "timeout_stage",
)


def upgrade() -> None:
    op.add_column("route_attempts", sa.Column("operation_type", sa.String(length=32), nullable=True), schema=SCHEMA)
    op.add_column("route_attempts", sa.Column("payload_kind", sa.String(length=32), nullable=True), schema=SCHEMA)
    op.add_column("route_attempts", sa.Column("request_bytes", sa.Integer(), nullable=True), schema=SCHEMA)
    op.add_column("route_attempts", sa.Column("response_bytes", sa.Integer(), nullable=True), schema=SCHEMA)
    op.add_column("route_attempts", sa.Column("media_count", sa.Integer(), nullable=True), schema=SCHEMA)
    op.add_column("route_attempts", sa.Column("image_count", sa.Integer(), nullable=True), schema=SCHEMA)
    op.add_column("route_attempts", sa.Column("provider_total_ms", sa.Integer(), nullable=True), schema=SCHEMA)
    op.add_column("route_attempts", sa.Column("request_prepare_ms", sa.Integer(), nullable=True), schema=SCHEMA)
    op.add_column("route_attempts", sa.Column("response_headers_ms", sa.Integer(), nullable=True), schema=SCHEMA)
    op.add_column("route_attempts", sa.Column("response_body_ms", sa.Integer(), nullable=True), schema=SCHEMA)
    op.add_column("route_attempts", sa.Column("response_parse_ms", sa.Integer(), nullable=True), schema=SCHEMA)
    op.add_column("route_attempts", sa.Column("timeout_kind", sa.String(length=64), nullable=True), schema=SCHEMA)
    op.add_column("route_attempts", sa.Column("timeout_stage", sa.String(length=64), nullable=True), schema=SCHEMA)
    with op.get_context().autocommit_block():
        op.create_index(
            "ix_route_attempts_proxy_created",
            "route_attempts",
            ["proxy_id", "created_at"],
            schema=SCHEMA,
            postgresql_concurrently=True,
        )
        op.create_index(
            "ix_route_attempts_operation_created",
            "route_attempts",
            ["operation_type", "created_at"],
            schema=SCHEMA,
            postgresql_concurrently=True,
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index(
            "ix_route_attempts_operation_created",
            table_name="route_attempts",
            schema=SCHEMA,
            postgresql_concurrently=True,
        )
        op.drop_index(
            "ix_route_attempts_proxy_created",
            table_name="route_attempts",
            schema=SCHEMA,
            postgresql_concurrently=True,
        )
    for column_name in reversed(_COLUMNS):
        op.drop_column("route_attempts", column_name, schema=SCHEMA)
