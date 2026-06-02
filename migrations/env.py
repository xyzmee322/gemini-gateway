from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

from gemini_gateway.config import get_gateway_settings
from gemini_gateway.db.models import GATEWAY_SCHEMA, GatewayBase

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_gateway_settings()
config.set_main_option("sqlalchemy.url", settings.postgres_sync_dsn)

target_metadata = GatewayBase.metadata


def include_name(name: str | None, type_: str, parent_names: dict[str, str | None]) -> bool:
    if type_ == "schema":
        return name == GATEWAY_SCHEMA
    return parent_names.get("schema_name") == GATEWAY_SCHEMA


def run_migrations_offline() -> None:
    context.configure(
        url=settings.postgres_sync_dsn,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_schemas=True,
        include_name=include_name,
        version_table="alembic_version",
        version_table_schema=GATEWAY_SCHEMA,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={"connect_timeout": 10},
    )

    with connectable.connect() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {GATEWAY_SCHEMA}"))
        connection.commit()
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            include_schemas=True,
            include_name=include_name,
            version_table="alembic_version",
            version_table_schema=GATEWAY_SCHEMA,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
