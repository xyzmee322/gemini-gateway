# syntax=docker/dockerfile:1.7
ARG UV_VERSION=0.11.17

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.12-slim

ARG UV_DEFAULT_INDEX=https://pypi.org/simple

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/
COPY --chown=appuser:appuser pyproject.toml alembic.ini ./
COPY --chown=appuser:appuser core ./core
COPY --chown=appuser:appuser gemini_gateway ./gemini_gateway
COPY --chown=appuser:appuser migrations ./migrations
COPY --chown=appuser:appuser scripts ./scripts

RUN UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX}" \
    UV_LINK_MODE=copy \
    uv pip install --system .

USER appuser

CMD ["uvicorn", "gemini_gateway.main:app", "--host", "0.0.0.0", "--port", "8010", "--no-access-log"]
