from __future__ import annotations

import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import UTC, datetime
from typing import Any

from core.error_taxonomy import public_message_for_reason
from core.secret_redaction import redact_secrets_in_text
from core.service_lifecycle import build_service_lifecycle_extra

_RESERVED_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}
_LOG_FILE_STEM_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_SECRET_LOG_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "token",
    "secret",
    "password",
    "cookie",
    "session_id",
}
_SECRET_LOG_KEY_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_access_token",
    "_refresh_token",
    "_token",
    "_secret",
    "_password",
    "_cookie",
    "_session_id",
)
_AIOGRAM_POLLING_FETCH_FAILED_PREFIX = "Failed to fetch updates - TelegramNetworkError"
_AIOGRAM_POLLING_RETRY_PREFIX = "Sleep for "
_AIOGRAM_POLLING_BOT_STARTED_PREFIX = "Run polling for bot "
_AIOGRAM_POLLING_BOT_STOPPED_PREFIX = "Polling stopped for bot "
_AIOGRAM_POLLING_BOT_PATTERN = re.compile(r"@(?P<username>\S+) id=(?P<bot_id>\d+) - '(?P<name>.*)'")
_UVICORN_SERVER_PROCESS_STARTED_PATTERN = re.compile(r"Started server process \[(?P<process_id>\d+)\]")
_UVICORN_SERVER_PROCESS_FINISHED_PATTERN = re.compile(r"Finished server process \[(?P<process_id>\d+)\]")
_UVICORN_SERVER_RUNNING_PATTERN = re.compile(r"Uvicorn running on (?P<bind_url>\S+)")
_UVICORN_STATIC_LIFECYCLE_EVENTS = {
    "Waiting for application startup.": (
        "uvicorn_application_startup_waiting",
        "starting",
        "Uvicorn application startup waiting",
    ),
    "Application startup complete.": (
        "uvicorn_application_startup_complete",
        "started",
        "Uvicorn application startup complete",
    ),
    "Shutting down": (
        "uvicorn_server_shutting_down",
        "stopping",
        "Uvicorn server shutting down",
    ),
    "Waiting for application shutdown.": (
        "uvicorn_application_shutdown_waiting",
        "stopping",
        "Uvicorn application shutdown waiting",
    ),
    "Application shutdown complete.": (
        "uvicorn_application_shutdown_complete",
        "stopped",
        "Uvicorn application shutdown complete",
    ),
}


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_secrets_in_text(record.getMessage()),
        }

        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_KEYS or key.startswith("_"):
                continue
            payload[key] = _redact_log_value(value, parent_key=key)

        if record.exc_info:
            payload["exception"] = redact_secrets_in_text(self.formatException(record.exc_info))

        return json.dumps(payload, ensure_ascii=False, default=_redact_json_default)


class _ThirdPartyLogNormalizationFilter(logging.Filter):
    def __init__(self, *, service_name: str, environment: str) -> None:
        super().__init__()
        self._service_name = service_name
        self._environment = environment

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "aiogram.dispatcher":
            _normalize_aiogram_dispatcher_record(
                record,
                service_name=self._service_name,
                environment=self._environment,
            )
        if record.name == "uvicorn.error":
            _normalize_uvicorn_error_record(
                record,
                service_name=self._service_name,
                environment=self._environment,
            )
        return True


def configure_logging(
    service_name: str,
    *,
    logs_dir: Path | str = "logs",
    environment: str | None = None,
) -> None:
    logs_dir = Path(logs_dir)
    raw_dir = logs_dir / "raw"
    log_environment = environment or os.getenv("ENVIRONMENT") or os.getenv("GEMINI_GATEWAY_ENVIRONMENT") or "development"

    formatter = JsonLogFormatter()
    third_party_normalizer = _ThirdPartyLogNormalizationFilter(
        service_name=service_name,
        environment=log_environment,
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(third_party_normalizer)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
    _route_loggers_to_root("uvicorn", "uvicorn.error")

    file_logging_enabled = False
    try:
        file_handler = _build_file_handler(
            logs_dir=logs_dir,
            raw_dir=raw_dir,
            service_name=service_name,
            formatter=formatter,
        )
        file_handler.addFilter(third_party_normalizer)
        root_logger.addHandler(file_handler)
        file_logging_enabled = True
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "File logging disabled because log path is not writable: %s",
            exc.__class__.__name__,
            extra={
                "event": "file_logging_disabled",
                "service": service_name,
                "environment": log_environment,
                "status": "fallback",
                "reason": "file_logging_unwritable",
                "retryable": False,
                "failed_stage": "file_logging_setup",
                "error_type": type(exc).__name__,
                "error_message": public_message_for_reason("request_failed"),
            },
        )

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured",
        extra=build_service_lifecycle_extra(
            service=service_name,
            environment=log_environment,
            event="logging_configured",
            status="configured",
            file_logging_enabled=file_logging_enabled,
        ),
    )


def _route_loggers_to_root(*logger_names: str) -> None:
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        logger.propagate = True


def _normalize_aiogram_dispatcher_record(
    record: logging.LogRecord,
    *,
    service_name: str,
    environment: str,
) -> None:
    message = record.getMessage()
    if message == "Start polling":
        _normalize_aiogram_polling_lifecycle_record(
            record,
            event="telegram_polling_started",
            status="started",
            message="Telegram polling started",
            service_name=service_name,
            environment=environment,
        )
        return

    if message.startswith(_AIOGRAM_POLLING_BOT_STARTED_PREFIX):
        _normalize_aiogram_polling_lifecycle_record(
            record,
            event="telegram_polling_bot_started",
            status="started",
            message="Telegram polling bot started",
            service_name=service_name,
            environment=environment,
            raw_bot_identity=message.removeprefix(_AIOGRAM_POLLING_BOT_STARTED_PREFIX),
        )
        return

    if message == "Polling stopped":
        _normalize_aiogram_polling_lifecycle_record(
            record,
            event="telegram_polling_stopped",
            status="stopped",
            message="Telegram polling stopped",
            service_name=service_name,
            environment=environment,
        )
        return

    if message.startswith(_AIOGRAM_POLLING_BOT_STOPPED_PREFIX):
        _normalize_aiogram_polling_lifecycle_record(
            record,
            event="telegram_polling_bot_stopped",
            status="stopped",
            message="Telegram polling bot stopped",
            service_name=service_name,
            environment=environment,
            raw_bot_identity=message.removeprefix(_AIOGRAM_POLLING_BOT_STOPPED_PREFIX),
        )
        return

    if message.startswith(_AIOGRAM_POLLING_FETCH_FAILED_PREFIX):
        record.msg = "Telegram polling update fetch failed"
        record.args = ()
        _set_record_field(record, "event", "telegram_polling_update_fetch_failed")
        _set_record_field(record, "service", service_name)
        _set_record_field(record, "environment", environment)
        _set_record_field(record, "status", "error")
        _set_record_field(record, "reason", "telegram_polling_update_fetch_failed")
        _set_record_field(record, "retryable", True)
        _set_record_field(record, "failed_stage", "telegram_polling_update_fetch")
        _set_record_field(record, "error_type", "TelegramNetworkError")
        _set_record_field(record, "error_message", public_message_for_reason("network_timeout"))
        return

    if message.startswith(_AIOGRAM_POLLING_RETRY_PREFIX) and "try again" in message:
        record.msg = "Telegram polling retry scheduled"
        record.args = ()
        _set_record_field(record, "event", "telegram_polling_retry_scheduled")
        _set_record_field(record, "service", service_name)
        _set_record_field(record, "environment", environment)
        _set_record_field(record, "status", "retrying")
        _set_record_field(record, "reason", "telegram_polling_retry_scheduled")
        _set_record_field(record, "retryable", True)
        _set_record_field(record, "failed_stage", "telegram_polling_retry")
        _set_record_field(record, "error_message", public_message_for_reason("network_timeout"))


def _normalize_uvicorn_error_record(
    record: logging.LogRecord,
    *,
    service_name: str,
    environment: str,
) -> None:
    message = record.getMessage()

    process_started_match = _UVICORN_SERVER_PROCESS_STARTED_PATTERN.fullmatch(message)
    if process_started_match is not None:
        _normalize_uvicorn_lifecycle_record(
            record,
            event="uvicorn_server_process_started",
            status="started",
            message="Uvicorn server process started",
            service_name=service_name,
            environment=environment,
        )
        _set_record_field(record, "process_id", int(process_started_match.group("process_id")))
        return

    process_finished_match = _UVICORN_SERVER_PROCESS_FINISHED_PATTERN.fullmatch(message)
    if process_finished_match is not None:
        _normalize_uvicorn_lifecycle_record(
            record,
            event="uvicorn_server_process_finished",
            status="stopped",
            message="Uvicorn server process finished",
            service_name=service_name,
            environment=environment,
        )
        _set_record_field(record, "process_id", int(process_finished_match.group("process_id")))
        return

    running_match = _UVICORN_SERVER_RUNNING_PATTERN.match(message)
    if running_match is not None:
        _normalize_uvicorn_lifecycle_record(
            record,
            event="uvicorn_server_running",
            status="running",
            message="Uvicorn server running",
            service_name=service_name,
            environment=environment,
        )
        _set_record_field(record, "bind_url", running_match.group("bind_url"))
        return

    lifecycle_event = _UVICORN_STATIC_LIFECYCLE_EVENTS.get(message)
    if lifecycle_event is None:
        return

    event, status, normalized_message = lifecycle_event
    _normalize_uvicorn_lifecycle_record(
        record,
        event=event,
        status=status,
        message=normalized_message,
        service_name=service_name,
        environment=environment,
    )


def _normalize_uvicorn_lifecycle_record(
    record: logging.LogRecord,
    *,
    event: str,
    status: str,
    message: str,
    service_name: str,
    environment: str,
) -> None:
    record.msg = message
    record.args = ()
    if hasattr(record, "color_message"):
        delattr(record, "color_message")
    _set_record_field(record, "event", event)
    _set_record_field(record, "service", service_name)
    _set_record_field(record, "environment", environment)
    _set_record_field(record, "status", status)
    _set_record_field(record, "reason", event)


def _normalize_aiogram_polling_lifecycle_record(
    record: logging.LogRecord,
    *,
    event: str,
    status: str,
    message: str,
    service_name: str,
    environment: str,
    raw_bot_identity: str | None = None,
) -> None:
    record.msg = message
    record.args = ()
    _set_record_field(record, "event", event)
    _set_record_field(record, "service", service_name)
    _set_record_field(record, "environment", environment)
    _set_record_field(record, "status", status)
    _set_record_field(record, "reason", event)

    if raw_bot_identity is None:
        return

    match = _AIOGRAM_POLLING_BOT_PATTERN.search(raw_bot_identity)
    if match is None:
        return
    _set_record_field(record, "telegram_bot_username", match.group("username"))
    _set_record_field(record, "telegram_bot_id", int(match.group("bot_id")))
    _set_record_field(record, "telegram_bot_name", match.group("name"))


def _set_record_field(record: logging.LogRecord, key: str, value: Any) -> None:
    if not hasattr(record, key):
        setattr(record, key, value)


def _build_file_handler(
    *,
    logs_dir: Path,
    raw_dir: Path,
    service_name: str,
    formatter: logging.Formatter,
) -> logging.Handler:
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_handler = _rotating_file_handler(logs_dir / f"{_safe_log_file_stem(service_name)}.log")
    file_handler.setFormatter(formatter)
    return file_handler


def _rotating_file_handler(path: Path) -> RotatingFileHandler:
    return RotatingFileHandler(
        path,
        maxBytes=20 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )


def _safe_log_file_stem(service_name: str) -> str:
    normalized = _LOG_FILE_STEM_PATTERN.sub("-", service_name.strip()).strip(".-")
    return normalized or "service"


def _redact_log_value(value: Any, *, parent_key: str | None = None) -> Any:
    if _is_secret_log_key(parent_key):
        return "***"
    if isinstance(value, str):
        return redact_secrets_in_text(value)
    if isinstance(value, dict):
        return {key: _redact_log_value(item, parent_key=str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_log_value(item, parent_key=parent_key) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_log_value(item, parent_key=parent_key) for item in value)
    return value


def _redact_json_default(value: Any) -> str:
    return redact_secrets_in_text(str(value))


def _is_secret_log_key(key: str | None) -> bool:
    if key is None:
        return False
    normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in _SECRET_LOG_KEYS or normalized.endswith(_SECRET_LOG_KEY_SUFFIXES)
