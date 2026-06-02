from __future__ import annotations

from typing import Any


def build_service_lifecycle_extra(
    *,
    service: str,
    environment: str,
    event: str,
    status: str,
    **fields: Any,
) -> dict[str, Any]:
    """Собирает базовые поля lifecycle-события сервиса."""

    return {
        "event": event,
        "service": service,
        "environment": environment,
        "status": status,
        "reason": event,
        **fields,
    }
