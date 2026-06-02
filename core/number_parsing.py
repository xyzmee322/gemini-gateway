from __future__ import annotations

from typing import Any


def parse_optional_int(value: Any) -> int | None:
    """Парсит необязательное целое число для audit/metrics, не принимая bool."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def parse_int_like(value: Any) -> int | None:
    """Парсит int-compatible значение, сохраняя семантику локальных media helpers."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_positive_int(value: Any) -> int | None:
    """Парсит положительное int-compatible значение, не принимая bool."""

    parsed = parse_int_like(value)
    if parsed is None:
        return None
    return parsed if parsed > 0 else None
