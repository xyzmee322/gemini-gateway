from __future__ import annotations

from typing import Any

from core.number_parsing import parse_optional_int


def first_string_value(payload: dict[str, Any], *keys: str) -> str | None:
    """Возвращает первую строку из payload по списку ключей."""

    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def first_int_value(payload: dict[str, Any], *keys: str) -> int | None:
    """Возвращает первое целое значение, не принимая bool за int."""

    value = next((payload.get(key) for key in keys if payload.get(key) is not None), None)
    return parse_optional_int(value)
