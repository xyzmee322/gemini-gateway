from __future__ import annotations

from typing import Any


def parse_optional_string(value: Any) -> str | None:
    """Возвращает stripped string или None для пустого значения."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None
