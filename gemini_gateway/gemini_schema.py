from __future__ import annotations

from typing import Any

_OMITTED_SCHEMA_KEYWORDS = frozenset({"maxItems", "multipleOf"})


def adapt_gemini_json_schema(value: Any) -> Any:
    """Создаёт совместимую с Gemini копию JSON Schema."""

    if isinstance(value, dict):
        adapted = {
            key: adapt_gemini_json_schema(item)
            for key, item in value.items()
            if key not in _OMITTED_SCHEMA_KEYWORDS
        }
        schema_types = adapted.get("type")
        if isinstance(schema_types, list) and len(schema_types) == 2 and "null" in schema_types:
            adapted["type"] = next(schema_type for schema_type in schema_types if schema_type != "null")
            adapted["nullable"] = True
        return adapted
    if isinstance(value, list):
        return [adapt_gemini_json_schema(item) for item in value]
    return value


__all__ = ["adapt_gemini_json_schema"]
