from __future__ import annotations

from typing import Any

import pytest

from gemini_gateway import value_extractors


def test_first_int_value_delegates_present_value_to_shared_int_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    def fake_parse_optional_int(value: Any) -> int | None:
        calls.append(value)
        return 42

    monkeypatch.setattr(value_extractors, "parse_optional_int", fake_parse_optional_int)

    assert (
        value_extractors.first_int_value({"missing": None, "tokens": "7"}, "missing", "tokens")
        == 42
    )
    assert calls == ["7"]


def test_first_int_value_stops_on_first_present_value() -> None:
    assert value_extractors.first_int_value({"first": True, "second": 12}, "first", "second") is None
