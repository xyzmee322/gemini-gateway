from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from time import monotonic
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

_LOGGER = logging.getLogger(__name__)
_QUOTA_PER_USD = Decimal(500_000)
_MONEY_QUANTUM = Decimal("0.000000000001")


@dataclass(frozen=True)
class OpenLuxPricingSnapshot:
    """Тариф OpenLux, действовавший во время provider-вызова."""

    model: str
    fetched_at: datetime
    input_usd_per_token: Decimal
    output_usd_per_token: Decimal
    metadata: dict[str, Any]

    def as_json(self) -> dict[str, Any]:
        return {
            "version": "openlux_tariff_v1",
            "provider": "OpenLux",
            "model": self.model,
            "source": "openlux_public_pricing_api",
            "fetched_at": self.fetched_at.isoformat(),
            "currency": "USD",
            "rates": {
                "input_usd_per_token": _decimal_text(self.input_usd_per_token),
                "output_usd_per_token": _decimal_text(self.output_usd_per_token),
                "request_usd": "0",
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class _CacheEntry:
    snapshot: OpenLuxPricingSnapshot | None
    expires_at: float


class OpenLuxPricingCatalog:
    """Читает публичные NewAPI ratios OpenLux с TTL и stale fallback."""

    def __init__(
        self,
        *,
        group: str,
        base_url: str = "https://api.openlux.ai/v1",
        timeout_seconds: float = 5.0,
        cache_ttl_seconds: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._group = group.strip()
        self._cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._failure_ttl_seconds = min(max(self._cache_ttl_seconds, 1.0), 30.0)
        self._now = now or (lambda: datetime.now(UTC))
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            base_url=_openlux_origin(base_url),
            timeout=timeout_seconds,
            transport=transport,
            trust_env=False,
        )

    async def get_snapshot(self, model: str) -> OpenLuxPricingSnapshot | None:
        normalized_model = model.strip()
        if not normalized_model:
            return None
        cached = self._cache.get(normalized_model)
        now_monotonic = monotonic()
        if cached is not None and cached.expires_at > now_monotonic:
            return cached.snapshot

        async with self._lock:
            cached = self._cache.get(normalized_model)
            now_monotonic = monotonic()
            if cached is not None and cached.expires_at > now_monotonic:
                return cached.snapshot
            try:
                snapshot = await self._fetch_snapshot(normalized_model)
            except (httpx.HTTPError, InvalidOperation, KeyError, TypeError, ValueError) as error:
                _LOGGER.warning(
                    "openlux_pricing_refresh_failed",
                    extra={"error_type": type(error).__name__, "model": normalized_model},
                )
                stale_snapshot = None if cached is None else cached.snapshot
                self._cache[normalized_model] = _CacheEntry(
                    snapshot=stale_snapshot,
                    expires_at=now_monotonic + self._failure_ttl_seconds,
                )
                return stale_snapshot

            self._cache[normalized_model] = _CacheEntry(
                snapshot=snapshot,
                expires_at=now_monotonic + self._cache_ttl_seconds,
            )
            return snapshot

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _fetch_snapshot(self, model: str) -> OpenLuxPricingSnapshot | None:
        response = await self._client.get("/api/pricing")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping) or payload.get("success") is not True:
            return None
        resolved = _resolve_openlux_price(payload=payload, model=model, group_name=self._group)
        if resolved is None:
            return None
        model_price, group_ratio, completion_ratio, price_type = resolved
        input_rate = model_price * group_ratio / _QUOTA_PER_USD
        output_rate = input_rate * completion_ratio
        return OpenLuxPricingSnapshot(
            model=model,
            fetched_at=self._now(),
            input_usd_per_token=input_rate,
            output_usd_per_token=output_rate,
            metadata={
                "billing_formula": "newapi_quota_ratio_v1",
                "completion_ratio": _decimal_text(completion_ratio),
                "group": self._group,
                "group_ratio": _decimal_text(group_ratio),
                "model_price": _decimal_text(model_price),
                "price_type": price_type,
                "quota_per_usd": int(_QUOTA_PER_USD),
            },
        )


def calculate_openlux_cost(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    snapshot: OpenLuxPricingSnapshot,
) -> Decimal:
    """Считает USD по тем же NewAPI ratios, что применяет OpenLux."""

    cost = (
        Decimal(max(prompt_tokens, 0)) * snapshot.input_usd_per_token
        + Decimal(max(completion_tokens, 0)) * snapshot.output_usd_per_token
    )
    return cost.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def _resolve_openlux_price(
    *,
    payload: Mapping[str, Any],
    model: str,
    group_name: str,
) -> tuple[Decimal, Decimal, Decimal, int] | None:
    data = payload.get("data")
    if isinstance(data, Mapping):
        groups = _mapping(data.get("model_group"))
        group = _mapping(groups.get(group_name))
        model_prices = _mapping(group.get("ModelPrice"))
        model_price = _mapping(model_prices.get(model))
        price_type = _integer(model_price.get("priceType"))
        if price_type != 0:
            return None
        completion_ratios = _mapping(data.get("model_completion_ratio"))
        return (
            _non_negative_decimal(model_price.get("price")),
            _non_negative_decimal(group.get("GroupRatio")),
            _non_negative_decimal(completion_ratios.get(model, 1)),
            price_type,
        )

    if not isinstance(data, list):
        return None
    model_entry = next(
        (item for item in data if isinstance(item, Mapping) and item.get("model_name") == model),
        None,
    )
    if model_entry is None or model_entry.get("available") is False:
        return None
    enabled_groups = model_entry.get("enable_groups")
    if isinstance(enabled_groups, list) and group_name not in enabled_groups:
        return None
    price_type = _integer(model_entry.get("quota_type"))
    if price_type != 0:
        return None
    group_ratios = _mapping(payload.get("group_ratio"))
    return (
        _non_negative_decimal(model_entry.get("model_ratio")),
        _non_negative_decimal(group_ratios.get(group_name)),
        _non_negative_decimal(model_entry.get("completion_ratio", 1)),
        price_type,
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("OpenLux pricing payload must contain JSON objects")
    return value


def _non_negative_decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError("OpenLux pricing value must be a non-negative number")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("OpenLux pricing value must be a non-negative number")
    return result


def _integer(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("OpenLux pricing type must be an integer")
    result = int(value)
    if Decimal(str(value)) != Decimal(result):
        raise ValueError("OpenLux pricing type must be an integer")
    return result


def _openlux_origin(base_url: str) -> str:
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OpenLux base URL must be absolute")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _decimal_text(value: Decimal) -> str:
    normalized = format(value, "f").rstrip("0").rstrip(".")
    return normalized or "0"
