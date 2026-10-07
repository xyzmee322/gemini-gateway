from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest

from gemini_gateway.openlux_pricing import OpenLuxPricingCatalog, calculate_openlux_cost


@pytest.mark.asyncio
async def test_openlux_pricing_uses_account_group_ratios_from_public_catalog() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.openlux.ai/api/pricing"
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "model_group": {
                        "Anti-Gemini-1": {
                            "GroupRatio": 0.07353,
                            "ModelPrice": {
                                "gemini-3.8-flash": {"price": 0.375, "priceType": 0}
                            },
                        }
                    },
                    "model_completion_ratio": {"gemini-3.8-flash": 5},
                },
            },
        )

    catalog = OpenLuxPricingCatalog(
        base_url="https://api.openlux.ai/v1",
        group="Anti-Gemini-1",
        transport=httpx.MockTransport(handler),
        now=lambda: datetime(2026, 8, 27, tzinfo=UTC),
    )

    snapshot = await catalog.get_snapshot("gemini-3.8-flash")

    assert snapshot is not None
    assert snapshot.input_usd_per_token == Decimal("0.0000000551475")
    assert snapshot.output_usd_per_token == Decimal("0.0000002757375")
    assert snapshot.metadata == {
        "billing_formula": "newapi_quota_ratio_v1",
        "completion_ratio": "5",
        "group": "Anti-Gemini-1",
        "group_ratio": "0.07353",
        "model_price": "0.375",
        "price_type": 0,
        "quota_per_usd": 500000,
    }
    assert calculate_openlux_cost(
        prompt_tokens=259,
        completion_tokens=14,
        snapshot=snapshot,
    ) == Decimal("0.000018143528")


@pytest.mark.asyncio
async def test_openlux_pricing_failure_returns_stale_snapshot() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        del request
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": [
                        {
                            "model_name": "gemini-3.8-flash",
                            "available": True,
                            "enable_groups": ["Anti-Gemini-1"],
                            "quota_type": 0,
                            "model_ratio": 0.375,
                            "completion_ratio": 5,
                        }
                    ],
                    "group_ratio": {"Anti-Gemini-1": 0.07353},
                },
            )
        return httpx.Response(503)

    catalog = OpenLuxPricingCatalog(
        group="Anti-Gemini-1",
        cache_ttl_seconds=0,
        transport=httpx.MockTransport(handler),
    )

    first = await catalog.get_snapshot("gemini-3.8-flash")
    second = await catalog.get_snapshot("gemini-3.8-flash")

    assert first is not None
    assert second == first
    assert calls == 2
