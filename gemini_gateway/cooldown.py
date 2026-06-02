from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from gemini_gateway.contracts import GatewayErrorReason

COOLDOWN_STEPS_SECONDS = (
    60,
    5 * 60,
    15 * 60,
    60 * 60,
    3 * 60 * 60,
    6 * 60 * 60,
    12 * 60 * 60,
    24 * 60 * 60,
    2 * 24 * 60 * 60,
    7 * 24 * 60 * 60,
)

CooldownScope = Literal["project_model", "key", "proxy", "binding", "provider_model"]


@dataclass(frozen=True)
class CooldownDecision:
    reason: GatewayErrorReason
    scopes: tuple[CooldownScope, ...]
    level: int
    seconds: int
    sleep_until: datetime
    half_open_after: datetime
    disable_needs_review: bool = False


class CooldownPolicy:
    """Рассчитывает sleep без блокировки event loop."""

    def __init__(self, *, provider_unavailable_cap_seconds: int = 15 * 60) -> None:
        self._provider_unavailable_cap_seconds = provider_unavailable_cap_seconds

    def next_decision(
        self,
        *,
        reason: GatewayErrorReason,
        current_level: int,
        retry_after_seconds: int | None = None,
        now: datetime | None = None,
    ) -> CooldownDecision | None:
        if reason in {"content_filtered", "invalid_response"}:
            return None

        normalized_now = now or datetime.now(UTC)
        scopes = self.scopes_for_reason(reason)
        if reason == "auth_failed":
            return CooldownDecision(
                reason=reason,
                scopes=scopes,
                level=max(0, current_level),
                seconds=0,
                sleep_until=normalized_now,
                half_open_after=normalized_now,
                disable_needs_review=True,
            )

        next_level = max(0, current_level)
        ladder_seconds = COOLDOWN_STEPS_SECONDS[min(next_level, len(COOLDOWN_STEPS_SECONDS) - 1)]
        if reason == "provider_unavailable":
            ladder_seconds = min(ladder_seconds, self._provider_unavailable_cap_seconds)
        seconds = max(ladder_seconds, retry_after_seconds or 0)
        sleep_until = normalized_now + timedelta(seconds=seconds)
        return CooldownDecision(
            reason=reason,
            scopes=scopes,
            level=next_level + 1,
            seconds=seconds,
            sleep_until=sleep_until,
            half_open_after=sleep_until,
        )

    @staticmethod
    def scopes_for_reason(reason: GatewayErrorReason) -> tuple[CooldownScope, ...]:
        if reason in {"rate_limited", "quota_exhausted"}:
            return ("project_model",)
        if reason == "auth_failed":
            return ("key",)
        if reason in {"proxy_failed", "network_timeout"}:
            return ("proxy", "binding")
        if reason == "provider_unavailable":
            return ("binding",)
        return ("binding",)
