from __future__ import annotations

from datetime import UTC, datetime, timedelta

from gemini_gateway.cooldown import COOLDOWN_STEPS_SECONDS, CooldownPolicy


def test_cooldown_policy_escalates_by_ladder() -> None:
    now = datetime(2026, 5, 23, tzinfo=UTC)
    policy = CooldownPolicy()

    decision = policy.next_decision(reason="rate_limited", current_level=1, now=now)

    assert decision is not None
    assert decision.level == 2
    assert decision.seconds == COOLDOWN_STEPS_SECONDS[1]
    assert decision.sleep_until == now + timedelta(seconds=COOLDOWN_STEPS_SECONDS[1])


def test_cooldown_policy_uses_retry_after_when_longer() -> None:
    now = datetime(2026, 5, 23, tzinfo=UTC)
    policy = CooldownPolicy()

    decision = policy.next_decision(reason="quota_exhausted", current_level=0, retry_after_seconds=600, now=now)

    assert decision is not None
    assert decision.seconds == 600


def test_cooldown_policy_skips_non_route_errors() -> None:
    policy = CooldownPolicy()

    assert policy.next_decision(reason="content_filtered", current_level=0) is None


def test_auth_failed_disables_key_without_half_open() -> None:
    policy = CooldownPolicy()

    decision = policy.next_decision(reason="auth_failed", current_level=3)

    assert decision is not None
    assert decision.scopes == ("key",)
    assert decision.disable_needs_review is True
    assert decision.seconds == 0


def test_provider_unavailable_cooldown_is_capped() -> None:
    policy = CooldownPolicy(provider_unavailable_cap_seconds=900)

    decision = policy.next_decision(reason="provider_unavailable", current_level=5)

    assert decision is not None
    assert decision.scopes == ("binding",)
    assert decision.seconds == 900
