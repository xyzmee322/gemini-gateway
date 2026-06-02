from __future__ import annotations

from typing import Any


def build_wide_event(
    *,
    event: str,
    service: str,
    environment: str | None = None,
    status: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": event,
        "service": service,
    }
    if environment is not None:
        payload["environment"] = environment
    if status is not None:
        payload["status"] = status
    payload.update(fields)
    return payload


def build_chat_batch_event(batch: Any, *, environment: str) -> dict[str, Any]:
    events = list(batch.events)
    message_ids = [event.message_id for event in events]
    threadchain_ids = [
        event.threadchain_id
        for event in events
        if getattr(event, "threadchain_id", None) is not None
    ]
    return build_wide_event(
        event="chat_batch_processed",
        service="worker",
        environment=environment,
        status="started",
        chat_id=batch.chat_id,
        message_ids=message_ids,
        telegram_message_id=message_ids[-1] if message_ids else None,
        threadchain_ids=threadchain_ids,
        threadchain_id=threadchain_ids[-1] if threadchain_ids else None,
        events_count=len(events),
        context_messages=0,
        gate_decision=None,
        gate_confidence=None,
        actions_count=0,
        actions_summary=[],
        executed_messages_count=0,
        internal_tools_count=0,
        error_type=None,
        duration_ms=0,
        context_ms=0,
        gate_ms=0,
        gate_decision_normalized=None,
        gate_confidence_threshold=None,
        gate_wake_states=[],
        gate_reflection_length=0,
        main_ms=0,
        execution_ms=0,
    )
