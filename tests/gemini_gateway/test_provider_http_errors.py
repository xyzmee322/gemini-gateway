from __future__ import annotations

import httpx

from gemini_gateway.provider_http_errors import build_gateway_error_from_response


def test_build_gateway_error_from_response_preserves_safe_retry_metadata() -> None:
    response = httpx.Response(
        429,
        headers={"Retry-After": "15"},
        json={"error": {"message": "quota failed for AIza-raw-secret"}},
    )

    error = build_gateway_error_from_response(
        response=response,
        request_id="req-http-error",
        supports_content_filter=True,
    )

    assert error.reason == "rate_limited"
    assert error.retryable is True
    assert error.provider_status_code == 429
    assert error.status_code == 429
    assert error.retry_after_seconds == 15
    assert error.request_id == "req-http-error"
    assert "AIza-raw-secret" not in str(error.provider_message_safe)


def test_build_gateway_error_from_response_respects_content_filter_capability() -> None:
    response = httpx.Response(400, json={"error": {"message": "blocked by safety policy"}})

    supported = build_gateway_error_from_response(
        response=response,
        request_id="req-supported",
        supports_content_filter=True,
    )
    unsupported = build_gateway_error_from_response(
        response=response,
        request_id="req-unsupported",
        supports_content_filter=False,
    )

    assert supported.reason == "content_filtered"
    assert unsupported.reason == "invalid_response"
