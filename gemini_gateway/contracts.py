from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.provider_metadata import allowlist_provider_metadata_tree, allowlist_provider_specific_fields

GatewayErrorReason = Literal[
    "no_route",
    "cooldown_active",
    "rate_limited",
    "quota_exhausted",
    "auth_failed",
    "proxy_failed",
    "network_timeout",
    "provider_unavailable",
    "content_filtered",
    "invalid_response",
    "request_failed",
    "unauthorized",
    "bad_request",
]
TransportMode = Literal["proxy"]


def sanitize_gateway_provider_specific_fields(payload: Any) -> dict[str, Any]:
    """Оставляет только metadata, нужную для reasoning/tool continuity."""

    return allowlist_provider_specific_fields(payload)


def sanitize_gateway_choices(value: Any) -> list[dict[str, Any]]:
    """Удаляет произвольную provider metadata из success choices."""

    if not isinstance(value, list):
        return []
    return [allowlist_provider_metadata_tree(choice) for choice in value if isinstance(choice, dict)]


class GatewayRouteRequest(BaseModel):
    """Базовый запрос, который требует lease маршрута key+proxy."""

    model_config = ConfigDict(extra="allow")

    request_id: str = Field(min_length=1, max_length=80)
    soybob_request_id: str | None = Field(default=None, min_length=1, max_length=80)
    source_service: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=255)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=180.0)
    estimated_input_tokens: int | None = Field(default=None, ge=1)
    retry_count: int = Field(default=0, ge=0)
    chat_id: int | None = None
    telegram_message_id: int | None = None

    @field_validator("model", "source_service", "request_id")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("soybob_request_id")
    @classmethod
    def strip_optional_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


class GatewayChatRequest(GatewayRouteRequest):
    """OpenAI-compatible запрос с метаданными внутреннего роутинга."""

    messages: list[dict[str, Any]] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GatewayTTSRequest(GatewayRouteRequest):
    """Native Gemini TTS запрос с общим route lease."""

    text: str = Field(min_length=1, max_length=4000)
    voice_name: str = Field(default="Kore", min_length=1, max_length=80)


class GatewayEmbeddingInputPart(BaseModel):
    """Часть multimodal embedding-запроса."""

    model_config = ConfigDict(extra="allow")

    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_payload_for_type(self) -> "GatewayEmbeddingInputPart":
        if self.type == "text":
            if self.text is None or not self.text.strip():
                raise ValueError("text embedding part requires text")
            return self
        image_url = self.image_url if isinstance(self.image_url, dict) else {}
        url = image_url.get("url")
        if url is None or not str(url).strip():
            raise ValueError("image_url embedding part requires image_url.url")
        return self


class GatewayEmbeddingRequest(GatewayRouteRequest):
    """Native Gemini embedding запрос с общим route lease."""

    input: list[GatewayEmbeddingInputPart] = Field(min_length=1, max_length=8)
    dimensions: int = Field(default=1536)

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: int) -> int:
        if value not in {768, 1536, 3072}:
            raise ValueError("dimensions must be one of 768, 1536, 3072")
        return value


class GatewayTTSResponse(BaseModel):
    """Ответ Gemini TTS с аудио и метаданными маршрута."""

    model_config = ConfigDict(extra="allow")

    request_id: str
    model: str
    audio_base64: str = Field(min_length=1)
    audio_mime_type: str = Field(default="audio/wav", min_length=1)
    usage: dict[str, Any] = Field(default_factory=dict)
    route: GatewayRouteMetadata | dict[str, Any] = Field(default_factory=dict)
    generation_id: str | None = None
    finish_reason: str | None = None
    provider_specific_fields: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @field_validator("provider_specific_fields", mode="before")
    @classmethod
    def sanitize_provider_specific_fields(cls, value: Any) -> dict[str, Any]:
        return sanitize_gateway_provider_specific_fields(value)


class GatewayRouteMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    route_label: str
    project_label: str
    key_label: str
    proxy_label: str = Field(min_length=1)
    transport_mode: TransportMode = "proxy"


class GatewayChatResponse(BaseModel):
    """OpenAI-compatible ответ Gemini с метаданными маршрута."""

    model_config = ConfigDict(extra="allow")

    request_id: str
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any] = Field(default_factory=dict)
    route: GatewayRouteMetadata | dict[str, Any] = Field(default_factory=dict)
    generation_id: str | None = None
    finish_reason: str | None = None
    provider_specific_fields: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @field_validator("choices", mode="before")
    @classmethod
    def sanitize_choices(cls, value: Any) -> list[dict[str, Any]]:
        return sanitize_gateway_choices(value)

    @field_validator("provider_specific_fields", mode="before")
    @classmethod
    def sanitize_provider_specific_fields(cls, value: Any) -> dict[str, Any]:
        return sanitize_gateway_provider_specific_fields(value)

    @model_validator(mode="after")
    def fill_finish_reason_from_choices(self) -> "GatewayChatResponse":
        if self.generation_id is None and isinstance(self.raw_response.get("id"), str):
            self.generation_id = self.raw_response["id"]
        if self.finish_reason is not None:
            return self
        if not self.choices:
            return self
        first_choice = self.choices[0]
        if isinstance(first_choice, dict) and isinstance(first_choice.get("finish_reason"), str):
            self.finish_reason = first_choice["finish_reason"]
        return self


class GatewayEmbeddingResponse(BaseModel):
    """Ответ Gemini embeddings с вектором и метаданными маршрута."""

    model_config = ConfigDict(extra="allow")

    request_id: str
    model: str
    embedding: list[float] = Field(min_length=1)
    dimensions: int
    usage: dict[str, Any] = Field(default_factory=dict)
    route: GatewayRouteMetadata | dict[str, Any] = Field(default_factory=dict)
    generation_id: str | None = None
    finish_reason: str | None = None
    provider_specific_fields: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @field_validator("provider_specific_fields", mode="before")
    @classmethod
    def sanitize_provider_specific_fields(cls, value: Any) -> dict[str, Any]:
        return sanitize_gateway_provider_specific_fields(value)


GatewayProviderResponse = GatewayChatResponse | GatewayTTSResponse | GatewayEmbeddingResponse


class GatewayErrorResponse(BaseModel):
    request_id: str
    error: str
    reason: GatewayErrorReason
    error_code: str | None = None
    retryable: bool
    retry_after_seconds: int | None = Field(default=None, gt=0)
    quota_scope: Literal["minute", "day"] | None = None
    quota_reset_at: str | None = None
    eligible_routes_count: int | None = Field(default=None, ge=0)
    exhausted_routes_count: int | None = Field(default=None, ge=0)
    disabled_routes_count: int | None = Field(default=None, ge=0)
    cooldown_scope: str | None = None
    cooldown_level: int | None = Field(default=None, ge=0)
    sleep_until: str | None = None
    provider_reason: str | None = None
    provider_status_code: int | None = None
    route_label: str | None = None
    project_label: str | None = None
    key_label: str | None = None
    proxy_label: str | None = None
    transport_mode: TransportMode | None = None


class RouteCandidate(BaseModel):
    """Кандидат маршрута key+proxy для конкретной модели."""

    model_config = ConfigDict(extra="allow")

    binding_id: int | str
    project_id: int | str
    api_key_id: int | str = ""
    proxy_id: int | str | None = None
    api_key: str
    proxy_url: str | None = None
    model: str
    route_label: str
    project_label: str
    key_label: str
    proxy_label: str | None = None
    transport_mode: TransportMode = "proxy"
    requests_per_minute: int
    tokens_per_minute: int
    requests_per_day: int
    tokens_per_day: int | None = None
    minute_requests_used: int = 0
    minute_tokens_reserved: int = 0
    day_requests_used: int = 0
    day_tokens_reserved: int = 0
    cooldown_until: datetime | None = None
    half_open: bool = False
    priority: int = 100
    last_used_at: datetime | None = None


class RouteLease(BaseModel):
    """Аренда маршрута на один provider-вызов."""

    model_config = ConfigDict(extra="allow")

    attempt_id: int | str
    binding_id: int | str
    project_id: int | str
    api_key_id: int | str = ""
    proxy_id: int | str
    api_key: str
    proxy_url: str = Field(min_length=1)
    model: str
    route_label: str
    project_label: str
    key_label: str
    proxy_label: str = Field(min_length=1)
    transport_mode: TransportMode = "proxy"
    estimated_tokens: int
    leased_at: datetime


class SeedModelLimit(BaseModel):
    model: str = Field(min_length=1, max_length=255)
    requests_per_minute: int = Field(gt=0)
    tokens_per_minute: int = Field(gt=0)
    requests_per_day: int = Field(gt=0)
    tokens_per_day: int | None = Field(default=None, gt=0)


class SeedProject(BaseModel):
    label: str = Field(min_length=1, max_length=80)
    owner_name: str = Field(min_length=1, max_length=160)
    owner_contact: str | None = Field(default=None, max_length=255)
    google_project_ref: str | None = Field(default=None, max_length=255)
    tier: str = Field(default="free", min_length=1, max_length=64)
    model_limits: list[SeedModelLimit] = Field(min_length=1)


class SeedApiKey(BaseModel):
    project_label: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=80)
    api_key: str = Field(min_length=1)


class SeedProxy(BaseModel):
    label: str = Field(min_length=1, max_length=80)
    url: str | None = Field(default=None, exclude=True)
    scheme: Literal["http", "https"] = "http"
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(ge=1, le=65535)
    username: str | None = None
    password: str | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_parts_from_url(cls, raw_data: Any) -> Any:
        if not isinstance(raw_data, dict):
            return raw_data
        proxy_url = raw_data.get("url")
        if not proxy_url:
            return raw_data

        parsed = urlsplit(str(proxy_url).strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port is None:
            raise ValueError("proxy url must be http(s)://user:pass@host:port")

        data = dict(raw_data)
        data.setdefault("scheme", parsed.scheme)
        data.setdefault("host", parsed.hostname)
        data.setdefault("port", parsed.port)
        if parsed.username is not None:
            data.setdefault("username", unquote(parsed.username))
        if parsed.password is not None:
            data.setdefault("password", unquote(parsed.password))
        return data


class SeedBinding(BaseModel):
    label: str = Field(min_length=1, max_length=80)
    project_label: str | None = Field(default=None, min_length=1, max_length=80)
    api_key_label: str = Field(min_length=1, max_length=80)
    proxy_label: str | None = Field(default=None, min_length=1, max_length=80)
    transport_mode: TransportMode = "proxy"

    @field_validator("transport_mode", mode="before")
    @classmethod
    def reject_direct_transport_mode(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() == "direct":
            raise ValueError("direct bindings are disabled; use a proxy binding")
        return value


class SeedConfig(BaseModel):
    projects: list[SeedProject] = Field(min_length=1)
    api_keys: list[SeedApiKey] = Field(min_length=1)
    proxies: list[SeedProxy] = Field(default_factory=list)
    bindings: list[SeedBinding] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> "SeedConfig":
        projects = {project.label for project in self.projects}
        api_key_refs = {(key.project_label, key.label) for key in self.api_keys}
        api_key_label_counts = Counter(key.label for key in self.api_keys)
        proxies = {proxy.label for proxy in self.proxies}
        for key in self.api_keys:
            if key.project_label not in projects:
                raise ValueError(f"unknown project label: {key.project_label}")
        for binding in self.bindings:
            if binding.project_label is not None and binding.project_label not in projects:
                raise ValueError(f"unknown project label: {binding.project_label}")
            if binding.project_label is None:
                if api_key_label_counts[binding.api_key_label] == 0:
                    raise ValueError(f"unknown api key label: {binding.api_key_label}")
                if api_key_label_counts[binding.api_key_label] > 1:
                    raise ValueError(f"ambiguous api key label: {binding.api_key_label}")
            elif (binding.project_label, binding.api_key_label) not in api_key_refs:
                raise ValueError(f"unknown api key label for project: {binding.project_label}/{binding.api_key_label}")
            if binding.transport_mode == "direct":
                raise ValueError("direct bindings are disabled; use a proxy binding")
            if binding.proxy_label is None:
                raise ValueError("proxy binding requires proxy_label")
            if binding.proxy_label not in proxies:
                raise ValueError(f"unknown proxy label: {binding.proxy_label}")
        return self
