from __future__ import annotations

import re

_SECRET_PATTERNS = (
    re.compile(r"(api[_ -]?key\s*[=:]\s*)[^\s,;&]+", re.IGNORECASE),
    re.compile(r"(authorization\s*[:=]\s*bearer\s+)[^\s,;&]+", re.IGNORECASE),
    re.compile(r"((?:^|[\s,;])authorization\s*[:=]\s*)(?!\s*bearer\b)[^\r\n,;]+", re.IGNORECASE),
    re.compile(
        r"((?:^|[\s,;])"
        r"(?:[A-Za-z0-9_-]+[-_ ])*"
        r"(?:"
        r"api[-_ ]?key"
        r"|access[-_ ]?token"
        r"|refresh[-_ ]?token"
        r"|secret[-_ ]?token"
        r"|password"
        r"|cookie"
        r"|session[-_ ]?id"
        r"|token"
        r"|secret"
        r")"
        r"\s*[:=]\s*)[^\s,;&]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"((?:^|[\s,;])"
        r"(?:[A-Za-z0-9_-]+[-_ ])*"
        r"(?:"
        r"access[-_ ]?key"
        r"|encryption[-_ ]?key"
        r"|hmac[-_ ]?key"
        r"|private[-_ ]?key"
        r"|secret[-_ ]?key"
        r"|signing[-_ ]?key"
        r")"
        r"\s*[:=]\s*)[^\s,;&]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"([?&]"
        r"(?:"
        r"api[_-]?key"
        r"|apikey"
        r"|access[_-]?key"
        r"|accesskey"
        r"|access[_-]?token"
        r"|client[_-]?secret"
        r"|encryption[_-]?key"
        r"|hmac[_-]?key"
        r"|id[_-]?token"
        r"|private[_-]?key"
        r"|refresh[_-]?token"
        r"|secret[_-]?key"
        r"|session[_-]?token"
        r"|signing[_-]?key"
        r"|authorization"
        r"|auth"
        r"|key"
        r"|password"
        r"|secret"
        r"|signature"
        r"|sig"
        r"|token"
        r"|awsaccesskeyid"
        r"|x-amz-[A-Za-z0-9_-]+"
        r"|x-goog-[A-Za-z0-9_-]+"
        r")=)[^&#\s]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"(([\"'])?"
        r"(?:"
        r"(?:[A-Za-z0-9_-]+[_ -])?api[_ -]?key"
        r"|(?:[A-Za-z0-9_-]+[_ -])?access[_ -]?key"
        r"|authorization"
        r"|(?:[A-Za-z0-9_-]+[_ -])?encryption[_ -]?key"
        r"|(?:[A-Za-z0-9_-]+[_ -])?hmac[_ -]?key"
        r"|(?:[A-Za-z0-9_-]+[_ -])?private[_ -]?key"
        r"|(?:[A-Za-z0-9_-]+[_ -])?secret[_ -]?key"
        r"|(?:[A-Za-z0-9_-]+[_ -])?signing[_ -]?key"
        r"|(?:[A-Za-z0-9_-]+[_ -])?token"
        r"|(?:[A-Za-z0-9_-]+[_ -])?secret"
        r"|password"
        r"|cookie"
        r"|session[_ -]?id"
        r")"
        r"\2\s*:\s*([\"']))[^\"']+",
        re.IGNORECASE,
    ),
    re.compile(r"(api\.telegram\.org/(?:file/)?bot)[^/\s?&#]+", re.IGNORECASE),
    re.compile(r"\bAIza[0-9A-Za-z_-]{8,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bSECRET(?:_[A-Z0-9]+)+\b(?!\s*[\"']?\s*[:=])", re.IGNORECASE),
    re.compile(r"://([^:/@\s]+):([^/@\s]+)@", re.IGNORECASE),
)


def redact_secrets_in_text(value: str) -> str:
    """Маскирует секреты внутри произвольного текста для логов и audit."""
    safe_value = value
    for pattern in _SECRET_PATTERNS:
        safe_value = pattern.sub(_mask_secret_match, safe_value)
    return safe_value


def _mask_secret_match(match: re.Match[str]) -> str:
    if match.lastindex and match.re.pattern.startswith("://"):
        return "://***:***@"
    if match.lastindex:
        return f"{match.group(1)}***"
    return "***"
