"""
Shogunet input hygiene
======================

Local, dependency-free text sanitation mirroring the ShugoCore security
primitives' contracts (``security.sanitize_text`` / ``security.redact``).
When ShugoCore is importable, ``shugocore_bridge`` prefers its hardened
versions; these equivalents keep Shogunet independently testable and
installable.
"""

import math

MAX_TEXT_LENGTH = 2048

_SECRET_KEYS = {"key", "token", "secret", "password", "passwd",
                "authorization", "api_key", "apikey", "credential"}


def sanitize_text(text: object, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Stringify, replace non-printable characters with spaces, hard-truncate.

    This is the last line of defense before any string crosses the wire or
    enters memory: control characters can smuggle delimiters into downstream
    loggers and databases, so they never survive ingress/egress.
    """
    raw = "" if text is None else str(text)
    cleaned = "".join(ch if ch.isprintable() else " " for ch in raw)
    return cleaned[: max(0, int(max_length))]


def redact(value: object) -> object:
    """Shallow-recursive redaction of secret-looking dictionary keys."""
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            if str(key).lower() in _SECRET_KEYS:
                out[key] = "[REDACTED]"
            else:
                out[key] = redact(val)
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def clamp_int(value: object, default: int, lo: int, hi: int) -> int:
    """Best-effort bounded integer coercion (never raises)."""
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def clamp_float(value: object, default: float, lo: float, hi: float) -> float:
    """Best-effort bounded float coercion; NaN/Inf collapse to default."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(num):
        return default
    return max(lo, min(hi, num))
