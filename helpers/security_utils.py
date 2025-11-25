from __future__ import annotations

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from .config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS


def _format_limit_string() -> str | None:
    if RATE_LIMIT_REQUESTS <= 0 or RATE_LIMIT_WINDOW_SECONDS <= 0:
        return None

    window = RATE_LIMIT_WINDOW_SECONDS
    requests = RATE_LIMIT_REQUESTS

    if window % 3600 == 0:
        hours = window // 3600
        unit = "hour" if hours == 1 else "hours"
        return f"{requests}/{hours} {unit}"
    if window % 60 == 0:
        minutes = window // 60
        unit = "minute" if minutes == 1 else "minutes"
        return f"{requests}/{minutes} {unit}"

    unit = "second" if window == 1 else "seconds"
    return f"{requests}/{window} {unit}"


def _rate_limit_handler(request: Request, exc: Exception):
    if isinstance(exc, RateLimitExceeded):
        return _rate_limit_exceeded_handler(request, exc)
    raise exc


def configure_rate_limiting(app: FastAPI) -> None:
    """Attach SlowAPI rate limiting to the app if configured."""
    limit = _format_limit_string()
    if not limit:
        return
    limiter = Limiter(key_func=get_remote_address, default_limits=[limit])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    app.add_middleware(SlowAPIMiddleware)


__all__ = ["configure_rate_limiting"]
