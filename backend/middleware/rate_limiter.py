"""
Rate Limiting for ShuttleSense
Uses slowapi for request rate limiting by IP and user.
"""

import logging
from typing import Optional, Callable

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse

from config.settings import get_settings
from auth.jwt_handler import get_user_id_from_request

logger = logging.getLogger(__name__)


def get_rate_limit_key(request: Request) -> str:
    """
    Generate rate limit key based on user ID (if authenticated) or IP address.
    This allows for per-user rate limiting when authenticated.
    """
    # Try to get user ID from JWT
    user_id = get_user_id_from_request(request)
    
    if user_id:
        return f"user:{user_id}"
    
    # Fall back to IP address
    return get_remote_address(request)


def create_limiter() -> Limiter:
    """
    Create and configure the rate limiter.
    """
    settings = get_settings()
    
    return Limiter(
        key_func=get_rate_limit_key,
        default_limits=[settings.RATE_LIMIT_GLOBAL],
        enabled=settings.RATE_LIMIT_ENABLED,
        storage_uri="memory://",  # Use Redis in production: "redis://localhost:6379"
        strategy="fixed-window"
    )


# Global limiter instance
limiter = create_limiter()


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    Custom handler for rate limit exceeded errors.
    """
    logger.warning(
        f"Rate limit exceeded for {get_rate_limit_key(request)}",
        extra={
            "path": str(request.url.path),
            "method": request.method,
            "limit": str(exc.detail)
        }
    )
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "RATE_LIMIT_EXCEEDED",
            "detail": f"Rate limit exceeded: {exc.detail}",
            "path": str(request.url.path)
        },
        headers={
            "Retry-After": "3600",  # 1 hour
            "X-RateLimit-Limit": str(exc.detail)
        }
    )


def setup_rate_limiting(app) -> None:
    """
    Setup rate limiting on the FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    settings = get_settings()
    
    if not settings.RATE_LIMIT_ENABLED:
        logger.info("Rate limiting is disabled")
        return
    
    # Add limiter to app state
    app.state.limiter = limiter
    
    # Add exception handler
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    
    logger.info("Rate limiting enabled")


# Rate limit decorators for common limits
def limit_upload(func: Callable) -> Callable:
    """Decorator for upload endpoints"""
    settings = get_settings()
    return limiter.limit(settings.RATE_LIMIT_UPLOAD)(func)


def limit_analyze(func: Callable) -> Callable:
    """Decorator for analyze endpoints"""
    settings = get_settings()
    return limiter.limit(settings.RATE_LIMIT_ANALYZE)(func)


def limit_chat(func: Callable) -> Callable:
    """Decorator for chat endpoints"""
    settings = get_settings()
    return limiter.limit(settings.RATE_LIMIT_CHAT)(func)
