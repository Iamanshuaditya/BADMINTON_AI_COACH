"""
Performance Monitoring Middleware for ShuttleSense
Tracks request duration and logs performance metrics.
"""

import time
import logging
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Middleware that tracks request performance and adds:
    - Request timing
    - Correlation IDs for request tracing
    - Performance logging
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:8])
        
        # Start timing
        start_time = time.perf_counter()
        
        # Store correlation ID in request state
        request.state.correlation_id = correlation_id
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Add headers
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        
        # Log performance (skip health checks for cleaner logs)
        if not request.url.path.startswith("/health"):
            log_level = logging.WARNING if duration_ms > 5000 else logging.INFO
            
            logger.log(
                log_level,
                f"{request.method} {request.url.path} - {response.status_code} ({duration_ms:.2f}ms)",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                    "slow": duration_ms > 5000
                }
            )
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Detailed request/response logging middleware.
    Use sparingly in production due to overhead.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Log request
        logger.debug(
            f"Request: {request.method} {request.url}",
            extra={
                "headers": dict(request.headers),
                "query_params": dict(request.query_params),
                "client_ip": request.client.host if request.client else None
            }
        )
        
        response = await call_next(request)
        
        # Log response
        logger.debug(
            f"Response: {response.status_code}",
            extra={
                "status_code": response.status_code,
                "headers": dict(response.headers)
            }
        )
        
        return response
