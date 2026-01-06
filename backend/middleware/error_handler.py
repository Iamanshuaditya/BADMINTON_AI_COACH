"""
Global Error Handlers for ShuttleSense
Provides consistent error response formatting.
"""

import logging
import traceback
from typing import Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from exceptions import ShuttleSenseException
from config.settings import get_settings

logger = logging.getLogger(__name__)


def setup_error_handlers(app: FastAPI) -> None:
    """
    Register global exception handlers on the FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    settings = get_settings()
    
    @app.exception_handler(ShuttleSenseException)
    async def shuttlesense_exception_handler(
        request: Request,
        exc: ShuttleSenseException
    ) -> JSONResponse:
        """Handle custom ShuttleSense exceptions"""
        logger.error(
            f"ShuttleSense error: {exc.code} - {exc.message}",
            extra={
                "code": exc.code,
                "status_code": exc.status_code,
                "path": str(request.url.path),
                "method": request.method,
                "details": exc.details
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.code,
                "detail": exc.message,
                "path": str(request.url.path),
                **({"details": exc.details} if exc.details else {})
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle standard HTTP exceptions"""
        logger.warning(
            f"HTTP error: {exc.status_code} - {exc.detail}",
            extra={
                "status_code": exc.status_code,
                "path": str(request.url.path),
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "detail": exc.detail,
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors"""
        errors = exc.errors()
        
        # Format validation errors
        formatted_errors = []
        for error in errors:
            formatted_errors.append({
                "field": ".".join(str(loc) for loc in error.get("loc", [])),
                "message": error.get("msg", "Validation error"),
                "type": error.get("type", "unknown")
            })
        
        logger.warning(
            f"Validation error on {request.url.path}",
            extra={
                "path": str(request.url.path),
                "method": request.method,
                "errors": formatted_errors
            }
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "VALIDATION_ERROR",
                "detail": "Request validation failed",
                "path": str(request.url.path),
                "validation_errors": formatted_errors
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions"""
        # Log full traceback
        logger.error(
            f"Unhandled exception: {type(exc).__name__} - {str(exc)}",
            exc_info=True,
            extra={
                "path": str(request.url.path),
                "method": request.method,
                "exception_type": type(exc).__name__
            }
        )
        
        # Only show detailed error in development
        detail = str(exc) if settings.DEBUG else "An unexpected error occurred"
        
        response_content = {
            "error": "INTERNAL_SERVER_ERROR",
            "detail": detail,
            "path": str(request.url.path)
        }
        
        # Include traceback in debug mode
        if settings.DEBUG:
            response_content["traceback"] = traceback.format_exc()
        
        return JSONResponse(
            status_code=500,
            content=response_content
        )
