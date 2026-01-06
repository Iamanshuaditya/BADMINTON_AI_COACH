"""
Middleware module for ShuttleSense
"""

from .error_handler import setup_error_handlers
from .performance import PerformanceMiddleware

__all__ = ["setup_error_handlers", "PerformanceMiddleware"]
