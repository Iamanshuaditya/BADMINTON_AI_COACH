"""
Structured Logging Configuration for ShuttleSense
Provides JSON-formatted logs for production with correlation IDs.
"""

import logging
import json
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from contextvars import ContextVar
import uuid

# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Get current correlation ID or generate new one"""
    cid = correlation_id_var.get()
    if not cid:
        cid = str(uuid.uuid4())[:8]
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current context"""
    correlation_id_var.set(correlation_id)


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    Outputs logs in JSON format for easy parsing by log aggregators.
    """
    
    # Fields to exclude from extra data
    RESERVED_ATTRS = {
        'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
        'funcName', 'levelname', 'levelno', 'lineno', 'module', 'msecs',
        'message', 'msg', 'name', 'pathname', 'process', 'processName',
        'relativeCreated', 'stack_info', 'thread', 'threadName',
        'taskName'  # Python 3.12+
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        # Base log data
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add correlation ID
        correlation_id = get_correlation_id()
        if correlation_id:
            log_data["correlation_id"] = correlation_id
        
        # Add location info for errors
        if record.levelno >= logging.WARNING:
            log_data["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith('_'):
                try:
                    # Ensure value is JSON serializable
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_data, default=str)


class PrettyFormatter(logging.Formatter):
    """
    Pretty formatter for development.
    Human-readable format with colors.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors"""
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET
        
        # Get correlation ID
        correlation_id = get_correlation_id()
        cid_str = f"[{correlation_id}] " if correlation_id else ""
        
        # Build message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = f"{color}{record.levelname:8}{reset}"
        
        message = f"{timestamp} {level} {cid_str}{record.name}: {record.getMessage()}"
        
        # Add extra fields
        extras = []
        for key, value in record.__dict__.items():
            if key not in JSONFormatter.RESERVED_ATTRS and not key.startswith('_'):
                extras.append(f"{key}={value}")
        
        if extras:
            message += f" | {', '.join(extras)}"
        
        # Add exception
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message


class CorrelationFilter(logging.Filter):
    """Filter that adds correlation ID to all log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None
) -> None:
    """
    Configure application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (for production)
        log_file: Optional file path for log output
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add correlation filter
    correlation_filter = CorrelationFilter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.addFilter(correlation_filter)
    
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(PrettyFormatter())
    
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.addFilter(correlation_filter)
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    root_logger.info(
        "Logging configured",
        extra={
            "level": level,
            "json_format": json_format,
            "log_file": log_file
        }
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    Convenience function that ensures proper setup.
    """
    return logging.getLogger(name)


# Convenience class for structured logging
class StructuredLogger:
    """
    Helper class for structured logging with context.
    Provides methods for logging with automatic extra fields.
    """
    
    def __init__(self, name: str, default_context: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
        self.default_context = default_context or {}
    
    def _log(self, level: int, message: str, **extra):
        """Internal log method"""
        context = {**self.default_context, **extra}
        self.logger.log(level, message, extra=context)
    
    def debug(self, message: str, **extra):
        self._log(logging.DEBUG, message, **extra)
    
    def info(self, message: str, **extra):
        self._log(logging.INFO, message, **extra)
    
    def warning(self, message: str, **extra):
        self._log(logging.WARNING, message, **extra)
    
    def error(self, message: str, exc_info: bool = False, **extra):
        self.logger.error(message, exc_info=exc_info, extra={**self.default_context, **extra})
    
    def critical(self, message: str, exc_info: bool = True, **extra):
        self.logger.critical(message, exc_info=exc_info, extra={**self.default_context, **extra})
    
    def with_context(self, **context) -> "StructuredLogger":
        """Create a new logger with additional context"""
        new_context = {**self.default_context, **context}
        return StructuredLogger(self.logger.name, new_context)


# Timer context manager for performance logging
class LogTimer:
    """Context manager for timing operations"""
    
    def __init__(self, logger: logging.Logger, operation: str, **extra):
        self.logger = logger
        self.operation = operation
        self.extra = extra
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        
        if exc_type:
            self.logger.error(
                f"{self.operation} failed after {duration_ms:.2f}ms",
                extra={**self.extra, "duration_ms": duration_ms, "error": str(exc_val)}
            )
        else:
            log_level = logging.WARNING if duration_ms > 5000 else logging.INFO
            self.logger.log(
                log_level,
                f"{self.operation} completed in {duration_ms:.2f}ms",
                extra={**self.extra, "duration_ms": duration_ms}
            )
        
        return False  # Don't suppress exceptions
