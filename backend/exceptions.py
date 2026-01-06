"""
Custom Exceptions for ShuttleSense
Provides structured error handling with error codes and HTTP status mapping.
"""

from typing import Optional, Dict, Any


class ShuttleSenseException(Exception):
    """Base exception for all ShuttleSense errors"""
    
    def __init__(
        self,
        message: str,
        code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to API response format"""
        result = {
            "error": self.code,
            "detail": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


# =============================================================================
# Authentication & Authorization Errors (401, 403)
# =============================================================================

class AuthenticationError(ShuttleSenseException):
    """Raised when authentication fails"""
    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, "AUTHENTICATION_REQUIRED", 401)


class TokenExpiredError(ShuttleSenseException):
    """Raised when JWT token has expired"""
    def __init__(self, message: str = "Token has expired"):
        super().__init__(message, "TOKEN_EXPIRED", 401)


class InvalidTokenError(ShuttleSenseException):
    """Raised when JWT token is invalid"""
    def __init__(self, message: str = "Invalid token"):
        super().__init__(message, "INVALID_TOKEN", 401)


class AuthorizationError(ShuttleSenseException):
    """Raised when user lacks permission"""
    def __init__(self, message: str = "Permission denied"):
        super().__init__(message, "PERMISSION_DENIED", 403)


# =============================================================================
# Resource Errors (404, 409)
# =============================================================================

class SessionNotFound(ShuttleSenseException):
    """Raised when session doesn't exist"""
    def __init__(self, session_id: str):
        super().__init__(
            f"Session not found: {session_id}",
            "SESSION_NOT_FOUND",
            404,
            {"session_id": session_id}
        )


class VideoNotFound(ShuttleSenseException):
    """Raised when video file doesn't exist"""
    def __init__(self, video_id: str):
        super().__init__(
            f"Video not found: {video_id}",
            "VIDEO_NOT_FOUND",
            404,
            {"video_id": video_id}
        )


class ResourceConflict(ShuttleSenseException):
    """Raised when resource already exists or conflicts"""
    def __init__(self, message: str, resource_type: str):
        super().__init__(
            message,
            "RESOURCE_CONFLICT",
            409,
            {"resource_type": resource_type}
        )


# =============================================================================
# Validation Errors (400, 413, 415, 422)
# =============================================================================

class ValidationError(ShuttleSenseException):
    """Raised when input validation fails"""
    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(message, "VALIDATION_ERROR", 400, details)


class InvalidVideoFormat(ShuttleSenseException):
    """Raised when video format is not supported"""
    def __init__(self, content_type: str, allowed_types: list):
        super().__init__(
            f"Invalid video format: {content_type}. Allowed: {', '.join(allowed_types)}",
            "INVALID_VIDEO_FORMAT",
            415,
            {"content_type": content_type, "allowed_types": allowed_types}
        )


class FileTooLarge(ShuttleSenseException):
    """Raised when uploaded file exceeds size limit"""
    def __init__(self, file_size_mb: float, max_size_mb: int):
        super().__init__(
            f"File too large: {file_size_mb:.1f}MB (max {max_size_mb}MB)",
            "FILE_TOO_LARGE",
            413,
            {"file_size_mb": file_size_mb, "max_size_mb": max_size_mb}
        )


class InsufficientPoseData(ShuttleSenseException):
    """Raised when not enough poses are detected for analysis"""
    def __init__(self, message: str = "Insufficient pose data for analysis"):
        super().__init__(message, "INSUFFICIENT_POSE_DATA", 422)


# =============================================================================
# Processing Errors (422, 500)
# =============================================================================

class VideoProcessingError(ShuttleSenseException):
    """Raised when video processing fails"""
    def __init__(self, message: str, stage: Optional[str] = None):
        details = {"stage": stage} if stage else {}
        super().__init__(message, "VIDEO_PROCESSING_ERROR", 422, details)


class PoseExtractionError(ShuttleSenseException):
    """Raised when pose extraction fails"""
    def __init__(self, message: str):
        super().__init__(message, "POSE_EXTRACTION_ERROR", 422, {"stage": "pose_extraction"})


class FeatureComputationError(ShuttleSenseException):
    """Raised when feature computation fails"""
    def __init__(self, message: str):
        super().__init__(message, "FEATURE_COMPUTATION_ERROR", 422, {"stage": "feature_computation"})


class ReportGenerationError(ShuttleSenseException):
    """Raised when report generation fails"""
    def __init__(self, message: str):
        super().__init__(message, "REPORT_GENERATION_ERROR", 500, {"stage": "report_generation"})


class ChatError(ShuttleSenseException):
    """Raised when chat/AI processing fails"""
    def __init__(self, message: str):
        super().__init__(message, "CHAT_ERROR", 500, {"stage": "chat"})


# =============================================================================
# Resource Exhaustion Errors (503, 507)
# =============================================================================

class ResourceExhausted(ShuttleSenseException):
    """Raised when system resources are exhausted"""
    def __init__(self, resource: str, current: Optional[str] = None, limit: Optional[str] = None):
        details = {"resource": resource}
        if current:
            details["current"] = current
        if limit:
            details["limit"] = limit
        super().__init__(
            f"Resource exhausted: {resource}",
            "RESOURCE_EXHAUSTED",
            507,
            details
        )


class InsufficientDiskSpace(ShuttleSenseException):
    """Raised when disk space is insufficient"""
    def __init__(self, required_mb: float, available_mb: float):
        super().__init__(
            f"Insufficient disk space: need {required_mb:.1f}MB, have {available_mb:.1f}MB",
            "INSUFFICIENT_DISK_SPACE",
            507,
            {"required_mb": required_mb, "available_mb": available_mb}
        )


class RateLimitExceeded(ShuttleSenseException):
    """Raised when rate limit is exceeded"""
    def __init__(self, limit: str, retry_after: Optional[int] = None):
        details = {"limit": limit}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(
            f"Rate limit exceeded: {limit}",
            "RATE_LIMIT_EXCEEDED",
            429,
            details
        )


class ServiceUnavailable(ShuttleSenseException):
    """Raised when service is temporarily unavailable"""
    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(message, "SERVICE_UNAVAILABLE", 503)
