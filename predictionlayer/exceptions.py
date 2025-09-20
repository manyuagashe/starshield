"""
Custom exception classes for the StarShield prediction layer.

This module defines exception hierarchy and error handling patterns
for clean error propagation and user-friendly error responses.
"""

from typing import Dict, Any, Optional


class StarShieldException(Exception):
    """Base exception class for StarShield prediction layer."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "internal_error",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.details = details or {}
        super().__init__(message)


class ValidationError(StarShieldException):
    """Exception for request validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=400,
            error_type="validation_error",
            details=details,
        )


class UpstreamError(StarShieldException):
    """Exception for upstream API errors."""
    
    def __init__(
        self,
        message: str,
        upstream_service: str,
        upstream_status: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_details = details or {}
        error_details.update({
            "upstream_service": upstream_service,
            "upstream_status": upstream_status,
        })
        
        # Map upstream status to appropriate response status
        if upstream_status == 429:
            status_code = 429
            error_type = "rate_limited"
        elif upstream_status and 500 <= upstream_status < 600:
            status_code = 502
            error_type = "upstream_error"
        elif upstream_status and 400 <= upstream_status < 500:
            status_code = 502
            error_type = "upstream_client_error"
        else:
            status_code = 504
            error_type = "upstream_timeout"
        
        super().__init__(
            message=message,
            status_code=status_code,
            error_type=error_type,
            details=error_details,
        )


class CacheError(StarShieldException):
    """Exception for cache-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_type="cache_error",
            details=details,
        )


class ModelError(StarShieldException):
    """Exception for model computation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_type="model_error",
            details=details,
        )


class RateLimitExceeded(StarShieldException):
    """Exception for rate limit violations."""
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded", 
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_details = details or {}
        if retry_after:
            error_details["retry_after"] = retry_after
            
        super().__init__(
            message=message,
            status_code=429,
            error_type="rate_limit_exceeded",
            details=error_details,
        )


class CircuitBreakerOpen(StarShieldException):
    """Exception when circuit breaker is open."""
    
    def __init__(
        self, 
        message: str = "Service temporarily unavailable", 
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=503,
            error_type="circuit_breaker_open",
            details=details,
        )


class DataNotFound(StarShieldException):
    """Exception when requested data is not found."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=404,
            error_type="data_not_found",
            details=details,
        )